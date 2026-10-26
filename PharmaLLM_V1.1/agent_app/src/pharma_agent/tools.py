import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError
from sqlalchemy import or_, select

from pharma_agent.db import ToolJob, audit, new_id, now
from pharma_agent.errors import AppError
from pharma_agent.models import config_load
from pharma_agent.observability import digest, span, trace_id
from pharma_agent.schemas import (
    DraftInput,
    DraftResult,
    EmptyInput,
    JobQuery,
    JobsResult,
    PreferencesResult,
    ReferencesResult,
    SearchInput,
    SegmentationInput,
    SegmentationResult,
)
from pharma_agent.security import Actor, preferences, require_owned, require_scope


@dataclass(frozen=True)
class ToolContext:
    actor: Actor
    request_id: str
    run_id: str
    services: Any
    conversation_id: str | None = None
    asset_ids: tuple[str, ...] = ()
    segmentation_model_id: str | None = None
    execution_mode: str = "ask_for_approval"


class ToolAdapter(Protocol):
    name: str
    version: str
    description: str
    required_scope: str
    read_only: bool
    input_model: type[BaseModel]
    output_model: type[BaseModel]

    async def execute(self, args: BaseModel, context: ToolContext) -> BaseModel: ...


def job_dict(job):
    return {
        "id": job.id,
        "status": job.status,
        "tool_name": job.tool_name,
        "tool_version": job.tool_version,
        "result": job.result_json,
        "error_code": job.error_code,
        "created_at": job.created_at.isoformat(),
        "input": {
            k: v
            for k, v in job.input_json.items()
            if k in {"file_id", "file_name", "model_id", "task_type", "conversation_id", "execution_mode"}
        },
        "elapsed_seconds": max(
            0,
            int(((job.finished_at or now()) - (job.confirmed_at or job.created_at)).total_seconds()),
        ),
    }


async def create_draft(services, actor, title, key):
    require_scope(actor, "tasks:draft")
    async with services.sessions() as session, session.begin():
        # Serialize on user so repeated tool invocations cannot race the unique constraint.
        from pharma_agent.db import User

        current_user = await session.scalar(
            select(User).where(User.id == actor.id, User.tenant_id == actor.tenant_id).with_for_update()
        )
        if not current_user or current_user.status != "active":
            raise AppError("PERMISSION_DENIED", "账户已停用", 403)
        require_scope(Actor(current_user.id, current_user.tenant_id, current_user.role,
                            current_user.department_id), "tasks:draft")
        old = (
            await session.scalars(
                select(ToolJob).where(
                    ToolJob.tenant_id == actor.tenant_id,
                    ToolJob.user_id == actor.id,
                    ToolJob.tool_name == "mock_report",
                    ToolJob.idempotency_key == key,
                )
            )
        ).first()
        if old:
            if old.input_json.get("title") != title:
                raise AppError("IDEMPOTENCY_CONFLICT", "同一幂等键不能用于不同任务", 409)
            return old
        job = ToolJob(
            id=new_id("job"),
            tenant_id=actor.tenant_id,
            user_id=actor.id,
            tool_name="mock_report",
            tool_version="mock-1.1",
            status="draft",
            idempotency_key=key,
            input_json={"title": title},
            result_json={},
        )
        session.add(job)
        audit(session, actor, "job.draft", "job", job.id, "created", trace_id())
        await session.flush()
        return job


class GetMyJob:
    name, version = "get_my_job", "1.1"
    description = "查询指定任务；不指定 ID 时，查询当前会话的影像任务及当前用户最近的演示任务，最多五个。不接受用户身份参数。"
    required_scope, read_only = "jobs:read", True
    input_model, output_model = JobQuery, JobsResult

    async def execute(self, args, context):
        if args.job_id:
            job = await require_owned(context.services, ToolJob, context.actor, args.job_id)
            jobs = [job]
        else:
            async with context.services.sessions() as session:
                conversation_filter = (
                    or_(
                        ToolJob.tool_name != "segment_medical_volume",
                        ToolJob.input_json["conversation_id"].as_string() == context.conversation_id,
                    )
                    if context.conversation_id
                    else True
                )
                jobs = (
                    await session.scalars(
                        select(ToolJob)
                        .where(
                            ToolJob.tenant_id == context.actor.tenant_id,
                            ToolJob.user_id == context.actor.id,
                            conversation_filter,
                        )
                        .order_by(ToolJob.created_at.desc())
                        .limit(5)
                    )
                ).all()
        return JobsResult(jobs=[job_dict(job) for job in jobs])


class SearchReference:
    name, version = "search_medical_reference", "1.1"
    description = "检索公共及当前租户获授权参考资料，返回可追溯引用。"
    required_scope, read_only = "references:read", True
    input_model, output_model = SearchInput, ReferencesResult

    async def execute(self, args, context):
        if context.services.settings.enable_faults and "[[fault:tool_timeout]]" in args.query:
            raise AppError("TOOL_TIMEOUT", "模拟参考资料检索超时", 504, True)
        result = await context.services.retrieval.search(context.actor.tenant_id, args.query, args.top_k)
        if context.services.settings.enable_faults and "[[fault:tool_schema]]" in args.query:
            return {"unexpected": "injected"}
        return result


class GetPreference:
    name, version = "get_user_preference", "1.1"
    description = "读取当前用户明确同意保存的语言、格式和引用偏好。"
    required_scope, read_only = "preferences:read", True
    input_model, output_model = EmptyInput, PreferencesResult

    async def execute(self, args, context):
        return PreferencesResult(preferences=await preferences(context.services.sessions, context.actor))


class CreateTaskDraft:
    name, version = "create_task_draft", "1.1"
    description = "创建合成演示任务的待确认草稿。仅创建草稿，需用户在 API/页面中明确确认后才能入队。"
    required_scope, read_only = "tasks:draft", False
    input_model, output_model = DraftInput, DraftResult

    async def execute(self, args, context):
        key = "tool:" + context.run_id + ":" + digest(args.model_dump())[7:31]
        job = await create_draft(context.services, context.actor, args.title, key)
        return DraftResult(
            job_id=job.id, status=job.status, requires_confirmation=job.status in {"draft", "failed"}
        )


class SegmentMedicalVolume:
    name, version = "segment_medical_volume", "1.0"
    description = "根据用户要求，为本轮附件或当前会话已授权的三维影像建立分割任务；只使用支持所需结构的允许模型和任务类型。后端按用户选定模式确认或自动提交。二维图像不可使用。"
    required_scope, read_only = "tasks:draft", False
    input_model, output_model = SegmentationInput, SegmentationResult

    async def execute(self, args, context):
        from pharma_agent.segmentation import create_segmentation_draft

        job = await create_segmentation_draft(
            context.services,
            context.actor,
            args,
            context.conversation_id,
            "seg:" + context.run_id + ":" + digest(args.model_dump())[7:31],
            context.asset_ids,
            execution_mode=context.execution_mode,
        )
        if context.execution_mode == "full_access" and job.status == "draft":
            job = await context.services.queue.confirm(context.actor, job.id, automatic=True)
        return SegmentationResult(
            job_id=job.id,
            status=job.status,
            requires_confirmation=job.status in {"draft", "failed"},
            model_id=args.model_id,
            result_asset_id=job.result_json.get("asset_id"),
            labels=job.result_json.get("labels", {}),
            error_code=job.error_code,
        )


class ToolRegistry:
    def __init__(self, settings):
        self.config = config_load(settings.tool_config)
        adapters = [GetMyJob(), SearchReference(), GetPreference(), CreateTaskDraft(), SegmentMedicalVolume()]
        self.adapters = {
            tool.name: tool
            for tool in adapters
            if self.config["tools"].get(tool.name, {}).get("enabled", False)
        }
        unknown = set(self.config["tools"]) - {tool.name for tool in adapters}
        if unknown:
            raise ValueError("存在没有实现的工具配置：" + ",".join(sorted(unknown)))

    def descriptions(self, actor):
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_model.model_json_schema(),
            }
            for tool in self.adapters.values()
            if tool.required_scope in actor.scopes
        ]


class ToolGateway:
    def __init__(self, registry):
        self.registry = registry

    def validate(self, name, raw_args, context):
        tool = self.registry.adapters.get(name)
        if not tool:
            raise AppError("INVALID_TOOL_ARGS", "工具不在启用白名单中", 422)
        if tool.required_scope not in context.actor.scopes:
            raise AppError("PERMISSION_DENIED", "没有调用此工具的权限", 403)
        try:
            args = tool.input_model.model_validate(raw_args)
        except ValidationError:
            raise AppError("INVALID_TOOL_ARGS", "工具参数未通过校验，请补充或更正参数", 422) from None
        if (
            name == "segment_medical_volume"
            and context.segmentation_model_id
            and args.model_id != context.segmentation_model_id
        ):
            raise AppError("MODEL_SELECTION_MISMATCH", "工具请求与本轮选定的分割模型不一致，未执行", 422)
        return tool, args

    async def invoke(self, name, raw_args, context):
        tool, args = self.validate(name, raw_args, context)
        config = self.registry.config["tools"][name]
        retries = config.get("retries", 0) if tool.read_only else 0
        with span(
            "tool.invoke",
            tool_name=name,
            tool_version=tool.version,
            input_hash=digest(args.model_dump()),
            run_id=context.run_id,
        ):
            try:
                for attempt in range(retries + 1):
                    try:
                        async with asyncio.timeout(config["timeout_seconds"]):
                            raw_output = await tool.execute(args, context)
                        output = tool.output_model.model_validate(raw_output).model_dump(mode="json")
                        break
                    except TimeoutError:
                        if attempt == retries:
                            raise AppError("TOOL_TIMEOUT", "工具执行超时", 504, True) from None
                    except ValidationError:
                        raise AppError("TOOL_BAD_RESPONSE", "工具输出未通过契约校验", 502) from None
                outcome = "succeeded"
                return output
            except BaseException:
                outcome = "failed"
                raise
            finally:
                async with context.services.sessions() as session, session.begin():
                    audit(session, context.actor, "tool." + name, "run", context.run_id, outcome, trace_id())
