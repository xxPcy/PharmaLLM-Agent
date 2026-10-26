"""Read fresh, owner-checked imaging evidence for this conversation on every turn."""

import json
import math

from sqlalchemy import or_, select

from pharma_agent.db import AgentRun, ToolJob
from pharma_agent.errors import AppError


LABEL_NAMES = {"femoralhead": "股骨头", "acetabulum": "髋臼"}


def measurements(rows):
    result = []
    for row in rows[:8]:
        if not isinstance(row, dict):
            continue
        item = {key: row[key] for key in ("label", "voxel_count") if key in row}
        name = str(row.get("name", ""))[:80]
        item.update(name=name, basis=str(row.get("basis", ""))[:120])
        item["display_name"] = LABEL_NAMES.get(name, name)
        value = row.get("volume_mm3")
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ):
            item.update(volume_mm3=value, volume_ml=value / 1000)
        result.append(item)
    return result


def imaging_evidence(context, byte_budget=4800):
    """Keep the latest full result ahead of older files, with complete JSON boundaries."""
    data = {"jobs": list(context.get("jobs", [])[:3]), "files": list(context.get("files", [])[:3])}
    while True:
        encoded = json.dumps(data, ensure_ascii=False)
        if len(encoded.encode()) <= byte_budget:
            return encoded
        if len(data["jobs"]) > 1:
            data["jobs"].pop()
        elif data["files"]:
            data["files"].pop()
        else:
            # Abnormal metadata must not consume the entire prompt or be cut into invalid JSON.
            return json.dumps({"jobs": [], "files": [], "context_too_large": True})


async def load_conversation_imaging(services, actor, conversation_id, run_id):
    from pharma_agent.imaging_api import ready_asset

    async with services.sessions() as session:
        attachments = (
            await session.scalars(
                select(AgentRun.model_snapshot["asset_ids"])
                .where(
                    AgentRun.tenant_id == actor.tenant_id,
                    AgentRun.user_id == actor.id,
                    AgentRun.conversation_id == conversation_id,
                    # Accepted attachments remain linked if generation later fails or is cancelled.
                    or_(AgentRun.status.in_(["completed", "failed", "cancelled"]), AgentRun.id == run_id),
                )
                .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
                .limit(12)
            )
        ).all()
        jobs = (
            await session.scalars(
                select(ToolJob)
                .where(
                    ToolJob.tenant_id == actor.tenant_id,
                    ToolJob.user_id == actor.id,
                    ToolJob.tool_name == "segment_medical_volume",
                    ToolJob.input_json["conversation_id"].as_string() == conversation_id,
                )
                .order_by(ToolJob.created_at.desc(), ToolJob.id.desc())
                .limit(5)
            )
        ).all()

    # Defense in depth: both queries and individual asset reads enforce ownership.
    jobs = [
        job
        for job in jobs
        if (
            getattr(job, "tenant_id", None) == actor.tenant_id
            and getattr(job, "user_id", None) == actor.id
            and getattr(job, "tool_name", None) == "segment_medical_volume"
            and job.input_json.get("conversation_id") == conversation_id
        )
    ]
    candidates = [job.input_json.get("file_id") for job in jobs]
    candidates.extend(asset_id for ids in attachments if isinstance(ids, list) for asset_id in ids)
    files, source_meta = [], {}
    for asset_id in dict.fromkeys(item for item in candidates if isinstance(item, str)):
        if len(files) >= 8:
            break
        try:
            asset, meta = await ready_asset(services, actor, asset_id)
        except AppError:
            continue
        if meta.get("conversation_id") != conversation_id or meta.get("kind") == "mask":
            continue
        source_meta[asset.id] = meta
        file_info = {
            key: meta[key]
            for key in ("asset_id", "name", "kind", "dimensions", "modality", "channels", "status")
            if key in meta
        }
        if "name" in file_info:
            file_info["name"] = str(file_info["name"])[:128]
        files.append(file_info)

    summaries = []
    for job in jobs[:3]:
        source_id = job.input_json.get("file_id")
        item = {
            "job_id": job.id,
            "status": job.status,
            "source_asset_id": source_id,
            "file_name": str(job.input_json.get("file_name", "影像"))[:128],
            "result_available": False,
        }
        summaries.append(item)
        if source_id not in source_meta:
            item["unavailable_reason"] = "source_unavailable"
            continue
        if job.status != "succeeded":
            item["phase"] = job.result_json.get("phase", job.status)
            continue
        mask_id = job.result_json.get("asset_id")
        if not mask_id:
            continue
        try:
            _, mask = await ready_asset(services, actor, mask_id)
        except AppError:
            item["unavailable_reason"] = "result_unavailable"
            continue
        if (
            mask.get("kind") != "mask"
            or mask.get("source_asset_id") != source_id
            or mask.get("conversation_id") != conversation_id
        ):
            item["unavailable_reason"] = "result_unavailable"
            continue
        item.update(
            result_available=True,
            result_asset_id=mask_id,
            labels={str(k): str(v)[:80] for k, v in list(mask.get("labels", {}).items())[:8]},
            measurements=measurements(mask.get("measurements", [])),
            measurement_scope="每个标签的全部体素；只有标签明确区分左右时才能分别报告左右体积",
        )
    return {"files": files, "jobs": summaries}
