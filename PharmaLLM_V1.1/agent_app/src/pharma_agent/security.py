import asyncio
from dataclasses import dataclass
from datetime import timedelta

import jwt
from pwdlib import PasswordHash
from sqlalchemy import select
from starlette.requests import Request

from pharma_agent.db import AgentRun, Asset, Conversation, Tenant, ToolJob, User, UserMemory, audit, now
from pharma_agent.errors import AppError, not_found
from pharma_agent.observability import digest, trace_id

password_hasher = PasswordHash.recommended()
# A precomputed hash is initialized lazily; API import does not hash a password.
_dummy_hash = None


@dataclass(frozen=True)
class Actor:
    id: str
    tenant_id: str
    role: str
    department_id: str | None = None

    @property
    def scopes(self):
        scopes = {"jobs:read", "references:read", "preferences:read"}
        if self.role in {"owner", "doctor", "member"}:
            scopes.update({"tasks:draft", "reviews:submit"})
        if self.role in {"owner", "reviewer"}:
            scopes.add("reviews:decide")
        if self.role == "owner":
            scopes.update({"documents:write", "organization:manage"})
        return scopes


async def hash_password(password: str):
    return await asyncio.to_thread(password_hasher.hash, password)


async def verify_password(password: str, hashed: str | None):
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = await hash_password("dummy-password-for-timing-1.1")
    try:
        valid = await asyncio.to_thread(password_hasher.verify, password, hashed or _dummy_hash)
        return valid and hashed is not None
    except Exception:
        return False


def issue_token(settings, actor: Actor):
    timestamp = now()
    return jwt.encode(
        {
            "sub": actor.id,
            "tenant": actor.tenant_id,
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
            "iat": timestamp,
            "nbf": timestamp,
            "exp": timestamp + timedelta(minutes=settings.access_token_minutes),
        },
        settings.jwt_secret.get_secret_value(),
        algorithm="HS256",
    )


async def authenticate(request: Request) -> Actor:
    value = request.headers.get("authorization", "")
    if not value.startswith("Bearer "):
        raise AppError("AUTH_REQUIRED", "请先登录", 401)
    settings = request.app.state.services.settings
    try:
        payload = jwt.decode(
            value[7:],
            settings.jwt_secret.get_secret_value(),
            algorithms=["HS256"],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            options={"require": ["exp", "iat", "nbf", "sub", "tenant"]},
        )
        uid, tenant_id = payload["sub"], payload["tenant"]
        if not isinstance(uid, str) or not isinstance(tenant_id, str):
            raise ValueError("invalid identity")
    except (jwt.InvalidTokenError, ValueError):
        raise AppError("AUTH_REQUIRED", "登录已失效，请重新登录", 401) from None
    async with request.app.state.services.sessions() as session:
        user = (
            await session.scalars(
                select(User)
                .join(Tenant, Tenant.id == User.tenant_id)
                .where(
                    User.id == uid,
                    User.tenant_id == tenant_id,
                    User.status == "active",
                    Tenant.status == "active",
                )
            )
        ).first()
        if user is None:
            raise AppError("AUTH_REQUIRED", "登录已失效，请重新登录", 401)
        return Actor(user.id, user.tenant_id, user.role, user.department_id)


def require_scope(actor: Actor, scope: str):
    if scope not in actor.scopes:
        raise AppError("PERMISSION_DENIED", "当前角色没有此操作权限", 403)


def owned_query(model, actor: Actor, resource_id: str):
    column = model.owner_user_id if model in {Conversation, Asset} else model.user_id
    query = select(model).where(
        model.id == resource_id, model.tenant_id == actor.tenant_id, column == actor.id
    )
    if model is Conversation:
        query = query.where(Conversation.deleted_at.is_(None))
    return query


async def require_owned(services, model, actor: Actor, resource_id: str, session=None, lock=False):
    if not isinstance(resource_id, str) or not 1 <= len(resource_id) <= 64:
        raise not_found()
    query = owned_query(model, actor, resource_id)
    if lock:
        query = query.with_for_update()
    if session is not None:
        item = (await session.scalars(query)).first()
    else:
        async with services.sessions() as connection:
            item = (await connection.scalars(query)).first()
    if item is None:
        async with services.sessions() as audit_session, audit_session.begin():
            audit(
                audit_session, actor, "access.denied", model.__tablename__, resource_id, "denied", trace_id()
            )
        raise not_found()
    return item


async def preferences(sessions, actor: Actor):
    async with sessions() as session:
        rows = (
            await session.scalars(
                select(UserMemory).where(
                    UserMemory.tenant_id == actor.tenant_id,
                    UserMemory.user_id == actor.id,
                    (UserMemory.expires_at.is_(None)) | (UserMemory.expires_at > now()),
                )
            )
        ).all()
        return {row.kind: row.value for row in rows}


RATE_LUA = """
local n = redis.call('INCR', KEYS[1])
if n == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
return n
"""


async def rate_limit(services, actor: Actor):
    settings = services.settings
    if settings.rate_limit_per_minute == 0:
        return
    if services.redis is None:
        if settings.rate_limit_fail_closed:
            raise AppError("RATE_LIMIT_UNAVAILABLE", "限流服务未启用", 503, True)
        return
    key = f"pharma:rate:{actor.tenant_id}:{actor.id}:{int(now().timestamp()) // 60}"
    try:
        count = await services.redis.eval(RATE_LUA, 1, key, 70)
    except Exception:
        if settings.rate_limit_fail_closed:
            raise AppError("RATE_LIMIT_UNAVAILABLE", "限流服务暂不可用", 503, True) from None
        return
    if int(count) > settings.rate_limit_per_minute:
        raise AppError("RATE_LIMITED", "请求过于频繁，请稍后重试", 429, True)


async def auth_rate_limit(services, client_host: str):
    if services.settings.app_env == "test" and services.settings.rate_limit_per_minute == 0:
        return
    if services.redis is None:
        return  # Development can disable Redis; production startup rejects that configuration.
    key = f"pharma:auth:{digest(client_host)[7:31]}:{int(now().timestamp()) // 60}"
    try:
        count = await services.redis.eval(RATE_LUA, 1, key, 70)
        if int(count) > 20:
            raise AppError("RATE_LIMITED", "登录或注册请求过于频繁", 429, True)
    except AppError:
        raise
    except Exception:
        if services.settings.rate_limit_fail_closed:
            raise AppError("RATE_LIMIT_UNAVAILABLE", "限流服务暂不可用", 503, True) from None


OWNED_MODELS = (AgentRun, ToolJob)
