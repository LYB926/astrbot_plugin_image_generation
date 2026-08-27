"""Hand slash image commands off to the session chat LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.message_components import Image, Reply

from ..shared.logging import log_prefix, mask_sensitive, safe_log_text

# Registered FunctionTool.name (config label "生图工具" is different).
HANDOFF_TOOL_NAME = "generate_image"

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.provider import ProviderRequest
    from astrbot.api.star import Context
    from astrbot.core.agent.tool import ToolSet

LOG = log_prefix("CommandHandoff")

# event.extra key: mark /生图 handoff so on_llm_request can re-clamp tools
# after AstrBot merges persona / builtin tools onto the ProviderRequest.
HANDOFF_EVENT_EXTRA_KEY = "image_gen_command_handoff"

HANDOFF_SYSTEM_PROMPT = """\
用户通过 /生图 命令请求生成或修改图片。你必须调用 generate_image 工具完成出图，不要只用文字空谈或假装已画图。

结合当前人设与对话上下文理解用户意图，编写具体、可独立理解的视觉 prompt；涉及自身形象、角色或人设时，优先填写 persona（若工具可用）并写清外观，不要把用户的模糊原话原样塞进 prompt。

工具返回任务已提交后，用符合人设的简短语气确认即可，不要复述技术细节或重复调用 generate_image。
本轮仅可使用 generate_image，不要调用其它工具。
""".strip()


def build_handoff_tool_set(context: Context) -> ToolSet | None:
    """Restrict handoff to generate_image only (avoid other tools / loop strip)."""
    try:
        from astrbot.core.agent.tool import ToolSet
    except Exception:
        logger.warning(f"{LOG} 无法导入 ToolSet", exc_info=True)
        return None

    get_manager = getattr(context, "get_llm_tool_manager", None)
    if not callable(get_manager):
        return None
    try:
        manager = get_manager()
    except Exception:
        logger.warning(f"{LOG} 获取 LLM 工具管理器失败", exc_info=True)
        return None
    if manager is None:
        return None

    tool = None
    get_func = getattr(manager, "get_func", None)
    if callable(get_func):
        try:
            tool = get_func(HANDOFF_TOOL_NAME)
        except Exception:
            logger.debug(f"{LOG} get_func({HANDOFF_TOOL_NAME}) 失败", exc_info=True)
            tool = None
    if tool is None:
        logger.warning(f"{LOG} 未找到工具 {HANDOFF_TOOL_NAME}，handoff 回退")
        return None
    if hasattr(tool, "active") and not bool(getattr(tool, "active", True)):
        logger.warning(f"{LOG} 工具 {HANDOFF_TOOL_NAME} 未激活，handoff 回退")
        return None

    tool_set = ToolSet()
    tool_set.add_tool(tool)
    return tool_set


def clamp_request_to_handoff_tools(
    context: Context,
    req: Any,
) -> bool:
    """Force req.func_tool to only generate_image. Returns True if clamped."""
    tool_set = build_handoff_tool_set(context)
    if tool_set is None:
        return False
    req.func_tool = tool_set
    return True


def build_handoff_prompt(*, raw_demand: str, image_count: int | None = None) -> str:
    """Build the user-side prompt passed to request_llm for /生图 handoff."""
    demand = str(raw_demand or "").strip()
    lines = [
        "用户通过命令要求生成图片。",
        "命令原文中的需求：",
        demand or "（空）",
        "",
        "请结合当前人设与对话上下文理解意图，调用 generate_image 工具完成出图。",
        "编写具体、可画的 prompt；若涉及自身形象/人设角色，优先使用 persona 参数（如已配置）并写清外观。",
        "不要只把上面那句原话原样塞进 prompt。",
        "同一轮只调用一次 generate_image（除非用户明确要多张且通过 image_count 表达）。",
    ]
    if image_count is not None and image_count > 0:
        lines.extend(
            [
                "",
                f"用户请求生成数量：{image_count}。请在调用 generate_image 时传入 image_count={image_count}。",
            ]
        )
    return "\n".join(lines).strip()


async def collect_handoff_image_urls(event: AstrMessageEvent) -> list[str]:
    """Collect message and replied image paths/URLs for request_llm."""
    urls: list[str] = []
    seen: set[str] = set()

    async def _append_from_image(component: Any) -> None:
        path_or_url = ""
        convert = getattr(component, "convert_to_file_path", None)
        if callable(convert):
            try:
                path_or_url = str(await convert() or "").strip()
            except Exception:
                logger.debug(f"{LOG} convert_to_file_path 失败", exc_info=True)
                path_or_url = ""
        if not path_or_url:
            path_or_url = str(
                getattr(component, "url", None) or getattr(component, "file", None) or ""
            ).strip()
        if not path_or_url or path_or_url in seen:
            return
        seen.add(path_or_url)
        urls.append(path_or_url)

    message = getattr(getattr(event, "message_obj", None), "message", None) or []
    for component in message:
        try:
            if isinstance(component, Image):
                await _append_from_image(component)
            elif isinstance(component, Reply):
                chain = getattr(component, "chain", None) or []
                for sub in chain:
                    if isinstance(sub, Image):
                        await _append_from_image(sub)
        except Exception:
            logger.debug(f"{LOG} 收集 handoff 图片失败", exc_info=True)
    return urls


async def resolve_session_conversation(
    context: Context,
    event: AstrMessageEvent,
) -> Any | None:
    """Load or create the current session conversation for persona and history."""
    conversation_manager = getattr(context, "conversation_manager", None)
    if conversation_manager is None:
        return None

    umo = event.unified_msg_origin
    try:
        curr_cid = await conversation_manager.get_curr_conversation_id(umo)
        if curr_cid:
            return await conversation_manager.get_conversation(umo, curr_cid)

        platform_id = ""
        if hasattr(event, "get_platform_id"):
            platform_id = str(event.get_platform_id() or "")
        curr_cid = await conversation_manager.new_conversation(
            umo,
            platform_id=platform_id,
        )
        if not curr_cid:
            return None
        return await conversation_manager.get_conversation(umo, curr_cid)
    except Exception as exc:
        logger.warning(
            f"{LOG} 获取会话 conversation 失败: 用户={mask_sensitive(umo)}，"
            f"错误={safe_log_text(exc, 160)}"
        )
        return None


async def try_request_llm_handoff(
    plugin: Any,
    event: AstrMessageEvent,
    *,
    raw_demand: str,
    image_count: int | None = None,
) -> ProviderRequest | None:
    """Build a ProviderRequest for command handoff, or None to fall back."""
    context: Context = plugin.context
    umo = event.unified_msg_origin
    masked_uid = mask_sensitive(umo)

    provider = None
    get_provider = getattr(context, "get_using_provider", None)
    if callable(get_provider):
        try:
            provider = get_provider(umo)
        except Exception:
            logger.warning(
                f"{LOG} 查询聊天模型失败: 用户={masked_uid}",
                exc_info=True,
            )
            provider = None
    if not provider:
        logger.warning(
            f"{LOG} 无可用聊天模型，/生图 回退为直接执行指令: 用户={masked_uid}"
        )
        return None

    tool_set = build_handoff_tool_set(context)
    if tool_set is None:
        logger.warning(
            f"{LOG} 无法构建仅含 generate_image 的工具集，/生图 回退为直接执行指令: "
            f"用户={masked_uid}"
        )
        return None

    conversation = await resolve_session_conversation(context, event)
    image_urls = await collect_handoff_image_urls(event)
    prompt = build_handoff_prompt(raw_demand=raw_demand, image_count=image_count)

    set_extra = getattr(event, "set_extra", None)
    if callable(set_extra):
        set_extra(HANDOFF_EVENT_EXTRA_KEY, True)

    logger.info(
        f"{LOG} 将 /生图 交给会话 LLM: 用户={masked_uid}，"
        f"需求长度={len(str(raw_demand or '').strip())}，"
        f"数量={image_count if image_count is not None else '默认'}，"
        f"参考图={len(image_urls)}，"
        f"conversation={'有' if conversation is not None else '无'}，"
        f"tools={tool_set.names() if hasattr(tool_set, 'names') else [HANDOFF_TOOL_NAME]}"
    )
    return event.request_llm(
        prompt=prompt,
        system_prompt=HANDOFF_SYSTEM_PROMPT,
        conversation=conversation,
        image_urls=image_urls,
        tool_set=tool_set,
    )


__all__ = (
    "HANDOFF_EVENT_EXTRA_KEY",
    "HANDOFF_SYSTEM_PROMPT",
    "HANDOFF_TOOL_NAME",
    "build_handoff_prompt",
    "build_handoff_tool_set",
    "clamp_request_to_handoff_tools",
    "collect_handoff_image_urls",
    "resolve_session_conversation",
    "try_request_llm_handoff",
)
