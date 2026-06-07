from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Sequence, TYPE_CHECKING, Union

from langchain_core.messages import ToolMessage as BaseToolMessage

if TYPE_CHECKING:
    from .thread import AIMessage, HumanMessage, Message, ToolMessage, EncoderType

HideScope = Literal[
    "global",
    "since_last_human",
    "since_last_assistant",
    "same_tool_call_chain",
]
HideStrategy = Literal["placeholder", "summarize", "drop"]

OnHideMessage = Union[str, Callable[["ToolMessage", "ToolResultHideRule"], str]]
SummarizeFn = Callable[["ToolMessage"], str]

_TOOL_HIDE_META_KEY = "_tool_hide"
_ORIGINAL_CONTENT_KEY = "original_content"
_ORIGINAL_TOKEN_COUNT_KEY = "original_token_count"
_ORIGINAL_PREVIEW_KEY = "original_preview"
_PREVIEW_CHARS = 200


@dataclass
class ToolResultHideRule:
    """Rule for collapsing older tool results when sending context to the model."""

    tool_name: str
    on_hide_message: OnHideMessage | None = (
        "[Earlier {tool_name} result hidden — see latest result above]"
    )
    server: str | None = None
    keep_latest: int = 1
    min_tokens_to_hide: int = 0
    scope: HideScope = "global"
    on_hide_strategy: HideStrategy = "placeholder"
    summarize_fn: SummarizeFn | None = None
    max_hidden_results: int | None = None
    encoder: "EncoderType" = "cl100k_base"

    def __post_init__(self) -> None:
        if self.keep_latest < 0:
            raise ValueError("keep_latest must be >= 0")
        if self.min_tokens_to_hide < 0:
            raise ValueError("min_tokens_to_hide must be >= 0")


def resolve_full_tool_name(server: str | None, tool_name: str) -> str:
    if server and tool_name != "*" and not tool_name.startswith(f"{server}_"):
        return f"{server}_{tool_name}"
    return tool_name


def tool_message_name(message: BaseToolMessage) -> str:
    return getattr(message, "name", None) or ""


def message_matches_rule(message: BaseToolMessage, rule: ToolResultHideRule) -> bool:
    name = tool_message_name(message)
    if not name:
        return False

    pattern = resolve_full_tool_name(rule.server, rule.tool_name)
    if pattern == "*":
        if rule.server is None:
            return True
        prefix = f"{rule.server}_"
        return name.startswith(prefix)

    if "*" in pattern or "?" in pattern or "[" in pattern:
        return fnmatch.fnmatch(name, pattern)

    return name == pattern


def scope_start_index(
    messages: Sequence["Message"],
    message_index: int,
    scope: HideScope,
) -> int:
    from .thread import AIMessage, HumanMessage

    if scope == "global":
        return 0

    if scope == "since_last_human":
        for idx in range(message_index - 1, -1, -1):
            if isinstance(messages[idx], HumanMessage):
                return idx + 1
        return 0

    if scope == "since_last_assistant":
        for idx in range(message_index - 1, -1, -1):
            if isinstance(messages[idx], AIMessage):
                return idx + 1
        return 0

    if scope == "same_tool_call_chain":
        tool_call_id = getattr(messages[message_index], "tool_call_id", None)
        if not tool_call_id:
            return 0
        for idx in range(message_index - 1, -1, -1):
            msg = messages[idx]
            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None) or []
                if any(
                    (tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None))
                    == tool_call_id
                    for tc in tool_calls
                ):
                    return idx
        return 0

    return 0


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(json.dumps(block, default=str))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def count_message_tokens(message: BaseToolMessage, encoder: "EncoderType") -> int:
    from .thread import _get_encoder

    text = _content_to_text(message.content)
    name = tool_message_name(message)
    if name:
        text = f"{name}\n{text}"
    return len(_get_encoder(encoder).encode(text))


def get_original_content(message: BaseToolMessage) -> str:
    meta = _get_hide_meta(message)
    if meta and _ORIGINAL_CONTENT_KEY in meta:
        original = meta[_ORIGINAL_CONTENT_KEY]
        return original if isinstance(original, str) else _content_to_text(original)
    return _content_to_text(message.content)


def is_hidden_for_model(message: BaseToolMessage) -> bool:
    meta = _get_hide_meta(message)
    return bool(meta and meta.get("hidden"))


def _get_hide_meta(message: BaseToolMessage) -> dict | None:
    kwargs = getattr(message, "additional_kwargs", None) or {}
    meta = kwargs.get(_TOOL_HIDE_META_KEY)
    return meta if isinstance(meta, dict) else None


def _set_hide_meta(message: "ToolMessage", meta: dict) -> "ToolMessage":
    dump = message.model_dump()
    kwargs = dict(dump.get("additional_kwargs") or {})
    kwargs[_TOOL_HIDE_META_KEY] = meta
    dump["additional_kwargs"] = kwargs
    from .thread import ToolMessage

    return ToolMessage.model_validate(dump)


def _render_hide_message(
    rule: ToolResultHideRule,
    message: "ToolMessage",
) -> str:
    if rule.on_hide_strategy == "summarize" and rule.summarize_fn is not None:
        return rule.summarize_fn(message)

    if callable(rule.on_hide_message):
        return rule.on_hide_message(message, rule)

    template = rule.on_hide_message or (
        "[Earlier {tool_name} result hidden — see latest result above]"
    )
    return template.format(
        tool_name=tool_message_name(message),
        server=rule.server or "",
        tool_call_id=getattr(message, "tool_call_id", "") or "",
    )


def _clone_tool_message(message: "ToolMessage", content: str) -> "ToolMessage":
    from .thread import ToolMessage

    dump = message.model_dump()
    dump["content"] = content
    return ToolMessage.model_validate(dump)


def _build_hidden_message(
    message: "ToolMessage",
    rule: ToolResultHideRule,
    *,
    persist: bool,
    message_index: int,
) -> Optional["ToolMessage"]:
    if rule.on_hide_strategy == "drop":
        return None

    hidden_content = _render_hide_message(rule, message)
    if not persist:
        return _clone_tool_message(message, hidden_content)

    original = get_original_content(message)
    token_count = count_message_tokens(message, rule.encoder)
    meta = {
        "hidden": True,
        "rule_tool_name": rule.tool_name,
        "rule_server": rule.server,
        "hidden_at_index": message_index,
        _ORIGINAL_CONTENT_KEY: original,
        _ORIGINAL_TOKEN_COUNT_KEY: token_count,
        _ORIGINAL_PREVIEW_KEY: original[:_PREVIEW_CHARS],
    }
    hidden = _clone_tool_message(message, hidden_content)
    return _set_hide_meta(hidden, meta)


def _matching_indices(
    messages: Sequence["Message"],
    rule: ToolResultHideRule,
    *,
    start: int = 0,
    end: int | None = None,
) -> list[int]:
    from .thread import ToolMessage

    end = len(messages) if end is None else end
    indices: list[int] = []
    for idx in range(start, end):
        msg = messages[idx]
        if isinstance(msg, ToolMessage) and message_matches_rule(msg, rule):
            indices.append(idx)
    return indices


def _last_message_index(
    messages: Sequence["Message"],
    message_type: type,
) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], message_type):
            return idx
    return -1


def _scope_cutoff_index(
    messages: Sequence["Message"],
    scope: HideScope,
) -> int:
    """Only tool messages with index > cutoff are eligible for hiding."""
    from .thread import AIMessage, HumanMessage

    if scope == "global":
        return -1
    if scope == "since_last_human":
        return _last_message_index(messages, HumanMessage)
    if scope == "since_last_assistant":
        return _last_message_index(messages, AIMessage)
    return -1


def _indices_to_hide_for_rule(
    messages: Sequence["Message"],
    rule: ToolResultHideRule,
) -> tuple[set[int], set[int]]:
    """Return (placeholder_indices, drop_indices) for a rule."""
    if not messages:
        return set(), set()

    cutoff = _scope_cutoff_index(messages, rule.scope)
    all_matches = _matching_indices(messages, rule)
    if not all_matches:
        return set(), set()

    to_hide: set[int] = set()
    to_drop: set[int] = set()

    grouped: dict[int, list[int]] = {}
    for idx in all_matches:
        if rule.scope in ("since_last_human", "since_last_assistant") and idx <= cutoff:
            continue
        scope_start = scope_start_index(messages, idx, rule.scope)
        grouped.setdefault(scope_start, []).append(idx)

    for indices in grouped.values():
        if len(indices) <= rule.keep_latest:
            continue

        hide_candidates = sorted(indices[: len(indices) - rule.keep_latest])
        eligible: list[int] = []
        for idx in hide_candidates:
            msg = messages[idx]
            if not isinstance(msg, BaseToolMessage):
                continue
            if is_hidden_for_model(msg):
                eligible.append(idx)
                continue
            token_count = count_message_tokens(msg, rule.encoder)
            if token_count >= rule.min_tokens_to_hide:
                eligible.append(idx)

        if rule.max_hidden_results is not None and len(eligible) > rule.max_hidden_results:
            overflow = len(eligible) - rule.max_hidden_results
            to_drop.update(eligible[:overflow])
            to_hide.update(eligible[overflow:])
        else:
            to_hide.update(eligible)

    return to_hide, to_drop


def apply_tool_hide_rules_to_messages(
    messages: Sequence["Message"],
    rules: Sequence[ToolResultHideRule],
    *,
    persist: bool = False,
) -> List["Message"]:
    """Return a copy of messages with older matching tool results collapsed for the model."""
    if not rules:
        return list(messages)

    hide_indices: set[int] = set()
    drop_indices: set[int] = set()
    hide_rule_by_index: dict[int, ToolResultHideRule] = {}

    for rule in rules:
        to_hide, to_drop = _indices_to_hide_for_rule(messages, rule)
        for idx in to_drop:
            drop_indices.add(idx)
        for idx in to_hide:
            if idx not in drop_indices:
                hide_indices.add(idx)
                hide_rule_by_index[idx] = rule

    if not hide_indices and not drop_indices:
        return list(messages)

    out: List["Message"] = []

    for idx, message in enumerate(messages):
        if idx in drop_indices:
            continue

        if idx not in hide_indices:
            out.append(message)
            continue

        rule = hide_rule_by_index[idx]
        from .thread import ToolMessage

        if not isinstance(message, ToolMessage):
            out.append(message)
            continue

        if rule.on_hide_strategy == "drop":
            continue

        hidden = _build_hidden_message(
            message,
            rule,
            persist=persist,
            message_index=idx,
        )
        if hidden is not None:
            out.append(hidden)

    return out


def persist_tool_hide_rules(
    messages: List["Message"],
    rules: Sequence[ToolResultHideRule],
) -> None:
    """Mutate thread storage in place, backing up original tool content in metadata."""
    if not rules:
        return
    messages[:] = apply_tool_hide_rules_to_messages(messages, rules, persist=True)


def restore_hidden_tool_messages(messages: List["Message"]) -> int:
    """Restore persisted hidden tool content. Returns count restored."""
    from .thread import ToolMessage

    restored = 0
    for idx, message in enumerate(messages):
        if not isinstance(message, ToolMessage):
            continue
        meta = _get_hide_meta(message)
        if not meta or not meta.get("hidden"):
            continue
        original = meta.get(_ORIGINAL_CONTENT_KEY)
        if original is None:
            continue
        dump = message.model_dump()
        dump["content"] = original
        kwargs = dict(dump.get("additional_kwargs") or {})
        kwargs.pop(_TOOL_HIDE_META_KEY, None)
        dump["additional_kwargs"] = kwargs
        messages[idx] = ToolMessage.model_validate(dump)
        restored += 1
    return restored


def trim_messages_to_token_budget(
    messages: Sequence["Message"],
    max_tokens: int,
    encoder: "EncoderType",
) -> List["Message"]:
    from .thread import TokenCalculator, SystemMessage

    if max_tokens <= 0:
        return []

    calculator = TokenCalculator(encoder)
    counts = calculator.counts(list(messages))
    total = sum(counts)
    if total <= max_tokens:
        return list(messages)

    keep_from = 0
    if messages and isinstance(messages[0], SystemMessage):
        keep_from = 1
        remaining = max_tokens - counts[0]
        if remaining <= 0:
            return [messages[0]]
    else:
        remaining = max_tokens

    selected: List["Message"] = []
    if keep_from:
        selected.append(messages[0])

    running = 0
    tail: List["Message"] = []
    for idx in range(len(messages) - 1, keep_from - 1, -1):
        msg = messages[idx]
        cost = counts[idx]
        if running + cost <= remaining:
            tail.append(msg)
            running += cost
        elif not tail:
            tail.append(msg)
            break

    tail.reverse()
    return selected + tail


def apply_rolling_message_window(
    messages: Sequence["Message"],
    max_messages: int,
) -> List["Message"]:
    from .thread import SystemMessage

    if max_messages <= 0:
        return []
    if len(messages) <= max_messages:
        return list(messages)

    if messages and isinstance(messages[0], SystemMessage):
        system = messages[0]
        rest = messages[1:]
        budget = max_messages - 1
        if budget <= 0:
            return [system]
        return [system] + list(rest[-budget:])

    return list(messages[-max_messages:])
