"""AutoGemini: A Python package for automating tasks with Gemini AI."""

__version__ = "0.1.0"

# gemini_chat
from .gemini_chat import (
    MessageRole,
    ChatMessage,
    StreamCancellation,
    stream_chat,
    fetch_available_models,
)

# auto_stream_processor
from .auto_stream_processor import (
    CallbackMsgType,
    AutoStreamProcessor,
    create_cot_processor,
    CLEAN_HTML_TAGS,
)

# template
from .template import (
    cot_template,
    ToolCodeInfo,
    COT,
    build_tool_code_template,
    build_tool_code_prompt,
    gemini_template,
    ParsedBlock,
    parse_agent_output,
)

# tool_code
from .tool_code import (
    DefaultApi,
    eval_tool_code,
    extract_tool_code,
    ToolCodeProcessor,
    process_streaming_response,
    create_streaming_handler,
    extract_and_execute_all_tool_codes,
)

__all__ = [
    # gemini_chat
    "MessageRole",
    "ChatMessage",
    "StreamCancellation",
    "stream_chat",
    "fetch_available_models",
    # auto_stream_processor
    "CallbackMsgType",
    "AutoStreamProcessor",
    "create_cot_processor",
    "CLEAN_HTML_TAGS",
    # template
    "cot_template",
    "ToolCodeInfo",
    "COT",
    "build_tool_code_template",
    "build_tool_code_prompt",
    "gemini_template",
    "ParsedBlock",
    "parse_agent_output",
    # tool_code
    "DefaultApi",
    "eval_tool_code",
    "extract_tool_code",
    "ToolCodeProcessor",
    "process_streaming_response",
    "create_streaming_handler",
    "extract_and_execute_all_tool_codes",
]
