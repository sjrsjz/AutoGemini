"""AutoGemini: A Python package for automating tasks with Gemini AI."""

__version__ = "0.1.0"

from .gemini_chat import (
    MessageRole,
    ChatMessage,
    StreamCancellation,
    stream_chat,
    fetch_available_models,
)

__all__ = [
    "MessageRole",
    "ChatMessage",
    "StreamCancellation",
    "stream_chat",
    "fetch_available_models",
]
