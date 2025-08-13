"""
Gemini Chat streaming function using the official google-generativeai library.
A single function for streaming chat with Gemini API.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Callable, Awaitable

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as google_exceptions

from .template import COT


class MessageRole(Enum):
    """Message roles for chat completion."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """Simple chat message."""

    role: MessageRole
    content: str


class StreamCancellation:
    """Simple cancellation token for stream control."""

    def __init__(self):
        self.cancelled = False

    def cancel(self):
        """Cancel the stream."""
        self.cancelled = True

    def is_cancelled(self) -> bool:
        """Check if stream is cancelled."""
        return self.cancelled


def _process_reasoning_content(text: str, model: str) -> str:
    """Process reasoning model content, filtering out <thought> tag content."""
    if "thinking" not in model:
        return text

    result = ""
    in_thought = False
    i = 0
    while i < len(text):
        if text[i : i + 9] == "<thought>":
            in_thought = True
            i += 9
        elif text[i : i + 10] == "</thought>":
            in_thought = False
            i += 10
        else:
            if not in_thought:
                result += text[i]
            i += 1
    return result


async def stream_chat(
    api_key: str,
    callback: Callable[[str], Awaitable[None]],
    history: Optional[List[ChatMessage]] = None,
    user_message: Optional[str] = None,
    model: str = "gemini-2.5-flash",  # Updated to a common, modern model
    system_prompt: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 8192,
    top_p: float = 0.95,
    top_k: int = 40,
    cancellation_token: Optional[StreamCancellation] = None,
    timeout: float = 300.0,
) -> str:
    """
    Send a message and get a streaming response from the Gemini API using the official library.

    Args:
        api_key: Gemini API key
        callback: Function to call with each chunk of response
        history: Optional list of previous chat messages
        user_message: Optional user's message to send (if None, uses only history)
        model: Gemini model to use
        system_prompt: Optional system prompt
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        cancellation_token: Optional token to cancel the stream
        timeout: Request timeout in seconds

    Returns:
        Complete response text

    Raises:
        ValueError: If API request fails or no response generated
        asyncio.CancelledError: If cancelled via cancellation_token or asyncio
    """
    if not user_message and not history:
        raise ValueError("Either history or user_message must be provided.")

    try:
        genai.configure(api_key=api_key)

        # Combine user system prompt with the CoT template
        full_system_prompt = f"# I have double checked that my CoT settings are as follows, I will never disobey them:\n{COT}"

        # Prepare generation config and safety settings
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_tokens,
        )

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Instantiate the model with system prompt and configs
        generative_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=full_system_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        # Build conversation history for the API
        # The library uses 'model' for the assistant's role.
        api_history = []
        if system_prompt:
            api_history.append(
                {
                    "role": "model",
                    "parts": [
                        f"# I have double checked that my basic system settings are as follows, I will never disobey them:\n{system_prompt}"
                    ],
                }
            )

        if history:
            for message in history:
                role = "user" if message.role == MessageRole.USER else "model"
                api_history.append({"role": role, "parts": [message.content]})

        # Add the user's new message to the end of the history to be sent
        messages_to_send = list(api_history)
        if user_message:
            messages_to_send.append({"role": "user", "parts": [user_message]})

        if not messages_to_send:
            raise ValueError("No content to send to the model.")

        # Start the stream generation
        response = await generative_model.generate_content_async(
            contents=messages_to_send,
            stream=True,
            request_options={"timeout": timeout},
        )

        full_response_text = ""
        has_received_data = False
        async for chunk in response:
            if cancellation_token and cancellation_token.is_cancelled():
                # Note: This stops processing, but the underlying API call may continue.
                # The official library doesn't have a direct `cancel()` method on the stream.
                break

            has_received_data = True
            if chunk.text:
                processed_text = _process_reasoning_content(chunk.text, model)
                if processed_text:
                    await callback(processed_text)
                    full_response_text += processed_text

        if not full_response_text and has_received_data:
            return "(Response received but contained no processable text)"
        elif not full_response_text and not (
            cancellation_token and cancellation_token.is_cancelled()
        ):
            # This can happen if the model's response is blocked by safety filters
            # which are not fully disabled or if there's another issue.
            try:
                # The reason is available on the resolved response object
                prompt_feedback = response.prompt_feedback
                finish_reason = response.candidates[0].finish_reason
                raise ValueError(
                    f"No text generated. Finish Reason: {finish_reason}. Prompt Feedback: {prompt_feedback}"
                )
            except (AttributeError, IndexError):
                raise ValueError(
                    "No text generated from the stream, and no specific reason found."
                )

        return full_response_text

    except asyncio.CancelledError:
        raise
    except (
        google_exceptions.GoogleAPICallError,
        google_exceptions.RetryError,
        google_exceptions.InvalidArgument,
    ) as e:
        raise ValueError(f"Gemini API request failed: {str(e)}") from e
    except Exception as e:
        raise ValueError(f"Stream chat failed unexpectedly: {str(e)}") from e


async def fetch_available_models(api_key: str) -> List[str]:
    """Get list of available Gemini models using the official library."""
    try:
        genai.configure(api_key=api_key)

        models = []
        for m in genai.list_models():
            # We only want models that support content generation.
            if "generateContent" in m.supported_generation_methods:
                model_id = m.name

                # The library returns names like 'models/gemini-pro'. Strip the prefix.
                if model_id.startswith("models/"):
                    cleaned_id = model_id[7:]
                else:
                    cleaned_id = model_id

                if _is_valid_gemini_model(cleaned_id):
                    models.append(cleaned_id)
        return models
    except (
        google_exceptions.GoogleAPICallError,
        google_exceptions.RetryError,
        google_exceptions.Unauthenticated,
    ) as e:
        raise ValueError(f"Failed to fetch models: {str(e)}") from e


def _is_valid_gemini_model(model_id: str) -> bool:
    """Check if model meets our filtering criteria."""
    import re

    # This filtering logic is kept from the original implementation.
    # Check if it matches gemini-[version]-* pattern
    pattern = re.compile(r"^gemini-([1-9]|10)\.([0-9]|10)-")
    if not pattern.match(model_id):
        # Also allow models without version numbers like 'gemini-pro'
        if not model_id.startswith("gemini-"):
            return False

    # Exclude models containing specific keywords
    excluded_keywords = [
        "vision",
        "embedding",
        "audio",
        "tts",
        "exp",
        "native",
        "dialog",
        "live",
        "image",
        # "thinking" is kept to allow reasoning models but filter their output
    ]

    model_lower = model_id.lower()
    for keyword in excluded_keywords:
        if keyword in model_lower:
            # Special case: allow 'gemini-1.5-pro-vision-latest' if needed in future
            # for now, we follow the original exclusion
            return False

    return True
