"""
Gemini Chat streaming function.
A single function for streaming chat with Gemini API.
"""

import json
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Callable
import httpx


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
    if 'thinking' not in model:
        return text
    
    result = ""
    chars = list(text)
    i = 0
    in_thought = False
    
    while i < len(chars):
        # Check for <thought> tag
        if i + 8 < len(chars):
            potential_tag = ''.join(chars[i:i+9])
            if potential_tag == '<thought>':
                in_thought = True
                i += 9
                continue
        
        # Check for </thought> tag
        if i + 9 < len(chars):
            potential_tag = ''.join(chars[i:i+10])
            if potential_tag == '</thought>':
                in_thought = False
                i += 10
                continue
        
        # If not in thought tags, add character to result
        if not in_thought:
            result += chars[i]
        
        i += 1
    
    return result


async def _process_stream_response(
    response: httpx.Response,
    callback: Callable[[str], None],
    model: str,
    cancellation_token: Optional[StreamCancellation] = None
) -> str:
    """Process streaming response and return text via callback."""
    full_response = ""
    has_received_data = False
    
    # Character-level parsing variables
    buffer = ""  
    buffer_lv = 0  # Track JSON nesting depth
    in_string = False
    escape_char = False
    
    try:
        async for chunk in response.aiter_bytes():
            # Check for cancellation
            if cancellation_token and cancellation_token.is_cancelled():
                break
                
            has_received_data = True
            chunk_str = chunk.decode('utf-8', errors='ignore')
            
            # Process character by character
            for c in chunk_str:
                # Check for cancellation during processing
                if cancellation_token and cancellation_token.is_cancelled():
                    break
                    
                # Handle escaping
                if in_string and not escape_char and c == '\\':
                    escape_char = True
                    buffer += c
                    continue
                    
                if in_string and escape_char:
                    escape_char = False
                    buffer += c
                    continue
                
                # String boundary handling
                if c == '"' and not escape_char:
                    in_string = not in_string
                # Increase nesting depth (only outside strings)
                elif (c in '{[') and not in_string:
                    buffer_lv += 1
                # Decrease nesting depth (only outside strings) 
                elif (c in '}]') and not in_string:
                    buffer_lv -= 1
                
                # Record characters when depth > 1 (inside JSON objects)
                if buffer_lv > 1:
                    if in_string and c == '\n':
                        buffer += '\\n'  # Handle newlines in strings
                    else:
                        buffer += c
                
                # When back to depth 1 (object end) and buffer not empty
                elif buffer_lv == 1 and buffer:
                    # Add closing brace since it was read but not added to buffer
                    buffer += '}'
                    
                    # Parse the complete object
                    try:
                        json_value = json.loads(buffer)
                        
                        # Extract text content
                        candidates = json_value.get('candidates', [])
                        if candidates:
                            candidate = candidates[0]
                            content = candidate.get('content', {})
                            parts = content.get('parts', [])
                            if parts:
                                part = parts[0]
                                text = part.get('text', '')
                                if text:
                                    # Process reasoning content for thinking models
                                    processed_text = _process_reasoning_content(text, model)
                                    
                                    if processed_text:  # Only callback if there's content
                                        callback(processed_text)
                                        full_response += processed_text
                    
                    except json.JSONDecodeError:
                        pass  # Ignore parse errors for incomplete objects
                    
                    # Clear buffer for next object
                    buffer = ""
            
            # Break outer loop if cancelled
            if cancellation_token and cancellation_token.is_cancelled():
                break
        
        # Process any remaining buffer
        if buffer and not (cancellation_token and cancellation_token.is_cancelled()):
            if buffer.startswith('{') and not buffer.endswith('}'):
                buffer += '}'
            
            try:
                json_value = json.loads(buffer)
                candidates = json_value.get('candidates', [])
                if candidates:
                    candidate = candidates[0]
                    content = candidate.get('content', {})
                    parts = content.get('parts', [])
                    if parts:
                        part = parts[0]
                        text = part.get('text', '')
                        if text:
                            processed_text = _process_reasoning_content(text, model)
                            if processed_text:
                                callback(processed_text)
                                full_response += processed_text
            except json.JSONDecodeError:
                pass  # Ignore parse errors for incomplete objects
    
    except asyncio.CancelledError:
        # Handle asyncio cancellation
        raise
    except Exception as e:
        # Handle other exceptions but don't fail silently
        if not full_response:
            raise e  # Re-raise if we haven't gotten any response yet
    
    # Check if response is empty but data was received
    if not full_response and has_received_data:
        return "(Response received but requires different format parsing)"
    elif not full_response and not (cancellation_token and cancellation_token.is_cancelled()):
        raise ValueError("No text generated from the stream")
    
    return full_response


async def stream_chat(
    api_key: str,
    user_message: str,
    callback: Callable[[str], None],
    history: Optional[List[ChatMessage]] = None,
    model: str = "gemini-2.0-flash-thinking-exp",
    system_prompt: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 8192,
    top_p: float = 0.95,
    top_k: int = 40,
    cancellation_token: Optional[StreamCancellation] = None,
    timeout: float = 300.0
) -> str:
    """
    Send a message and get streaming response from Gemini API.
    
    Args:
        api_key: Gemini API key
        user_message: The user's message to send
        callback: Function to call with each chunk of response
        history: Optional list of previous chat messages
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
    # Build conversation history
    contents = []
    
    # Add conversation history if provided
    if history:
        for message in history:
            role = "user" if message.role == MessageRole.USER else "model"
            contents.append({
                "role": role,
                "parts": [{"text": message.content}]
            })
    
    # Add current user message
    contents.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })
    
    # Build request body
    request_body = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "topK": top_k,
            "maxOutputTokens": max_tokens
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    }
    
    # Add system instruction if provided
    if system_prompt:
        request_body["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }
    
    # Build URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key={api_key}"
    
    # Make streaming request
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(url, json=request_body)
            
            if not response.is_success:
                error_text = await response.aread()
                raise ValueError(f"API request failed ({response.status_code}): {error_text.decode()}")
            
            # Process streaming response
            return await _process_stream_response(response, callback, model, cancellation_token)
            
        except asyncio.CancelledError:
            # Re-raise asyncio cancellation
            raise
        except Exception as e:
            # Wrap other exceptions
            raise ValueError(f"Stream chat failed: {str(e)}") from e

# Utility functions for model filtering
async def fetch_available_models(api_key: str) -> List[str]:
    """Get list of available Gemini models."""
    url = "https://generativelanguage.googleapis.com/v1beta/openai/models"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"}
        )
    
    if not response.is_success:
        error_text = await response.aread()
        raise ValueError(f"Failed to fetch models ({response.status_code}): {error_text.decode()}")
    
    response_json = response.json()
    
    # Parse model list
    models = []
    data = response_json.get('data', [])
    
    for model in data:
        model_id = model.get('id', '')
        
        # Remove 'models/' prefix if present
        if model_id.startswith('models/'):
            cleaned_id = model_id[7:]  # Remove "models/" prefix
        else:
            cleaned_id = model_id
        
        # Filter models that meet our criteria
        if _is_valid_gemini_model(cleaned_id):
            models.append(cleaned_id)
    
    return models


def _is_valid_gemini_model(model_id: str) -> bool:
    """Check if model meets our filtering criteria."""
    import re
    
    # Check if it matches gemini-[1-10].[0-10]-* pattern  
    pattern = re.compile(r'^gemini-([1-9]|10)\.([0-9]|10)-')
    if not pattern.match(model_id):
        return False
    
    # Exclude models containing specific keywords
    excluded_keywords = [
        'vision', 'thinking', 'tts', 'exp', 'embedding',
        'audio', 'native', 'dialog', 'live', 'image'
    ]
    
    model_lower = model_id.lower()
    for keyword in excluded_keywords:
        if keyword in model_lower:
            return False
    
    return True
