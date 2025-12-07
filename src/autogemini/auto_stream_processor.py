from enum import Enum


# 回调消息类型枚举
class CallbackMsgType(Enum):
    STREAM = "stream"  # AI流式输出
    TOOLCODE_START = "toolcode_start"  # ToolCode开始执行
    TOOLCODE_RESULT = "toolcode_result"  # 工具执行结果
    ERROR = "error"  # 异常
    INFO = "info"  # 其它流程信息


# API类型枚举
class APIType(Enum):
    GEMINI = "gemini"  # Gemini 原生API
    OPENAI = "openai"  # OpenAI 兼容API


from .gemini_chat import (
    stream_chat,
    stream_chat_openai,
    StreamCancellation,
    ChatMessage,
    MessageRole,
)
from .template import cot_template, ToolCodeInfo
from .tool_code import DefaultApi, eval_tool_code
import re
import asyncio
from typing import List, Optional, Callable, Tuple, Awaitable, Union


class AutoStreamProcessor:
    """
    自动流式处理器,处理AI流式输出中的ToolCode检测、执行和循环处理
    支持 Gemini 原生 API 和 OpenAI 兼容 API
    """

    def __init__(
        self,
        api_key: str,
        default_api: DefaultApi,
        model: str = "gemini-2.5-flash",
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 8192,
        top_p: float = 0.95,
        top_k: int = 40,
        timeout: float = 300.0,
        api_delay: float = 0.0,
        max_output_size: int = 65536,
        api_type: APIType = APIType.GEMINI,
        base_url: str = "https://api.openai-hk.com/v1",
        presence_penalty: float = 0.0,
        enable_multimodal: bool = True,
    ):
        """
        初始化自动流式处理器

        Args:
            api_key: API密钥 (Gemini API密钥 或 OpenAI兼容API密钥,如 hk-xxxxxx)
            default_api: ToolCode执行API处理器
            model: 使用的模型名称 (Gemini默认: gemini-2.5-flash, OpenAI如: gpt-5, claude-4-5-sonnet等)
            system_prompt: 系统提示词
            temperature: 采样温度
            max_tokens: 最大token数
            top_p: Top-p采样参数
            top_k: Top-k采样参数 (仅Gemini使用)
            timeout: 请求超时时间
            api_delay: API调用后的延迟时间(秒),用于避免速率限制
            max_output_size: ToolCode执行时print输出的最大字节数限制
            api_type: API类型, APIType.GEMINI 或 APIType.OPENAI
            base_url: OpenAI兼容API的基础URL (仅当api_type=APIType.OPENAI时使用)
            presence_penalty: 存在惩罚参数 (仅OpenAI使用)
            enable_multimodal: 是否启用多模态输入 (仅OpenAI兼容API使用, Gemini原生API默认支持)
        """
        self.api_key = api_key
        self.default_api = default_api
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.timeout = timeout
        self.api_delay = api_delay
        self.max_output_size = max_output_size
        self.api_type = api_type
        self.base_url = base_url
        self.presence_penalty = presence_penalty
        self.enable_multimodal = enable_multimodal

        # 对话历史
        self.history: List[ChatMessage] = []

        # 当前处理状态
        self.current_response = ""
        self.processing_complete = False

    async def process_conversation(
        self,
        user_message: Union[str, ChatMessage],
        callback: Optional[
            Callable[[str | Exception, "CallbackMsgType"], Awaitable[None]]
        ] = None,
        reset_history: bool = False,
        max_cycle_cost: int = 3,
        tool_code_timeout: float = 10.0,
        raw_response_callback: Optional[Callable[[object], Awaitable[None]]] = None,
    ) -> str:
        """
        处理完整的对话，包括ToolCode检测和执行循环

        Args:
            user_message: 用户消息，可以是字符串或ChatMessage对象（支持多媒体）
            callback: 回调函数，callback(chunk: str, msg_type: CallbackMsgType)
            reset_history: 是否重置对话历史
            max_cycle_cost: 最大循环次数
            tool_code_timeout: 工具代码超时时间
            raw_response_callback: 原始响应回调函数

        Returns:
            完整的AI响应
        """
        if reset_history:
            self.history.clear()

        # 处理不同类型的用户消息输入
        if isinstance(user_message, str):
            # 字符串消息：保持原有逻辑
            message_to_add = ChatMessage(
                MessageRole.USER,
                f"<reactAgentSegmentHeader>user_message</reactAgentSegmentHeader>{user_message}",
            )
        elif isinstance(user_message, ChatMessage):
            # ChatMessage对象：检查并调整格式
            if user_message.role != MessageRole.USER:
                raise ValueError("ChatMessage must have USER role")

            # 为ChatMessage添加header格式，保持与现有逻辑一致
            content = user_message.content
            if not content.startswith(
                "<reactAgentSegmentHeader>user_message</reactAgentSegmentHeader>"
            ):
                content = f"<reactAgentSegmentHeader>user_message</reactAgentSegmentHeader>{content}"

            message_to_add = ChatMessage(
                role=MessageRole.USER,
                content=content,
                media_files=getattr(user_message, "media_files", []),
            )
        else:
            raise TypeError(
                f"user_message must be str or ChatMessage, got {type(user_message)}"
            )

        # 添加消息到历史
        self.history.append(message_to_add)

        # 重置处理状态
        self.current_response = ""
        self.processing_complete = False

        # 开始处理循环 - 不再传递user_message
        final_response = await self._process_with_toolcode_loop(
            callback, max_cycle_cost, tool_code_timeout, raw_response_callback
        )

        # 最终响应就是累积的AI输出，不需要重复添加到历史
        # 因为在处理过程中已经逐步更新了历史

        return final_response

    async def _process_with_toolcode_loop(
        self,
        callback: Optional[
            Callable[[str | Exception, "CallbackMsgType"], Awaitable[None]]
        ] = None,
        max_cycle_cost: int = 3,
        tool_code_timeout: float = 10.0,
        raw_response_callback: Optional[Callable[[object], Awaitable[None]]] = None,
    ) -> str:
        """
        处理带有ToolCode循环检测的流式输出
        基于当前对话历史逐步构建AI响应，检测ToolCode并用assistant消息伪造返回值进行迭代
        截断after_toolcode块内容，只保留ToolCode前内容+执行结果

        callback说明：
            callback(chunk: str, msg_type: CallbackMsgType)
        """
        final_response = ""
        cost = 0
        while not self.processing_complete:
            stream_buffer = ""
            if cost > max_cycle_cost:
                raise RuntimeError(
                    f"Agent processing exceeded maximum cycle cost of {max_cycle_cost}."
                )
            cost += 1

            cancellation_token = StreamCancellation()

            async def stream_callback(chunk: str):
                nonlocal stream_buffer
                stream_buffer += chunk
                if callback:
                    await callback(chunk, CallbackMsgType.STREAM)
                # 检查是否出现了ToolCode块
                toolcode_match = self._detect_toolcode_in_call_block(stream_buffer)
                if toolcode_match:
                    cancellation_token.cancel()  # 取消流式输出

            # 基于当前历史请求AI
            try:
                if self.api_type == APIType.GEMINI:
                    # 使用 Gemini 原生 API
                    await stream_chat(
                        api_key=self.api_key,
                        callback=stream_callback,
                        history=self.history.copy(),
                        model=self.model,
                        system_prompt=self.system_prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        cancellation_token=cancellation_token,
                        timeout=self.timeout,
                        raw_response_callback=raw_response_callback,
                    )
                elif self.api_type == APIType.OPENAI:
                    # 使用 OpenAI 兼容 API
                    await stream_chat_openai(
                        api_key=self.api_key,
                        callback=stream_callback,
                        history=self.history.copy(),
                        model=self.model,
                        system_prompt=self.system_prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        presence_penalty=self.presence_penalty,
                        base_url=self.base_url,
                        cancellation_token=cancellation_token,
                        timeout=self.timeout,
                        enable_multimodal=self.enable_multimodal,
                        raw_response_callback=raw_response_callback,
                    )
                else:
                    raise ValueError(f"Unsupported api_type: {self.api_type}")

                # 添加API调用后的延迟,避免速率限制
                if self.api_delay > 0:
                    await asyncio.sleep(self.api_delay)

            except Exception as e:
                # stream_chat的异常直接抛出
                raise e

            if stream_buffer == "":
                continue  # 如果没有任何输出，继续循环
            ai_output = stream_buffer
            # 检查AI输出中是否有ToolCode
            toolcode_match = self._detect_toolcode_in_call_block(ai_output)
            if toolcode_match:
                toolcode_content, start_pos, end_pos = toolcode_match
                before_toolcode = ai_output[:start_pos]
                final_response += before_toolcode
                final_response += f"```tool_code\n{toolcode_content}\n```\n"

                async def handle_toolcode_result(result_text, is_error=False):
                    fake_result = f"<reactAgentSegmentHeader>system_feedback</reactAgentSegmentHeader>\nTool Result:\n{result_text}"
                    if cost >= max_cycle_cost:
                        fake_result += "<reactAgentSegmentHeader>system_feedback</reactAgentSegmentHeader>\nYOU HAVE REACHED THE MAXIMUM ITERATION COST. OUTPUT YOUR FINAL RESPONSE NOW."
                    self.history.append(
                        ChatMessage(
                            MessageRole.ASSISTANT,
                            before_toolcode
                            + "```tool_code\n"
                            + toolcode_content
                            + "\n```"
                            + fake_result,
                        )
                    )
                    self.history.append(
                        ChatMessage(
                            MessageRole.USER,
                            f"<reactAgentSegmentHeader>system_feedback</reactAgentSegmentHeader>\ncontinue ReAct processing by using `<reactAgentSegmentHeader>think</reactAgentSegmentHeader>`",
                        )
                    )
                    nonlocal final_response
                    final_response += fake_result
                    if callback:
                        cb_type = (
                            CallbackMsgType.ERROR
                            if is_error
                            else CallbackMsgType.TOOLCODE_RESULT
                        )
                        await callback(result_text, cb_type)

                try:
                    if callback:
                        await callback(toolcode_content, CallbackMsgType.TOOLCODE_START)
                    execution_results = await eval_tool_code(
                        toolcode_content,
                        self.default_api,
                        timeout=tool_code_timeout,
                        max_output_size=self.max_output_size,
                    )
                    result_text = self._format_execution_results(execution_results)
                    await handle_toolcode_result(result_text, is_error=False)
                except Exception as e:
                    await handle_toolcode_result(str(e), is_error=True)
                # 继续循环
                continue
            else:
                # 没有ToolCode，处理完成
                final_response += ai_output
                # 检查是否存在 `<reactAgentSegmentHeader>send_response_to_user</reactAgentSegmentHeader>`标记，如果final_response中不存在回复的内容，则继续
                if (
                    "<reactAgentSegmentHeader>send_response_to_user</reactAgentSegmentHeader>"
                    not in final_response
                ):
                    self.history.append(
                        ChatMessage(
                            MessageRole.ASSISTANT,
                            ai_output,
                        )
                    )
                    # 模拟系统消息，提示AI必须输出`<reactAgentSegmentHeader>send_response_to_user</reactAgentSegmentHeader>`标记
                    self.history.append(
                        ChatMessage(
                            MessageRole.USER,
                            f"<reactAgentSegmentHeader>system_alert</reactAgentSegmentHeader>\nNo `<reactAgentSegmentHeader>send_response_to_user</reactAgentSegmentHeader>` tag detected in the response. This response is invalid. Please ensure your final response includes the `<reactAgentSegmentHeader>send_response_to_user</reactAgentSegmentHeader>` tag and try again.",
                        )
                    )
                    if callback:
                        await callback(
                            "<Detected no response tag>", CallbackMsgType.INFO
                        )
                    continue
                self.processing_complete = True
                self.history.append(ChatMessage(MessageRole.ASSISTANT, ai_output))
        return final_response

    def _detect_toolcode_in_call_block(
        self, text: str
    ) -> Optional[Tuple[str, int, int]]:
        """
        精确检测AI输出中最后一个call_tool_code块内部的tool_code块。

        Args:
            text: 要检测的完整AI流式输出。

        Returns:
            如果成功找到，返回 (toolcode_content, start_pos, end_pos)
        """
        # 步骤 1: 使用 rfind() 高效、安全地定位最后一个 call_tool_code 块的头部
        # 这避免了依赖一个可能尚未出现的终止标签。
        call_block_header = (
            "<reactAgentSegmentHeader>call_tool_code</reactAgentSegmentHeader>"
        )
        last_call_block_start_pos = text.rfind(call_block_header)

        # 如果没有找到任何 call_tool_code 块，直接返回
        if last_call_block_start_pos == -1:
            return None

        # 步骤 2: 只在最后一个 call_tool_code 块之后的内容中进行搜索
        # 这就是我们的有效搜索区域
        search_region_text = text[last_call_block_start_pos:]

        # 定义用于匹配 ```tool_code...``` 的正则表达式
        tool_code_pattern = re.compile(
            r"```tool_code\n(.*?)\n```", re.DOTALL | re.MULTILINE
        )

        # 在限定的区域内搜索 tool_code
        tool_code_match = tool_code_pattern.search(search_region_text)

        # 如果在限定区域内没有找到 tool_code，返回 None
        if not tool_code_match:
            return None

        # 步骤 3: 计算并返回全局坐标
        toolcode_content = tool_code_match.group(1)

        # tool_code 的全局起始位置 = call_tool_code头的起始位置 + tool_code在搜索区域内的起始位置
        global_start_pos = last_call_block_start_pos + tool_code_match.start()
        global_end_pos = last_call_block_start_pos + tool_code_match.end()

        return (toolcode_content, global_start_pos, global_end_pos)

    def _format_execution_results(self, results: List[dict]) -> str:
        """
        格式化ToolCode执行结果

        Args:
            results: 执行结果列表

        Returns:
            格式化后的结果字符串
        """
        if not results:
            return "[无执行结果]"

        formatted_results = []
        for result in results:
            args = result.get("args", ())
            if args:
                # 将每个参数转换为字符串并收集
                for arg in args:
                    formatted_results.append(str(arg))

        return "\n".join(formatted_results) if formatted_results else "[Invalid result]"

    def load_history(self, history: List[ChatMessage]):
        """
        加载对话历史

        Args:
            history: ChatMessage列表
        """
        self.history = history.copy()

    def get_history(self) -> List[ChatMessage]:
        """获取完整的对话历史"""
        return self.history.copy()

    def clear_history(self):
        """清空对话历史"""
        self.history.clear()

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.system_prompt = prompt

    def create_user_message(
        self, content: str, media_files: Optional[List] = None
    ) -> ChatMessage:
        """
        创建用户消息的便利方法

        Args:
            content: 消息内容
            media_files: 媒体文件列表（文件路径字符串或MediaFile对象）

        Returns:
            ChatMessage对象
        """
        from .gemini_chat import MediaFile

        message = ChatMessage(role=MessageRole.USER, content=content)

        if media_files:
            # 确保media_files属性存在
            if not hasattr(message, "media_files"):
                message.media_files = []

            for media_item in media_files:
                if isinstance(media_item, str):
                    # 文件路径
                    media_file = MediaFile(file_path=media_item)
                    message.media_files.append(media_file)
                elif hasattr(media_item, "file_path") or hasattr(media_item, "data"):
                    # MediaFile对象
                    message.media_files.append(media_item)
                else:
                    raise ValueError(f"Invalid media item type: {type(media_item)}")

        return message


# TAGS描述，指导模型生成易于转换为纯文本的、干净的HTML
CLEAN_HTML_TAGS = """
Your final response must be formatted using a limited set of simple, semantic HTML tags.
This ensures the output can be displayed correctly in a web view AND be easily converted to clean plain text.
**Strictly AVOID** any tags or attributes related to complex layout, styling, or media.

---

### **Allowed Tags:**

- `<p>...</p>`: For standard paragraphs of text.
- `<h1>...</h1>`, `<h2>...</h2>`, `<h3>...</h3>`: For section headings. Do not use for styling text.
- `<strong>...</strong>` or `<b>...</b>`: For strong emphasis.
- `<em>...</em>` or `<i>...</i>`: For general emphasis.
- `<ul>...</ul>`: For unordered (bulleted) lists.
- `<ol>...</ol>`: For ordered (numbered) lists.
- `<li>...</li>`: For individual list items (must be inside `<ul>` or `<ol>`).
- `<code>...</code>`: For short, inline code snippets.
- `<pre>...</pre>`: For longer, multi-line code blocks. It's good practice to wrap the code inside with a `<code>` tag, like `<pre><code>...</code></pre>`.
- `<br>`: To insert a single line break where a new paragraph is not appropriate.
- `<a href="...">...</a>`: For hyperlinks. The `href` attribute is the only allowed attribute.

### **Strictly Forbidden:**

- **Layout Tags:** Do NOT use `<table>`, `<div>`, `<span>`, `<thead>`, `<tbody>`, `<tr>`, `<td>`, `<th>`.
- **Styling:** Do NOT use any attributes like `style`, `class`, `id`, `align`, `color`, `bgcolor`. Do NOT use tags like `<font>`, `<center>`.
- **Media:** Do NOT use `<img>`, `<video>`, `<audio>`, `<svg>`.
- **Other:** Do NOT use `<form>`, `<input>`, `<button>`, or `<script>`.
"""


# 便利函数：创建带有COT模板的处理器
def create_cot_processor(
    api_key: str,
    default_api: DefaultApi,
    tool_codes: List[ToolCodeInfo],
    character_description: str = "你是一个智能助手，能够执行工具代码并提供准确的回答。",
    respond_tags_description: str = CLEAN_HTML_TAGS,
    **kwargs,
) -> AutoStreamProcessor:
    """
    创建带有COT模板的自动流式处理器

    Args:
        api_key: API密钥
        default_api: ToolCode API处理器
        tool_codes: 可用的工具代码列表
        character_description: 角色描述
        respond_tags_description: 响应标签描述
        **kwargs: 其他参数传递给AutoStreamProcessor，包括api_delay等

    Returns:
        配置好的AutoStreamProcessor实例
    """
    # 生成COT系统提示词
    system_prompt = cot_template(
        tool_codes, character_description, respond_tags_description
    )

    return AutoStreamProcessor(
        api_key=api_key, default_api=default_api, system_prompt=system_prompt, **kwargs
    )
