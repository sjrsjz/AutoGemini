from enum import Enum
import re
import asyncio
import xml.etree.ElementTree as ET
from typing import List, Optional, Callable, Awaitable, Union

# 依赖项
from .gemini_chat import stream_chat, StreamCancellation, ChatMessage, MessageRole
from .template import cot_template, ToolCodeInfo, parse_agent_output, ParsedBlock
from .tool_code import DefaultApi, eval_tool_code


# 回调消息类型枚举
class CallbackMsgType(Enum):
    STREAM = "stream"
    TOOLCODE_START = "toolcode_start"
    TOOLCODE_RESULT = "toolcode_result"
    ERROR = "error"
    INFO = "info"
    FINAL_RESPONSE = "final_response"


class AutoStreamProcessor:
    """
    自动流式处理器，严格遵循您原始设计的“检测-打断-拼接”逻辑，
    并适配新的 `<do action="...">` 标签格式。
    """

    def __init__(
        self,
        api_key: str,
        default_api: DefaultApi,
        model: str = "gemini-1.5-flash-latest",
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 8192,
        top_p: float = 0.95,
        top_k: int = 40,
        timeout: float = 300.0,
        api_delay: float = 0.0,
        max_output_size: int = 65536,
    ):
        # 构造函数完全不变
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
        self.history: List[ChatMessage] = []
        # self.current_response 和 self.processing_complete 将在循环内部管理，无需作为类属性

    async def process_conversation(
        self,
        user_message: Union[str, ChatMessage],
        callback: Optional[
            Callable[[Union[str, Exception], "CallbackMsgType"], Awaitable[None]]
        ] = None,
        reset_history: bool = False,
        max_cycle_cost: int = 3,
        tool_code_timeout: float = 10.0,
    ) -> str:
        if reset_history or not self.history:
            self.history.clear()
            if self.system_prompt:
                self.history.append(ChatMessage(MessageRole.USER, self.system_prompt))
                self.history.append(
                    ChatMessage(
                        MessageRole.ASSISTANT,
                        '<do action="think">Understood. I will act as an Agent and follow all instructions.</do>',
                    )
                )

        if isinstance(user_message, str):
            user_message_obj = ChatMessage(MessageRole.USER, user_message)
        elif (
            isinstance(user_message, ChatMessage)
            and user_message.role == MessageRole.USER
        ):
            user_message_obj = user_message
        else:
            raise TypeError("user_message must be str or a USER role ChatMessage")

        # **适配**: 使用新的<conversation>标签
        self.history.append(
            ChatMessage(
                role=user_message_obj.role,
                content=f"<conversation>\nUser: {user_message_obj.content}\n</conversation>",
                media_files=user_message_obj.media_files,
            )
        )

        # 调用核心循环
        final_response_trajectory = await self._process_with_toolcode_loop(
            callback, max_cycle_cost, tool_code_timeout
        )
        return final_response_trajectory

    async def _process_with_toolcode_loop(
        self,
        callback: Optional[Callable],
        max_cycle_cost: int,
        tool_code_timeout: float,
    ) -> str:
        """
        处理带有ToolCode循环检测的流式输出，逻辑与您的原始版本保持一致。
        """
        final_response = ""
        processing_complete = False
        cost = 0

        while not processing_complete:
            if cost >= max_cycle_cost:
                # 您的原始代码在超过成本后会继续一次，但会添加警告。这里我们直接停止以防无限循环。
                if callback:
                    await callback(
                        "Exceeded maximum cycle cost.", CallbackMsgType.ERROR
                    )
                break

            cost += 1
            stream_buffer = ""
            cancellation_token = StreamCancellation()

            async def stream_callback(chunk: str):
                nonlocal stream_buffer
                stream_buffer += chunk
                if callback:
                    await callback(chunk, CallbackMsgType.STREAM)

                # **适配**: 使用新的打断点检测函数
                if self._detect_tool_code_in_stream(stream_buffer):
                    cancellation_token.cancel()

            # 发起流式请求 (与原始代码相同)
            await stream_chat(
                api_key=self.api_key,
                callback=stream_callback,
                history=self.history.copy(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                cancellation_token=cancellation_token,
                timeout=self.timeout,
            )

            if self.api_delay > 0:
                await asyncio.sleep(self.api_delay)

            ai_output = stream_buffer
            toolcode_match_result = self._detect_tool_code_in_stream(ai_output)

            if toolcode_match_result:
                toolcode_content, full_tool_tag, start_pos, end_pos = (
                    toolcode_match_result
                )

                # **逻辑保留**: 分割出工具调用之前的内容
                before_toolcode = ai_output[:start_pos]

                # **逻辑保留**: 累加到final_response
                final_response += before_toolcode
                final_response += full_tool_tag  # 添加完整的<do>标签

                # **逻辑保留**: 定义一个内部函数来处理结果，以避免代码重复
                async def handle_toolcode_result(result_text, is_error=False):
                    nonlocal final_response
                    # **适配**: 构建新的伪造系统反馈
                    feedback_str = f"\n<system_feedback>\n<tool_result>{result_text}</tool_result>\n</system_feedback>"
                    if cost >= max_cycle_cost:
                        feedback_str += "\n<system_alert>WARNING: You have reached the maximum cycle cost. You MUST provide a final response in your next turn.</system_alert>"

                    # **逻辑保留**: 累加到final_response
                    final_response += feedback_str

                    # **逻辑保留**: 更新历史记录
                    # 1. 添加AI本回合的输出
                    self.history.append(
                        ChatMessage(
                            MessageRole.ASSISTANT, before_toolcode + full_tool_tag
                        )
                    )
                    # 2. 添加伪造的系统反馈
                    self.history.append(ChatMessage(MessageRole.USER, feedback_str))

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

                # **逻辑保留**: 继续下一次循环
                continue
            else:
                # **逻辑保留**: 没有工具调用，这是最后一回合
                final_response += ai_output

                # **适配**: 使用新的结束标志
                if '<do action="response">' in final_response:
                    processing_complete = True
                    self.history.append(ChatMessage(MessageRole.ASSISTANT, ai_output))
                else:
                    # **逻辑保留**: 如果没有结束标志，认为AI还没完成，伪造一个提示让它继续
                    self.history.append(ChatMessage(MessageRole.ASSISTANT, ai_output))
                    self.history.append(
                        ChatMessage(
                            MessageRole.USER,
                            "<system_alert>You have not finished the task. Please either call another tool or provide a final response using '<do action=\"response\">'.</system_alert>",
                        )
                    )
                    if callback:
                        await callback(
                            "<Detected no response tag, continuing>",
                            CallbackMsgType.INFO,
                        )
                    # continue 会被 while 循环自动处理

        return final_response

    def _detect_tool_code_in_stream(self, text: str) -> Optional[tuple]:
        """
        适配: 精确检测流中的<do action="call_tool_code">...</do>块。
        返回 (代码内容, 完整的标签, 开始位置, 结束位置)
        """
        pattern = re.compile(
            r'(<do action="call_tool_code".*?>)(.*?)(</do>)', re.DOTALL
        )
        match = pattern.search(text)
        if match:
            full_tag = match.group(0)
            content = match.group(2).strip()
            start_pos = match.start()
            end_pos = match.end()
            return content, full_tag, start_pos, end_pos
        return None

    def _format_execution_results(self, results: List[dict]) -> str:
        """格式化ToolCode执行结果"""
        if not results:
            return "[No output was produced]"
        outputs = [
            str(r.get("args", ("",))[0])
            for r in results
            if r.get("type") == "print" and r.get("args")
        ]
        if not outputs and results:
            # 专门为您的 WolframAlpha 工具的返回格式做的适配
            if (
                isinstance(results, list)
                and len(results) > 0
                and isinstance(results[0], dict)
            ):
                return str(results)  # 以字符串形式返回整个结果列表
        return (
            "\n".join(outputs)
            if outputs
            else "[The tool executed but produced no printable output.]"
        )

    # 其他辅助函数保持不变
    def load_history(self, history: List[ChatMessage]):
        self.history = history.copy()

    def get_history(self) -> List[ChatMessage]:
        return self.history.copy()

    def clear_history(self):
        self.history.clear()

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt


# create_user_message 保持不变
def create_user_message(
    content: str, media_files: Optional[List] = None
) -> ChatMessage:
    from .gemini_chat import MediaFile

    message = ChatMessage(role=MessageRole.USER, content=content)
    if media_files:
        message.media_files = []
        for media_item in media_files:
            if isinstance(media_item, str):
                message.media_files.append(MediaFile(file_path=media_item))
            else:
                message.media_files.append(media_item)
    return message


# 便利函数：创建带有COT模板的处理器
def create_cot_processor(
    api_key: str,
    default_api: DefaultApi,
    tool_codes: List[ToolCodeInfo],
    character_description: str = "You are an intelligent AI Assistant.",
    respond_tags_description: str = "Use simple HTML for the final response.",
    **kwargs,
) -> AutoStreamProcessor:
    """创建带有成本优化COT模板的Agent处理器。"""
    system_prompt = cot_template(
        tool_codes, character_description, respond_tags_description
    )
    return AutoStreamProcessor(
        api_key=api_key, default_api=default_api, system_prompt=system_prompt, **kwargs
    )


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
