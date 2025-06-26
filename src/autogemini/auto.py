from enum import Enum


# 回调消息类型枚举
class CallbackMsgType(Enum):
    STREAM = "stream"  # AI流式输出
    TOOLCODE_START = "toolcode_start"  # ToolCode开始执行
    TOOLCODE_RESULT = "toolcode_result"  # 工具执行结果
    ERROR = "error"  # 异常
    INFO = "info"  # 其它流程信息


from .gemini_chat import stream_chat, StreamCancellation, ChatMessage, MessageRole
from .template import cot_template, ToolCodeInfo
from .tool_code import DefaultApi, eval_tool_code
import re
import asyncio
from typing import List, Optional, Callable, Tuple, Any
import json


class AutoStreamProcessor:
    """
    自动流式处理器，处理AI流式输出中的ToolCode检测、执行和循环处理
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
    ):
        """
        初始化自动流式处理器

        Args:
            api_key: Gemini API密钥
            default_api: ToolCode执行API处理器
            model: 使用的模型名称
            system_prompt: 系统提示词
            temperature: 采样温度
            max_tokens: 最大token数
            top_p: Top-p采样参数
            top_k: Top-k采样参数
            timeout: 请求超时时间
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

        # 对话历史
        self.history: List[ChatMessage] = []

        # ToolCode检测正则表达式，兼容前面有空格或tab的情况
        self.tool_code_pattern = re.compile(
            r"^[ \t]*```tool_code\n(.*?)\n[ \t]*```", re.DOTALL | re.MULTILINE
        )

        # 当前处理状态
        self.current_response = ""
        self.processing_complete = False

    async def process_conversation(
        self,
        user_message: str,
        callback: Optional[Callable[[str, "CallbackMsgType"], None]] = None,
        reset_history: bool = False,
    ) -> str:
        """
        处理完整的对话，包括ToolCode检测和执行循环

        Args:
            user_message: 用户消息
            callback: 回调函数，callback(chunk: str, msg_type: CallbackMsgType)
            reset_history: 是否重置对话历史

        Returns:
            完整的AI响应
        """
        if reset_history:
            self.history.clear()

        # 只在开始时添加用户消息到历史
        self.history.append(ChatMessage(MessageRole.USER, user_message))

        # 重置处理状态
        self.current_response = ""
        self.processing_complete = False

        # 开始处理循环 - 不再传递user_message
        final_response = await self._process_with_toolcode_loop(callback)

        # 最终响应就是累积的AI输出，不需要重复添加到历史
        # 因为在处理过程中已经逐步更新了历史

        return final_response

    async def _process_with_toolcode_loop(
        self, callback: Optional[Callable[[str, "CallbackMsgType"], None]] = None
    ) -> str:
        """
        处理带有ToolCode循环检测的流式输出
        基于当前对话历史逐步构建AI响应，检测ToolCode并用assistant消息伪造返回值进行迭代
        截断after_toolcode块内容，只保留ToolCode前内容+执行结果

        callback说明：
            callback(chunk: str, msg_type: CallbackMsgType)
        """
        final_response = ""
        while not self.processing_complete:
            stream_buffer = ""

            def stream_callback(chunk: str):
                nonlocal stream_buffer
                stream_buffer += chunk
                if callback:
                    callback(chunk, CallbackMsgType.STREAM)

            try:
                # 基于当前历史请求AI
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
                    timeout=self.timeout,
                )
                ai_output = stream_buffer
                # 检查AI输出中是否有ToolCode
                toolcode_match = self._detect_complete_toolcode(ai_output)
                if toolcode_match:
                    toolcode_content, start_pos, end_pos = toolcode_match
                    before_toolcode = ai_output[:start_pos]

                    self.history.append(
                        ChatMessage(
                            MessageRole.ASSISTANT,
                            before_toolcode
                            + "```tool_code\n"
                            + toolcode_content
                            + "\n```",
                        )
                    )
                    final_response += before_toolcode
                    try:
                        if callback:
                            callback(toolcode_content, CallbackMsgType.TOOLCODE_START)
                        execution_results = eval_tool_code(
                            toolcode_content, self.default_api
                        )
                        result_text = self._format_execution_results(execution_results)
                        fake_result = (
                            f"```result(invisible to user)\n{result_text}\n```"
                        )
                        self.history.append(ChatMessage(MessageRole.USER, fake_result))
                        if callback:
                            callback(result_text, CallbackMsgType.TOOLCODE_RESULT)
                    except Exception as e:
                        fake_result = f"```error(invisible to user)\n{str(e)}\n```"
                        self.history.append(ChatMessage(MessageRole.USER, fake_result))
                        if callback:
                            callback(str(e), CallbackMsgType.ERROR)
                    # 继续循环
                    continue
                else:
                    # 没有ToolCode，处理完成
                    final_response += ai_output
                    self.processing_complete = True
                    self.history.append(ChatMessage(MessageRole.ASSISTANT, ai_output))

            except Exception as e:
                if callback:
                    callback(f"[处理异常: {str(e)}]", CallbackMsgType.ERROR)
                self.processing_complete = True
                self.history.append(
                    ChatMessage(MessageRole.ASSISTANT, f"[处理异常: {str(e)}]")
                )
        return final_response

    def _detect_complete_toolcode(self, text: str) -> Optional[Tuple[str, int, int]]:
        """
        检测文本中是否包含完整的ToolCode块

        Args:
            text: 要检测的文本

        Returns:
            如果找到完整ToolCode，返回(toolcode_content, start_pos, end_pos)
            否则返回None
        """
        match = self.tool_code_pattern.search(text)
        if match:
            return (match.group(1), match.start(), match.end())
        return None

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

        return "\n".join(formatted_results) if formatted_results else "[无有效输出]"

    def get_conversation_history(self) -> List[ChatMessage]:
        """获取完整的对话历史"""
        return self.history.copy()

    def clear_history(self):
        """清空对话历史"""
        self.history.clear()

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.system_prompt = prompt


# 便利函数：创建带有COT模板的处理器
def create_cot_processor(
    api_key: str,
    default_api: DefaultApi,
    tool_codes: List[ToolCodeInfo],
    character_description: str = "你是一个智能助手，能够执行工具代码并提供准确的回答。",
    **kwargs,
) -> AutoStreamProcessor:
    """
    创建带有COT模板的自动流式处理器

    Args:
        api_key: API密钥
        default_api: ToolCode API处理器
        tool_codes: 可用的工具代码列表
        character_description: 角色描述
        **kwargs: 其他参数传递给AutoStreamProcessor

    Returns:
        配置好的AutoStreamProcessor实例
    """
    # 生成COT系统提示词
    system_prompt = cot_template(tool_codes, character_description)

    return AutoStreamProcessor(
        api_key=api_key, default_api=default_api, system_prompt=system_prompt, **kwargs
    )
