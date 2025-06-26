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
        model: str = "gemini-2.0-flash-thinking-exp",
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

        # ToolCode检测正则表达式
        self.tool_code_pattern = re.compile(r"```tool_code\n(.*?)\n```", re.DOTALL)

        # 当前处理状态
        self.current_response = ""
        self.processing_complete = False

    async def process_conversation(
        self,
        user_message: str,
        callback: Optional[Callable[[str], None]] = None,
        reset_history: bool = False,
    ) -> str:
        """
        处理完整的对话，包括ToolCode检测和执行循环

        Args:
            user_message: 用户消息
            callback: 流式输出回调函数
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
        self, callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        处理带有ToolCode循环检测的流式输出
        基于当前对话历史逐步构建AI响应，检测并替换ToolCode
        """
        accumulated_response = ""

        while not self.processing_complete:
            # 创建流式输出缓冲区
            stream_buffer = ""

            # 定义内部回调函数来监控流式输出
            def stream_callback(chunk: str):
                nonlocal stream_buffer, accumulated_response
                stream_buffer += chunk

                # 实时更新积累响应
                current_total = accumulated_response + stream_buffer

                # 检查积累的总响应中是否有完整的ToolCode
                toolcode_match = self._detect_complete_toolcode(current_total)
                if toolcode_match:
                    # 发现完整ToolCode，不需要取消，让流式输出自然完成
                    pass

                # 调用用户提供的回调
                if callback:
                    callback(chunk)

            try:
                # 开始流式输出 - 完全基于历史记录
                response = await stream_chat(
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

                # 更新积累响应
                accumulated_response += stream_buffer

                # 检查积累的总响应中是否有ToolCode需要执行
                toolcode_match = self._detect_complete_toolcode(accumulated_response)
                if toolcode_match:
                    # 执行ToolCode并替换积累响应中的ToolCode
                    accumulated_response = await self._execute_and_replace_toolcode(
                        accumulated_response, toolcode_match, callback
                    )

                    # 更新历史记录，基于替换后的响应继续迭代
                    self._update_history_with_current_response(accumulated_response)

                    # 继续处理循环（可能还有更多ToolCode）
                    continue
                else:
                    # 没有ToolCode，处理完成
                    self.processing_complete = True
                    self._update_history_with_final_response(accumulated_response)

            except Exception as e:
                # 处理异常
                if callback:
                    callback(f"\n[处理异常: {str(e)}]\n")
                accumulated_response += stream_buffer
                self.processing_complete = True
                self._update_history_with_final_response(accumulated_response)

        return accumulated_response

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

    async def _execute_and_replace_toolcode(
        self,
        accumulated_response: str,
        toolcode_match: Tuple[str, int, int],
        callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        执行ToolCode并在积累响应中替换它

        Args:
            accumulated_response: 当前积累的AI响应
            toolcode_match: ToolCode匹配结果 (content, start, end)
            callback: 回调函数

        Returns:
            替换ToolCode后的响应文本
        """
        toolcode_content, start_pos, end_pos = toolcode_match

        try:
            # 执行ToolCode
            execution_results = eval_tool_code(toolcode_content, self.default_api)

            # 格式化执行结果
            result_text = self._format_execution_results(execution_results)

            # 通知用户ToolCode正在执行
            if callback:
                callback(f"\n[执行ToolCode...]\n{result_text}\n")

            # 在积累响应中替换ToolCode为执行结果
            # 移除整个 ```tool_code...``` 块，替换为执行结果
            before_toolcode = accumulated_response[:start_pos]
            after_toolcode = accumulated_response[end_pos:]

            # 替换：移除ToolCode语法，保留执行结果
            replaced_response = before_toolcode + result_text + after_toolcode

            return replaced_response

        except Exception as e:
            error_msg = f"[ToolCode执行失败: {str(e)}]"
            if callback:
                callback(f"\n{error_msg}\n")

            # 替换为错误信息
            before_toolcode = accumulated_response[:start_pos]
            after_toolcode = accumulated_response[end_pos:]
            return before_toolcode + error_msg + after_toolcode

    def _update_history_with_current_response(self, current_response: str):
        """
        更新历史记录，用当前响应替换最后的AI消息
        用于ToolCode执行后的中间状态

        Args:
            current_response: 当前的AI响应内容
        """
        # 移除历史中最后的ASSISTANT消息（如果存在）
        if self.history and self.history[-1].role == MessageRole.ASSISTANT:
            self.history.pop()

        # 添加更新后的AI响应
        self.history.append(ChatMessage(MessageRole.ASSISTANT, current_response))

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
                # 将结果转换为字符串
                result_str = " ".join(str(arg) for arg in args)
                formatted_results.append(result_str)

        return "\n".join(formatted_results) if formatted_results else "[无有效输出]"

    def _update_history_with_toolcode_result(self, toolcode: str, result: str):
        """
        更新对话历史，记录ToolCode执行

        在新架构中，这个方法主要用于调试和日志记录
        实际的历史更新通过_update_history_with_current_response完成

        Args:
            toolcode: 执行的ToolCode内容
            result: 执行结果
        """
        # 在新架构中，我们不再单独添加ToolCode执行记录到历史
        # 因为执行结果已经集成到AI的响应中
        # 这里可以用于日志记录或调试
        pass

    def _update_history_with_partial_response(self, partial_response: str):
        """
        更新历史记录，添加AI的部分响应
        用于ToolCode执行后的中间状态

        注意：此方法已被_update_history_with_current_response替代

        Args:
            partial_response: AI的部分响应内容
        """
        self._update_history_with_current_response(partial_response)

    def _update_history_with_final_response(self, final_response: str):
        """
        更新历史记录，添加AI的最终完整响应

        Args:
            final_response: AI的最终完整响应
        """
        # 移除历史中最后的ASSISTANT消息（如果存在）
        if self.history and self.history[-1].role == MessageRole.ASSISTANT:
            self.history.pop()

        # 添加最终AI响应
        self.history.append(ChatMessage(MessageRole.ASSISTANT, final_response))

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
