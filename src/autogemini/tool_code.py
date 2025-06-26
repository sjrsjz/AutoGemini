from typing import Any, Callable, List, Dict, Tuple, Optional
import ast
import operator
import re


class DefaultApi:
    def __init__(self, default_handler: Callable[[str, Any], Any]) -> None:
        self._handlers = {}
        self._default_handler = default_handler

    def __getattr__(self, name: str) -> Any:
        """Handle API method calls for unknown attributes."""

        def method(*args, **kwargs):
            """Handle API method calls."""
            if name in self._handlers:
                return self._handlers[name](*args, **kwargs)
            return self._default_handler(name, *args, **kwargs)

        return method

    def add_handler(self, name: str, handler: Callable[[str, Any], Any]) -> None:
        """Add a handler for a specific API method."""
        self._handlers[name] = handler

    def remove_handler(self, name: str) -> None:
        """Remove a handler for a specific API method."""
        if name in self._handlers:
            del self._handlers[name]


def eval_tool_code(tool_code: str, default_api: Optional[DefaultApi] = None) -> Any:
    """Evaluate tool code in a restricted context."""
    # 如果没有提供default_api，创建一个默认的
    if default_api is None:
        default_api = DefaultApi(
            lambda name, *args, **kwargs: f"Default handler called for {name} with args {args} and kwargs {kwargs}"
        )

    # 结果收集器
    results = []

    def safe_print(*args, **kwargs):
        results.append({"args": args, "kwargs": kwargs})

    # 创建受限的安全执行环境
    safe_globals = {
        "__builtins__": {
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "print": safe_print,
        },
        "default_api": default_api,
    }

    # 首先检查代码的安全性
    try:
        # 解析AST来检查是否包含危险操作
        tree = ast.parse(tool_code)
        _validate_ast_safety(tree)

        # 在受限环境中执行
        exec(tool_code, safe_globals, {})
    except Exception as e:
        raise RuntimeError(f"Error evaluating tool code: {e}")

    return results


def _validate_ast_safety(node):
    """验证AST节点的安全性，禁止危险操作"""
    dangerous_nodes = {
        ast.Import: "Import statements are not allowed",
        ast.ImportFrom: "Import statements are not allowed",
        ast.FunctionDef: "Function definitions are not allowed",
        ast.ClassDef: "Class definitions are not allowed",
        ast.Global: "Global statements are not allowed",
        ast.Nonlocal: "Nonlocal statements are not allowed",
    }

    dangerous_names = {
        "open",
        "file",
        "input",
        "raw_input",
        "execfile",
        "reload",
        "compile",
        "eval",
        "exec",
        "__import__",
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "help",
        "copyright",
        "credits",
        "license",
        "quit",
        "exit",
    }

    for child in ast.walk(node):
        # 检查危险的AST节点类型
        for dangerous_type, message in dangerous_nodes.items():
            if isinstance(child, dangerous_type):
                raise ValueError(f"Unsafe operation detected: {message}")

        # 检查危险的函数名和变量名
        if isinstance(child, ast.Name) and child.id in dangerous_names:
            raise ValueError(f"Unsafe name detected: {child.id}")

        # 检查属性访问，防止访问私有属性或危险方法
        if isinstance(child, ast.Attribute):
            if child.attr.startswith("_"):
                raise ValueError(
                    f"Access to private attribute not allowed: {child.attr}"
                )

            dangerous_attrs = {"__class__", "__bases__", "__subclasses__", "__mro__"}
            if child.attr in dangerous_attrs:
                raise ValueError(
                    f"Access to dangerous attribute not allowed: {child.attr}"
                )


def extract_tool_code(tool_code: str):
    """Extract tool codes from a string."""
    # tool_code是由 ```tool_code 和 ``` 包裹的代码块
    import re

    # 匹配 ```tool_code 开始和 ``` 结束的代码块
    pattern = r"```tool_code\n(.*?)\n```"
    matches = re.findall(pattern, tool_code, re.DOTALL)

    # 返回所有找到的代码片段
    return matches


class ToolCodeProcessor:
    """处理流式输出中的tool_code执行和替换"""

    def __init__(self, default_api: DefaultApi):
        self.default_api = default_api
        self.buffer = ""
        self.tool_code_pattern = re.compile(r"```tool_code\n(.*?)\n```", re.DOTALL)

    def process_stream_chunk(self, chunk: str) -> Tuple[str, bool]:
        """
        处理流式输出的一个chunk

        Args:
            chunk: 新的流式输出片段

        Returns:
            (processed_output, should_interrupt): 处理后的输出和是否需要中断流式输出
        """
        self.buffer += chunk

        # 检查是否有完整的tool_code块
        match = self.tool_code_pattern.search(self.buffer)
        if match:
            # 找到完整的tool_code，需要中断流式输出
            tool_code = match.group(1)
            start_pos = match.start()
            end_pos = match.end()

            # 执行tool_code
            try:
                execution_result = self._execute_tool_code(tool_code)
                replacement = self._format_execution_result(execution_result)
            except Exception as e:
                replacement = f"[Tool execution error: {str(e)}]"

            # 替换tool_code块为执行结果
            processed_output = (
                self.buffer[:start_pos] + replacement + self.buffer[end_pos:]
            )

            # 更新buffer，移除已处理的部分
            self.buffer = processed_output

            return processed_output, True  # 需要中断

        return self.buffer, False  # 不需要中断

    def _execute_tool_code(self, tool_code: str) -> List[Dict]:
        """执行tool_code并返回结果"""
        return eval_tool_code(tool_code, self.default_api)

    def _format_execution_result(self, results: List[Dict]) -> str:
        """格式化执行结果为字符串"""
        if not results:
            return ""

        # 假设最后一个print调用是我们要的结果
        # 通常AI会写类似 print(default_api.function(...)) 这样的代码
        last_result = results[-1]
        args = last_result.get("args", ())

        if args:
            # 返回第一个参数作为结果
            return str(args[0])

        return ""

    def reset_buffer(self):
        """重置缓冲区"""
        self.buffer = ""

    def get_remaining_buffer(self) -> str:
        """获取剩余的缓冲区内容"""
        return self.buffer


def process_streaming_response(
    stream_generator,
    default_api: DefaultApi,
    on_tool_execution: Optional[Callable[[str, Any], None]] = None,
) -> str:
    """
    处理完整的流式响应，自动处理tool_code执行和替换

    Args:
        stream_generator: 流式输出生成器
        default_api: API处理器实例
        on_tool_execution: 工具执行时的回调函数

    Returns:
        完整的处理后响应
    """
    processor = ToolCodeProcessor(default_api)
    final_output = ""

    for chunk in stream_generator:
        processed_chunk, should_interrupt = processor.process_stream_chunk(chunk)

        if should_interrupt:
            # 有tool_code被执行，记录执行事件
            if on_tool_execution:
                on_tool_execution("tool_executed", processed_chunk)

            # 这里应该中断当前流，并用处理后的内容重新开始流式请求
            # 具体实现取决于您使用的LLM API
            final_output = processed_chunk
            break
        else:
            final_output = processed_chunk

    return final_output


# 使用示例和辅助函数


def create_streaming_handler(default_api: DefaultApi):
    """
    创建一个流式处理句柄，用于实际的LLM API集成

    使用示例:
    ```python
    # 设置API处理器
    def my_api_handler(method_name, *args, **kwargs):
        if method_name == "get_weather":
            return f"Weather in {args[0]}: 晴天 25°C"
        elif method_name == "calculate":
            return eval(args[0])  # 简单计算
        return f"Unknown method: {method_name}"

    api = DefaultApi(my_api_handler)
    handler = create_streaming_handler(api)

    # 模拟流式输出处理
    stream_chunks = [
        "根据您的需求，我来查询天气信息：\n\n```tool_code\nprint(default_api.get_weather('北京'))\n```",
        "\n\n基于查询结果，今天是个好天气！"
    ]

    for chunk in stream_chunks:
        result, should_interrupt = handler.process_stream_chunk(chunk)
        if should_interrupt:
            print("需要中断流式输出，执行工具代码")
            print("处理后的结果:", result)
            break
    ```
    """
    return ToolCodeProcessor(default_api)


def extract_and_execute_all_tool_codes(text: str, default_api: DefaultApi) -> str:
    """
    一次性提取并执行文本中的所有tool_code块

    Args:
        text: 包含tool_code的文本
        default_api: API处理器

    Returns:
        替换所有tool_code后的文本
    """
    processor = ToolCodeProcessor(default_api)

    # 模拟流式处理，但一次性处理完整文本
    result, _ = processor.process_stream_chunk(text)

    # 继续处理直到没有更多tool_code
    while True:
        processor.reset_buffer()
        new_result, has_tool_code = processor.process_stream_chunk(result)
        if not has_tool_code:
            break
        result = new_result

    return result
