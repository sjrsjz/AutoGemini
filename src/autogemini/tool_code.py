import asyncio
import ast
import re
from typing import Any, Callable, List, Dict, Tuple, Optional, Awaitable


# ==============================================================================
# 1. 异步API处理器定义 (DefaultApi)
# ==============================================================================
class DefaultApi:
    """
    一个异步API处理器，用于管理和调用由AI生成的工具函数。
    所有方法都被设计为异步的。
    """

    def __init__(self, default_handler: Callable[..., Awaitable[Any]]) -> None:
        self._handlers: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self._default_handler = default_handler

    async def __call__(self, name: str, *args, **kwargs) -> Any:
        """使得实例本身可以被调用，用于分发到具体的处理器。"""
        if name in self._handlers:
            return await self._handlers[name](*args, **kwargs)
        return await self._default_handler(name, *args, **kwargs)

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        """
        通过属性访问（如 `default_api.get_weather`）来获取一个可调用的异步方法。
        """

        async def method(*args, **kwargs):
            return await self(name, *args, **kwargs)

        return method

    def add_handler(self, name: str, handler: Callable[..., Awaitable[Any]]) -> None:
        """为特定的API方法添加一个处理器。"""
        self._handlers[name] = handler

    def remove_handler(self, name: str) -> None:
        """移除一个已有的处理器。"""
        if name in self._handlers:
            del self._handlers[name]


# ==============================================================================
# 2. 安全性组件 (AST 变换与验证)
# ==============================================================================
class AsyncApiTransformer(ast.NodeTransformer):
    """
    AST变换器，自动将 `default_api.some_method()` 调用
    转换为 `await default_api.some_method()`。
    """

    def visit_Call(self, node: ast.Call) -> Any:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "default_api"
        ):
            return ast.Await(value=node)
        return node


# 白名单，只包含绝对安全的内建函数
SAFE_BUILTINS = {
    "len",
    "str",
    "int",
    "float",
    "bool",
    "list",
    "dict",
    "tuple",
    "set",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "sum",
    "min",
    "max",
    "abs",
    "round",
    "sorted",
    "any",
    "all",
    "isinstance",
    "issubclass",
    "repr",
}


# ==============================================================================
# 3. 核心沙箱执行器 (eval_tool_code)
# ==============================================================================
async def eval_tool_code(
    tool_code: str, default_api: DefaultApi, timeout: float = 5.0
) -> List[Dict]:
    """
    在一个安全、受限、带资源限制的环境中，异步地评估工具代码。
    集成了多层安全防护。

    Args:
        tool_code: AI生成的、需要执行的代码字符串。
        default_api: 异步API处理器实例。
        timeout: 代码执行的超时秒数。

    Returns:
        由 `print` 函数捕获的执行结果列表。
    """
    results = []

    def safe_print(*args, **kwargs):
        # Limit the total print output size to prevent memory attacks
        total_size = sum(len(str(a)) for a in args)
        if total_size > 65536:  # 64KB
            raise MemoryError(
                "The output content of the tool code exceeds the 64KB limit."
            )
        results.append({"args": args, "kwargs": kwargs})

    # [安全层2：受限环境] 创建一个只包含白名单内建函数的作用域
    limited_builtins = {k: __builtins__[k] for k in SAFE_BUILTINS}
    limited_builtins["print"] = safe_print

    scope = {
        "default_api": default_api,
        "__builtins__": limited_builtins,
    }

    async def aexec_sandboxed():
        # [Security Layer 1: Static Analysis]
        try:
            tree = ast.parse(tool_code)
            _validate_ast_safety(tree)
        except Exception as e:
            raise ValueError(f"Static code analysis failed: {e}")

        # Apply AST transformation, automatically add await
        transformer = AsyncApiTransformer()
        transformed_tree = transformer.visit(tree)

        # Dynamically create and compile async function
        async_wrapper_func = ast.AsyncFunctionDef(
            name="_async_tool_exec_wrapper",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=transformed_tree.body,
            decorator_list=[],
            returns=None,
        )
        wrapper_module = ast.Module(body=[async_wrapper_func], type_ignores=[])
        ast.fix_missing_locations(wrapper_module)
        code_obj = compile(wrapper_module, "<string>", "exec")

        # Define and call the function asynchronously in the restricted scope
        exec(code_obj, scope)
        await scope["_async_tool_exec_wrapper"]()

    try:
        # [Security Layer 3: Resource Limitation] Use asyncio.wait_for for timeout control
        await asyncio.wait_for(aexec_sandboxed(), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Code execution timed out (exceeded {timeout} seconds).")
    except Exception as e:
        raise RuntimeError(f"Error occurred while executing tool code: {e}")

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
        # Check for dangerous AST node types
        for dangerous_type, message in dangerous_nodes.items():
            if isinstance(child, dangerous_type):
                raise ValueError(f"Unsafe operation detected: {message}")

        # Check for dangerous function and variable names
        if isinstance(child, ast.Name) and child.id in dangerous_names:
            raise ValueError(f"Unsafe name detected: {child.id}")

        # Check attribute access, prevent access to private or dangerous attributes
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


# ==============================================================================
# 4. 流式处理器 (ToolCodeProcessor)
# ==============================================================================
class ToolCodeProcessor:
    """
    处理流式输出，自动识别、执行和替换 `tool_code` 代码块。
    整个流程是异步的。
    """

    def __init__(self, default_api: DefaultApi):
        self.default_api = default_api
        self.buffer = ""
        self.tool_code_pattern = re.compile(r"", re.DOTALL)

    async def process_stream_chunk(self, chunk: str) -> Tuple[str, bool]:
        """
        异步地处理流式输出的一个chunk。
        在实际应用中，您会把从LLM收到的每个数据块传入这里。
        """
        self.buffer += chunk
        match = self.tool_code_pattern.search(self.buffer)

        if match:
            tool_code = match.group(1)
            start_pos, end_pos = match.start(), match.end()

            try:
                execution_result = await self._execute_tool_code(tool_code)
                replacement = self._format_execution_result(execution_result)
            except Exception as e:
                replacement = f"\n[工具执行失败: {e}]\n"

            # 替换tool_code块为执行结果
            processed_output = (
                self.buffer[:start_pos] + replacement + self.buffer[end_pos:]
            )
            self.buffer = ""  # 清空缓冲区，准备处理后续内容
            return processed_output, True

        return chunk, False

    async def _execute_tool_code(self, tool_code: str) -> List[Dict]:
        """异步执行工具代码，调用核心沙箱函数。"""
        return await eval_tool_code(tool_code, self.default_api, timeout=5.0)

    def _format_execution_result(self, results: List[Dict]) -> str:
        """格式化执行结果为字符串。"""
        if not results:
            return ""
        # 假设我们只关心最后一个print调用的第一个参数
        last_result = results[-1]
        args = last_result.get("args", ())
        return str(args) if args else ""

    def get_remaining_buffer(self) -> str:
        """获取缓冲区中尚未形成完整代码块的剩余内容。"""
        return self.buffer

    def reset_buffer(self):
        """重置缓冲区"""
        self.buffer = ""


async def process_streaming_response(
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
        processed_chunk, should_interrupt = await processor.process_stream_chunk(chunk)

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


async def extract_and_execute_all_tool_codes(text: str, default_api: DefaultApi) -> str:
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
    result, _ = await processor.process_stream_chunk(text)

    # 继续处理直到没有更多tool_code
    while True:
        processor.reset_buffer()
        new_result, has_tool_code = await processor.process_stream_chunk(result)
        if not has_tool_code:
            break
        result = new_result

    return result
