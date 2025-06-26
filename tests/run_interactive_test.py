#!/usr/bin/env python3
"""
AutoGemini 交互式CLI客户端
提供命令行界面与AI进行对话，支持ToolCode执行
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import colorama
from colorama import Fore, Back, Style

# 添加src路径到Python路径
current_dir = Path(__file__).parent
src_path = current_dir.parent / "src"
sys.path.insert(0, str(src_path))

from autogemini.auto import create_cot_processor
from autogemini.tool_code import DefaultApi
from autogemini.template import ToolCodeInfo

# 初始化colorama用于跨平台彩色输出
colorama.init(autoreset=True)


class PythonEvalTool:
    """简易的Python eval工具，提供安全的代码执行环境"""

    def __init__(self):
        # 定义允许导入的安全模块白名单
        self.safe_modules = {
            "math",  # 数学计算模块
            "random",  # 随机数生成模块
            "datetime",  # 日期时间模块
            "json",  # JSON处理模块
            "re",  # 正则表达式模块
            "statistics",  # 统计模块
            "decimal",  # 精确十进制计算
            "fractions",  # 分数计算
        }

        # 预定义的安全执行环境
        self.safe_globals = {
            "__builtins__": {
                # 数学函数
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                # 类型转换
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                # 数学函数
                "pow": pow,
                # 输出函数
                "print": print,
                # 自定义安全的__import__函数
                "__import__": self._safe_import,
            },
        }
        self.local_vars = {}

    def _safe_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """安全的模块导入函数，只允许导入白名单中的模块"""
        if name in self.safe_modules:
            # 使用真实的__import__函数导入允许的模块
            import builtins

            return builtins.__import__(name, globals, locals, fromlist, level)
        else:
            raise ImportError(
                f"出于安全考虑，模块 '{name}' 不允许导入。允许的模块: {', '.join(sorted(self.safe_modules))}"
            )

    def execute(self, code: str) -> str:
        """执行Python代码并返回结果"""
        try:
            code = code.strip()
            if not code:
                return "空代码"

            # 使用io.StringIO捕获print输出
            import io
            import sys

            # 保存原始stdout
            old_stdout = sys.stdout
            captured_output = io.StringIO()

            try:
                # 重定向stdout到我们的缓冲区
                sys.stdout = captured_output

                # 直接使用exec执行代码
                exec(code, self.safe_globals, self.local_vars)

                # 获取捕获的输出
                output = captured_output.getvalue()

                # 如果有输出内容，返回输出；否则返回执行成功
                return output.strip() if output.strip() else "执行成功"

            finally:
                # 恢复原始stdout
                sys.stdout = old_stdout
                captured_output.close()

        except ZeroDivisionError:
            return "错误：除零"
        except NameError as e:
            return f"错误：未定义的变量或函数 - {str(e)}"
        except ValueError as e:
            return f"错误：值错误 - {str(e)}"
        except TypeError as e:
            return f"错误：类型错误 - {str(e)}"
        except Exception as e:
            return f"错误：{type(e).__name__} - {str(e)}"

    def reset_vars(self):
        """重置所有变量"""
        self.local_vars.clear()
        return "变量已重置"

    def get_vars(self) -> str:
        """获取当前所有变量"""
        if not self.local_vars:
            return "无变量"

        vars_list = []
        for key, value in self.local_vars.items():
            if not key.startswith("_"):
                vars_list.append(f"{key} = {value}")

        return "\n".join(vars_list) if vars_list else "无用户定义变量"


class InteractiveCLI:
    """交互式命令行界面"""

    def __init__(self):
        self.config: Optional[Dict[str, Any]] = None
        self.processor = None
        self.python_tool = PythonEvalTool()
        self.running = True

        # 颜色配置
        self.colors = {
            "user": Fore.CYAN,
            "ai": Fore.GREEN,
            "system": Fore.YELLOW,
            "error": Fore.RED,
            "toolcode": Fore.BLUE,
            "command": Fore.MAGENTA,
        }

    def load_config(self) -> bool:
        """加载配置文件"""
        config_path = Path(__file__).parent / "keys.json"
        example_path = Path(__file__).parent / "keys.json.example"

        if not config_path.exists():
            self.print_colored(f"❌ 配置文件不存在: {config_path}", "error")
            self.print_colored(
                f"请复制 {example_path} 到 {config_path} 并填入你的API密钥", "system"
            )
            return False

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

            # 验证必要的配置项
            if not self.config or not self.config.get("api_key"):
                self.print_colored("❌ 请在keys.json中设置有效的API密钥", "error")
                return False

            return True
        except json.JSONDecodeError as e:
            self.print_colored(f"❌ 配置文件格式错误: {e}", "error")
            return False
        except Exception as e:
            self.print_colored(f"❌ 加载配置文件失败: {e}", "error")
            return False

    def create_api_handler(self) -> DefaultApi:
        """创建API处理器"""

        def api_handler(method_name: str, *args, **kwargs) -> str:
            if method_name == "python_eval":
                # 支持位置参数和关键字参数
                code = None
                if args:
                    code = str(args[0])
                elif "code" in kwargs:
                    code = str(kwargs["code"])

                if code:
                    return self.python_tool.execute(code)
                return "错误：缺少代码参数"
            elif method_name == "python_reset":
                return self.python_tool.reset_vars()
            elif method_name == "python_vars":
                return self.python_tool.get_vars()
            elif method_name == "help":
                return """可用工具:
- python_eval(code): 执行Python代码
- python_reset(): 重置变量环境  
- python_vars(): 查看当前变量
- help(): 显示帮助信息"""
            else:
                return f"未知方法: {method_name}"

        return DefaultApi(api_handler)

    def create_tool_codes(self) -> list:
        """创建工具代码信息"""
        return [
            ToolCodeInfo(
                name="python_eval",
                description="执行Python代码并返回结果",
                detail="支持基础数学运算、变量赋值等操作。你将会获得代码执行完成后stdout的内容",
                args={
                    "code": "要执行的Python代码字符串，为了获取结果，请使用print()函数"
                },
            ),
            ToolCodeInfo(
                name="python_reset",
                description="重置Python执行环境",
                detail="清空所有用户定义的变量",
                args={},
            ),
            ToolCodeInfo(
                name="python_vars",
                description="查看当前Python环境中的变量",
                detail="显示所有用户定义的变量及其值",
                args={},
            ),
            ToolCodeInfo(
                name="help",
                description="显示可用工具的帮助信息",
                detail="列出所有可用的工具函数及其说明",
                args={},
            ),
        ]

    def initialize_processor(self):
        """初始化处理器"""
        if not self.config:
            raise ValueError("配置未加载")

        api = self.create_api_handler()
        tool_codes = self.create_tool_codes()

        self.processor = create_cot_processor(
            api_key=self.config["api_key"],
            default_api=api,
            tool_codes=tool_codes,
            character_description=self.config.get(
                "character_description",
                "你是一个智能助手，能够执行Python代码并提供准确的回答。",
            ),
            model=self.config.get("model", "gemini-2.5-flash"),
            temperature=self.config.get("temperature", 1.0),
            max_tokens=self.config.get("max_tokens", 8192),
            top_p=self.config.get("top_p", 0.95),
            top_k=self.config.get("top_k", 40),
            timeout=self.config.get("timeout", 300.0),
        )

    def print_colored(self, text: str, color: str = "system"):
        """打印彩色文本"""
        print(f"{self.colors.get(color, '')}{text}{Style.RESET_ALL}", flush=True)

    def force_print(self, text: str, color: str = "system", end: str = ""):
        """强制输出文本，确保立即显示"""
        colored_text = f"{self.colors.get(color, '')}{text}{Style.RESET_ALL}"
        print(colored_text, end=end, flush=True)

    def print_banner(self):
        """打印欢迎横幅"""
        banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                    🤖 AutoGemini CLI                         ║
║              智能对话 + Python代码执行                       ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.YELLOW}💡 使用提示:{Style.RESET_ALL}
  • 直接输入问题开始对话
  • 输入 {Fore.MAGENTA}/help{Style.RESET_ALL} 查看所有命令
  • 输入 {Fore.MAGENTA}/quit{Style.RESET_ALL} 或 {Fore.MAGENTA}/exit{Style.RESET_ALL} 退出程序
  • AI可以执行Python代码进行计算和分析

{Fore.GREEN}🚀 准备就绪！开始对话吧...{Style.RESET_ALL}
"""
        print(banner)

    def print_help(self):
        """打印帮助信息"""
        help_text = f"""
{Fore.MAGENTA}📋 可用命令:{Style.RESET_ALL}
  {Fore.CYAN}/help{Style.RESET_ALL}        - 显示此帮助信息
  {Fore.CYAN}/quit, /exit{Style.RESET_ALL} - 退出程序
  {Fore.CYAN}/clear{Style.RESET_ALL}       - 清空对话历史
  {Fore.CYAN}/reset{Style.RESET_ALL}       - 重置Python执行环境
  {Fore.CYAN}/vars{Style.RESET_ALL}        - 查看当前Python变量
  {Fore.CYAN}/config{Style.RESET_ALL}      - 显示当前配置

{Fore.YELLOW}💻 Python功能:{Style.RESET_ALL}
  • 支持基础数学运算: 1+1, (2+3)*4
  • 支持变量操作: x=10, y=x*2
  • 支持数学函数: math.sin(math.pi/2)
  • 自动执行AI生成的Python代码

{Fore.GREEN}💬 对话示例:{Style.RESET_ALL}
  "计算1到100的和"
  "定义一个变量x=5，然后计算x的平方"
  "帮我分析这个数据集的统计信息"
"""
        print(help_text)

    def handle_command(self, command: str) -> bool:
        """处理用户命令，返回True表示继续，False表示退出"""
        command = command.lower().strip()

        if command in ["/quit", "/exit"]:
            self.print_colored("👋 再见！", "system")
            return False

        elif command == "/help":
            self.print_help()

        elif command == "/clear":
            if self.processor:
                self.processor.clear_history()
                self.print_colored("🧹 对话历史已清空", "system")
            else:
                self.print_colored("❌ 处理器未初始化", "error")

        elif command == "/reset":
            result = self.python_tool.reset_vars()
            self.print_colored(f"🔄 {result}", "system")

        elif command == "/vars":
            vars_info = self.python_tool.get_vars()
            self.print_colored(f"📊 当前变量:\n{vars_info}", "system")

        elif command == "/config":
            if self.config:
                config_info = f"""当前配置:
模型: {self.config.get('model', 'N/A')}
温度: {self.config.get('temperature', 'N/A')}
最大Token: {self.config.get('max_tokens', 'N/A')}"""
                self.print_colored(config_info, "system")
            else:
                self.print_colored("❌ 配置未加载", "error")

        else:
            self.print_colored(f"❓ 未知命令: {command}，输入 /help 查看帮助", "system")

        return True

    async def process_user_message(self, message: str):
        """处理用户消息"""
        if not self.processor:
            self.print_colored("❌ 处理器未初始化", "error")
            return

        from autogemini.auto import CallbackMsgType

        def stream_callback(chunk: str, msg_type: CallbackMsgType):
            if msg_type == CallbackMsgType.STREAM:
                self.force_print(chunk, "ai")
            elif msg_type == CallbackMsgType.TOOLCODE_START:
                self.force_print(f"\n[ToolCode开始执行...]\n{chunk}\n", "toolcode")
            elif msg_type == CallbackMsgType.TOOLCODE_RESULT:
                self.force_print(f"\n[ToolCode执行结果]\n{chunk}\n", "toolcode")
            elif msg_type == CallbackMsgType.ERROR:
                self.force_print(f"\n[错误]\n{chunk}\n", "error")
            elif msg_type == CallbackMsgType.INFO:
                self.force_print(chunk, "system")
            else:
                self.force_print(chunk, "system")

        try:
            response = await self.processor.process_conversation(
                message, callback=stream_callback
            )

            # 流式输出完成后，显示最终完整回答
            self.print_colored("─" * 50, "system")
            self.print_colored("📋 最终回答:", "system")
            self.print_colored(response, "ai")
            self.print_colored("─" * 50, "system")

        except Exception as e:
            self.print_colored(f"\n❌ 处理失败: {e}", "error")

    async def start_interactive_session(self):
        """启动交互式会话"""
        self.print_banner()

        while self.running:
            try:
                # 获取用户输入
                user_input = input(
                    f"\n{self.colors['user']}👤 您:{Style.RESET_ALL} "
                ).strip()

                if not user_input:
                    continue

                # 处理命令
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue

                # 处理普通消息
                print(
                    f"\n{self.colors['ai']}🤖 AI:{Style.RESET_ALL} ", end="", flush=True
                )
                await self.process_user_message(user_input)

            except KeyboardInterrupt:
                self.print_colored("\n\n👋 收到中断信号，程序退出", "system")
                break
            except EOFError:
                self.print_colored("\n\n👋 程序退出", "system")
                break
            except Exception as e:
                self.print_colored(f"\n❌ 发生错误: {e}", "error")

    async def run(self):
        """运行CLI程序"""
        # 加载配置
        if not self.load_config():
            return

        # 初始化处理器
        try:
            self.initialize_processor()
        except Exception as e:
            self.print_colored(f"❌ 初始化失败: {e}", "error")
            return

        # 启动交互会话
        await self.start_interactive_session()


async def main():
    """主函数"""
    cli = InteractiveCLI()
    await cli.run()


if __name__ == "__main__":
    # 检查colorama依赖
    try:
        import colorama
    except ImportError:
        print("❌ 缺少colorama依赖，请运行: pip install colorama")
        sys.exit(1)

    # 运行CLI
    asyncio.run(main())
