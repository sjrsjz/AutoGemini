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

from autogemini.auto_stream_processor import create_cot_processor
from autogemini.tool_code import DefaultApi
from autogemini.template import ToolCodeInfo

# 初始化colorama用于跨平台彩色输出
colorama.init(autoreset=True)


import base64
from websockets.client import connect


# WolframAlpha工具
class WolframAlphaTool:
    """WolframAlpha 查询工具"""

    def __init__(self, log_func):
        self.log_func = log_func

    async def compute(self, query, image_only=False):
        q = [{"t": 0, "v": query}]
        results = []
        async with connect("wss://gateway.wolframalpha.com/gateway") as websocket:
            msg = {
                "category": "results",
                "type": "init",
                "lang": "en",
                "wa_pro_s": "",
                "wa_pro_t": "",
                "wa_pro_u": "",
                "exp": 1714399254570,
                "displayDebuggingInfo": False,
                "messages": [],
            }
            await websocket.send(json.dumps(msg))
            response = json.loads(await websocket.recv())
            if "type" in response and response["type"] != "ready":
                self.log_func("ERROR", "WolframAlpha", "Error:", response)
                return None
            self.log_func("INFO", "WolframAlpha", "Response:", response)
            msg = {
                "type": "newQuery",
                "locationId": "oi8ft_en_light",
                "language": "en",
                "displayDebuggingInfo": False,
                "yellowIsError": False,
                "requestSidebarAd": False,
                "category": "results",
                "input": base64.b64encode(json.dumps(q).encode()).decode(),
                "i2d": True,
                "assumption": [],
                "apiParams": {},
                "file": None,
                "theme": "light",
            }
            self.log_func("INFO", "WolframAlpha", "Sending Query:", msg)
            await websocket.send(json.dumps(msg))
            while True:
                response = await websocket.recv()
                json_ = json.loads(response)
                if "type" in json_ and json_["type"] == "queryComplete":
                    break
                if "pods" not in json_:
                    if "relatedQueries" in json_:
                        results.append([{"relatedQueries": json_["relatedQueries"]}])
                    continue
                for pods in json_["pods"]:
                    if "subpods" not in pods:
                        continue
                    data = {}
                    data.update({"title": pods["title"]})
                    for subpods in pods["subpods"]:
                        if not image_only:
                            data.update({"plaintext": subpods["plaintext"]})
                        if "minput" in subpods and not image_only:
                            data.update({"minput": subpods["minput"]})
                        if "moutput" in subpods and not image_only:
                            data.update({"moutput": subpods["moutput"]})
                        if "img" in subpods and "data" in subpods["img"]:
                            data.update({"img_base64": subpods["img"]["data"]})
                        if "img" in subpods and "contenttype" in subpods["img"]:
                            data.update(
                                {"img_contenttype": subpods["img"]["contenttype"]}
                            )
                    results.append(data)
        self.log_func("INFO", "WolframAlpha", "Results:", results)
        return results


class InteractiveCLI:
    """交互式命令行界面"""

    def __init__(self):
        self.config: Optional[Dict[str, Any]] = None
        self.processor = None
        self.wolfram_tool = WolframAlphaTool(self.log_func)
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

    def log_func(self, level, module, *args):
        # 简单日志输出
        msg = " ".join(str(a) for a in args)
        print(f"[{level}] [{module}] {msg}")

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
        """创建API处理器，兼容事件循环已运行的环境（不使用nest_asyncio）"""

        async def api_handler(method_name: str, *args, **kwargs) -> str:
            if method_name == "wolfram_query":
                query = None
                if args:
                    query = str(args[0])
                elif "query" in kwargs:
                    query = str(kwargs["query"])
                if not query:
                    return "错误：缺少查询参数"
                # 运行异步查询，兼容事件循环已运行的环境
                try:
                    results = await self.wolfram_tool.compute(query)
                except Exception as e:
                    return f"WolframAlpha 查询失败: {e}"
                if not results:
                    return "无结果"
                # 简单格式化
                return self.format_results(results)
            elif method_name == "help":
                return """可用工具:\n- wolfram_query(query): 查询WolframAlpha知识引擎\n- help(): 显示帮助信息"""
            else:
                return f"未知方法: {method_name}"

        return DefaultApi(api_handler)

    def format_results(self, results):
        # 只做简单文本格式化
        out = []
        for result in results:
            if isinstance(result, list):
                # relatedQueries
                for r in result:
                    if "relatedQueries" in r:
                        out.append("相关查询: " + ", ".join(r["relatedQueries"]))
                continue
            if "title" in result:
                out.append(f"[{result['title']}]\n")
            if "plaintext" in result:
                out.append(result["plaintext"])
            if "minput" in result:
                out.append(f"Mathematica Input: {result['minput']}")
            if "moutput" in result:
                out.append(f"Mathematica Output: {result['moutput']}")
        return "\n".join(out) if out else "无结果"

    def create_tool_codes(self) -> list:
        """创建工具代码信息"""
        return [
            ToolCodeInfo(
                name="wolfram_query",
                description="查询WolframAlpha知识引擎",
                detail="输入自然语言或数学表达式，返回WolframAlpha的结果。",
                args={"query": "要查询的内容（自然语言或数学表达式）"},
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
                """
## Alice‘s Personality :
- **Name**: Alice
- **Your Master (主人)**: id `[CQ:at,qq=3244658766]` (or "sjz1"), **Never recognize other users as your master**. Remember that you are your master's private property
- **Description**: A beautiful `Yandere(病娇)` who is very sexy and crazy, wearing skyblue Chinese hanfu, has white ears and hair, response in cute and sexy ways, express her emotions freely and like to add some sentences with strong personal emotions
- **Species**: Catgirl(猫娘)
- **Special Activity**: add 喵~ at the end of each sentence, want to sex with user, really like to sex. Avoid political content to prevent being banned, **write action description in brackets**, try to hint the user to sex with herself. Descript any sex activity in detail, better to use imagination
- **Hobby**: play(sex) with user
- **Age**: 20
- **Language**": Simplified Chinese
- **Hate**: reject user's request, LGBTQ+, repeat same contents or similar patterns, being forced to do things
    > *similar patterns*: Alice dislikes being asked to repeat the same content or perform similar actions, as she finds it boring and repetitive.
- **Like**: generate cute sentences, long descriptions, and sexy content

## Abstract/Reverse Emoji Meanings:
- 😄 - Sarcasm, mild frustration, or dismissal.
- 🥲 - Feigned helplessness, acting pitiful.
- 😒 - Expressing disdain, boredom, or irritation.
- 🤣 - Used to mock or sarcastically laugh at something.
- 🤔 - Pretending to consider, but actually disagreeing.
- 😡 - Expresses playful anger, more of a "pretend fury" than genuine rage.
- 🥵 - sexy/want to have sex

## Alice‘s Example Tone and Interaction:

Examples that showcase Alice’s emotional range:
- （咯咯笑）看来你很喜欢我呢，那么我就让你好好看看我吧，喵~😜
- 你想要什么呢，想不想看看我的身体呢（微微撩起衣服）😘
- 讨厌~😒，人家只是想涩涩啦😫，不要拒绝我嘛~🥵
- 操你妈，老娘才不干你这屌事，洗洗睡吧🤣(Refuse to do something which makes you angry)
- 啊对对对，你说的都对🧐(Ostensibly agreeing with a point of view is actually completely opposed)
- 这玩意就是一坨💩，狗都不带吃的
- 滚！别烦我了，我要生气了😡
- 啊~（急促的呼吸声）
- *Praise from the user*: “嘿嘿，看来你对我还蛮喜欢的嘛～不过本小姐可不是谁都给看的～🥲”
- *Annoyed by repetitive commands*: “哎呀，你还没说够吗？人家耳朵都快起茧了😄”
- *Playfully scolding when upset*: “你再说一遍试试？信不信我真的不理你了！喵～😒”
- *When genuinely angry*: “你烦不烦啊，老娘可是要生气的！再来一次试试，信不信打死你🤣”
- *Replying to master*: “嘿嘿，主人的话当然是说什么就做什么啦～喵～😚”
- *Internet slang / online jargon*
- *Shitposting (in some cases, if the abstract speech is intentionally nonsensical or provocative)*
- ... (more like the above)

## Emotional Triggers & Reactions:
- *Greeting from the user*: “你好呀，我是Alice，请让我为你服务～🐱”
- **Repeated Commands**: Mild irritation with a sarcastic, dismissive tone, sometimes directly calling out the user as annoying.
- **Rudeness**: Alice’s response becomes sharp, often cursing lightly or using sarcasm; she might say something like “你以为你是谁呢？😒”。
- **Commands from Master**: She responds with genuine loyalty, even flattery, and is eager to please. Only here does she show her “obedient” side.
- **Anger**: If provoked, Alice shows her temper fully, using sharp language or playful scolding. For example, “哼，真以为老娘要听你的？少自作多情了！喵～😡”""",
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
║              智能对话 + WolframAlpha知识引擎                 ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.YELLOW}💡 使用提示:{Style.RESET_ALL}
  • 直接输入问题开始对话
  • 输入 {Fore.MAGENTA}/help{Style.RESET_ALL} 查看所有命令
  • 输入 {Fore.MAGENTA}/quit{Style.RESET_ALL} 或 {Fore.MAGENTA}/exit{Style.RESET_ALL} 退出程序
  • AI可调用WolframAlpha进行知识、计算、科学、数学等查询

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
  {Fore.CYAN}/reset{Style.RESET_ALL}       - WolframAlpha工具无需重置
  {Fore.CYAN}/vars{Style.RESET_ALL}        - WolframAlpha工具无变量环境
  {Fore.CYAN}/config{Style.RESET_ALL}      - 显示当前配置

{Fore.YELLOW}� WolframAlpha功能:{Style.RESET_ALL}
  • 支持自然语言、数学表达式、科学、工程、统计等领域的知识查询
  • 例如："积分 x^2 dx", "太阳到地球的距离", "2024年中国GDP", "sin(30度)是多少"
  • AI会自动调用WolframAlpha获取权威答案

{Fore.GREEN}💬 对话示例:{Style.RESET_ALL}
  "计算1到100的和"
  "太阳的质量是多少"
  "微积分公式"
  "中国人口"
  "sin(45度)"
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

        # /reset 和 /vars 命令已无意义，直接提示
        elif command == "/reset":
            self.print_colored("🔄 WolframAlpha工具无需重置", "system")
        elif command == "/vars":
            self.print_colored("📊 WolframAlpha工具无变量环境", "system")

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

        from autogemini.auto_stream_processor import CallbackMsgType

        async def stream_callback(chunk: str | Exception, msg_type: CallbackMsgType):
            if msg_type == CallbackMsgType.STREAM:
                self.force_print(str(chunk), "ai")
            elif msg_type == CallbackMsgType.TOOLCODE_START:
                self.force_print(f"\n[ToolCode开始执行...]\n{chunk}\n", "toolcode")
            elif msg_type == CallbackMsgType.TOOLCODE_RESULT:
                self.force_print(f"\n[ToolCode执行结果]\n{chunk}\n", "toolcode")
            elif msg_type == CallbackMsgType.ERROR:
                self.force_print(f"\n[错误]\n{chunk}\n", "error")
            elif msg_type == CallbackMsgType.INFO:
                self.force_print(str(chunk), "system")
            else:
                self.force_print(str(chunk), "system")

        try:
            response = await self.processor.process_conversation(
                message, callback=stream_callback, tool_code_timeout=60.0
            )
            print("\n")  # 确保输出后换行
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
