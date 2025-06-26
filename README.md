# AutoGemini

AutoGemini 是一个用于与 Google Gemini AI 模型进行流式交互的 Python 包。它提供了完整的 API 客户端实现，支持流式响应、工具调用、图像处理等功能。

## 特性

- 🚀 **流式响应**: 支持实时流式文本生成
- 🛠️ **工具调用**: 支持函数调用和工具集成
- 🖼️ **图像处理**: 支持图像到文本的转换
- 🔍 **Google 搜索**: 可选的 Google 搜索依据功能
- 🌐 **URL 上下文**: 支持 URL 上下文工具
- ⚙️ **参数配置**: 灵活的模型参数配置
- 📝 **会话管理**: 支持聊天历史管理和上下文维护

## 安装

```bash
pip install autogemini
```

或者从源码安装：

```bash
git clone https://github.com/your-username/autogemini.git
cd autogemini
pip install -e .
```

## 快速开始

### 基本使用

```python
import asyncio
from autogemini import GeminiChat, ApiKey, ApiKeyType

async def main():
    # 初始化聊天客户端
    chat = GeminiChat()
    
    # 配置 API 密钥
    api_key = ApiKey(
        key="your-gemini-api-key-here",
        key_type=ApiKeyType.GEMINI
    )
    
    # 设置流式回调函数
    def stream_callback(chunk: str):
        print(chunk, end='', flush=True)
    
    # 发送消息并接收流式响应
    print("🤖 Assistant: ", end='')
    response = await chat.generate_response_stream(
        api_key=api_key,
        prompt="你好，请介绍一下量子计算。",
        callback=stream_callback
    )
    print(f"\n\n✅ 完整响应: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 图像处理

```python
import asyncio
from autogemini import image_to_text

async def main():
    api_key = "your-gemini-api-key-here"
    
    # 读取图像文件
    with open("example.jpg", "rb") as f:
        image_data = f.read()
    
    # 将图像转换为文本描述
    description = await image_to_text(api_key, image_data)
    print(f"图像描述: {description}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 工具调用

```python
import asyncio
from autogemini import (
    GeminiChat, ApiKey, ApiKeyType, Tool, FunctionDef,
    ChatCompletionMessage, MessageRole, Content
)

# 创建工具定义
weather_tool = Tool(
    function=FunctionDef(
        name="get_weather",
        description="获取城市天气信息",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["city"]
        }
    )
)

# 工具处理器
async def tool_handler(name: str, args: dict) -> str:
    if name == "get_weather":
        city = args.get("city", "未知城市")
        return f"{city}的天气是晴天，温度22°C"
    return "未知工具"

async def main():
    chat = GeminiChat()
    api_key = ApiKey(key="your-api-key", key_type=ApiKeyType.GEMINI)
    
    messages = [
        ChatCompletionMessage(
            role=MessageRole.USER,
            content=Content("北京今天天气怎么样？"),
            name=None,
            tool_calls=None,
            tool_call_id=None
        )
    ]
    
    response = await chat._chat_gemini_with_tools(
        api_key=api_key.key,
        messages=messages,
        tools=[weather_tool],
        tool_call_processor=tool_handler
    )
    
    print(f"响应: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 高级配置

### 模型参数配置

```python
chat = GeminiChat()

# 设置模型参数
chat.set_parameter("temperature", "0.7")
chat.set_parameter("max_tokens", "2048")
chat.set_parameter("top_p", "0.9")
chat.set_parameter("top_k", "50")

# 启用 Google 搜索
chat.set_google_search_enabled(True)

# 启用 URL 上下文工具
chat.set_url_context_enabled(True)
```

### 会话管理

```python
# 清除对话上下文
chat.clear_context()

# 设置系统提示词
chat.set_system_prompt("你是一个专业的编程助手。")

# 撤销最后一个回复
withdrawn_message = chat.withdraw_response()

# 重新生成回复
response = await chat.regenerate_response_stream(api_key, callback)
```

### 获取可用模型

```python
from autogemini import fetch_available_models

async def main():
    models = await fetch_available_models("your-api-key")
    print(f"可用模型: {models}")

if __name__ == "__main__":
    asyncio.run(main())
```

## API 参考

### GeminiChat 类

主要的聊天客户端类，提供与 Gemini API 交互的接口。

#### 方法

- `generate_response_stream(api_key, prompt, callback)`: 生成流式响应
- `regenerate_response_stream(api_key, callback)`: 重新生成最后的响应
- `set_parameter(key, value)`: 设置模型参数
- `clear_context()`: 清除聊天上下文
- `set_system_prompt(prompt)`: 设置系统提示词
- `withdraw_response()`: 撤销最后一个回复

#### 配置选项

- `temperature`: 控制输出随机性（0.0-2.0）
- `max_tokens`: 最大输出令牌数
- `top_p`: 核采样参数
- `top_k`: Top-K 采样参数
- `google_search_enabled`: 是否启用 Google 搜索
- `url_context_enabled`: 是否启用 URL 上下文工具

### 工具函数

- `image_to_text(api_key, image_data)`: 图像转文本
- `fetch_available_models(api_key)`: 获取可用模型列表

## 示例

查看 `examples/` 目录下的完整示例：

- `basic_usage.py`: 基本使用示例
- `image_example.py`: 图像处理示例
- `tool_calling_example.py`: 工具调用示例

## 开发

### 安装开发依赖

```bash
pip install -e ".[dev]"
```

### 运行测试

```bash
pytest
```

### 代码格式化

```bash
black src/
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v0.1.0

- 初始版本
- 支持流式响应
- 支持工具调用
- 支持图像处理
- 支持 Google 搜索和 URL 上下文工具