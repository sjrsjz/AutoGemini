# AutoGemini

> 流式 Gemini API 处理系统 —— 自动化 AI 工具调用与响应

## 项目简介

AutoGemini 是一个基于 Python 的流式 Gemini API 处理系统，支持自动化工具代码（ToolCode）检测、执行与多轮推理，适用于需要复杂链式思考（Chain of Thought, COT）和工具调用的 AI 应用场景。

- **流式输出**：支持 AI 响应的流式处理，实时检测并执行 ToolCode。
- **自动循环推理**：内置 COT 推理流程，自动多轮调用工具并合成最终答案。
- **可扩展工具接口**：通过 `DefaultApi` 可自定义扩展工具函数。
- **语义 HTML 响应**：最终输出为结构化 HTML，便于前端解析与展示。

## 主要功能

- Gemini API 流式对话与工具调用
- ToolCode 检测与安全执行
- 多轮推理与自动循环
- 结构化 HTML 响应生成
- 易于集成与二次开发

## 安装方法

建议使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理：

```bash
uv pip install -e .
```

或使用 pip：

```bash
pip install -e .
```

## 快速开始

### 作为命令行工具运行

```bash
python -m autogemini
```

### 作为库集成

```python
from autogemini import AutoStreamProcessor

processor = AutoStreamProcessor(api_key="YOUR_GEMINI_API_KEY")
# 详细用法见 src/autogemini/auto_stream_processor.py
```

## 目录结构

- `src/autogemini/`  —— 主代码目录
- `tests/`           —— 测试用例
- `pyproject.toml`    —— 项目配置

## 测试

推荐使用 uv 运行测试：

```bash
uv run python tests/run_interactive_test.py
```

或直接运行 pytest：

```bash
pytest
```

## 贡献

欢迎提交 issue 和 PR！如需自定义工具函数，请参考 `src/autogemini/tool_code.py`。

## License

本项目采用 MIT License，详见根目录 [LICENSE](LICENSE) 文件。
