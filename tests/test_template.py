import pytest
from autogemini.template import parse_agent_output


def test_parse_agent_output_basic():
    text = """
<|start_header|>think_before_new_cycle<|end_header|>
This is the first block.
<|start_header|>call_tool_code<|end_header|>
print(default_api.some_tool())
<|start_header|>response<|end_header|>
<p>Hello, world!</p>
<|start_header|>response<|end_header|>
<p>Hello, world!</p>
"""
    blocks = parse_agent_output(text)
    assert len(blocks) == 4
    assert blocks[0].type == "think_before_new_cycle"
    assert blocks[0].content == "This is the first block."
    assert blocks[1].type == "call_tool_code"
    assert blocks[1].content == "print(default_api.some_tool())"
    assert blocks[2].type == "response"
    assert blocks[2].content == "<p>Hello, world!</p>"
    assert blocks[3].type == "response"
    assert blocks[3].content == "<p>Hello, world!</p>"
