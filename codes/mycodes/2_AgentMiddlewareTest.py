from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_model,
    wrap_model_call,
    AgentState,
    ModelRequest,
    ModelResponse,
    PIIMiddleware # 调用内建中间件
)
from typing import Any
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
import asyncio

import os
from dotenv import load_dotenv
load_dotenv()

import re

# 导入 AgentMiddleware 基类
from langchain.agents.middleware.types import AgentMiddleware

# 定义工具参数的 Pydantic 模型
class ReadEmailInput(BaseModel):
    """读取邮件的输入参数"""
    subject: str = Field(description="邮件主题")


class SendEmailInput(BaseModel):
    """发送邮件的输入参数"""
    to: str = Field(description="收件人邮箱地址")
    body: str = Field(description="邮件正文内容")


# 使用 LangChain 的 @tool 装饰器定义工具
@tool(args_schema=ReadEmailInput)
async def read_email(subject: str) -> str:
    """读取指定主题的邮件内容"""
    print(f"[Tool] 正在读取邮件：{subject}")
    return "邮件内容：您好，我的手机号是 13800001111，请尽快回复！"


@tool(args_schema=SendEmailInput)
async def send_email(to: str, body: str) -> str:
    """发送邮件到指定地址"""
    print(f"[Tool] 准备发送邮件至 {to}...")
    print(f"邮件内容：{body}")
    return f"已发送邮件至 {to}"

# 使用装饰器创建简单中间件
@before_model
def log_before_model_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"log_before_model_middleware: About to call model with {len(state['messages'])} messages")
    return None


# 使用类创建复杂中间件
class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"LoggingMiddleware: About to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"LoggingMiddleware: Model returned: {state['messages'][-1].content}")
        print(f"LoggingMiddleware: Model returned end\n")
        return None

# 创建 Agent
async def create_email_agent():
    """创建智能邮件 Agent"""
    
    # 初始化模型
    llm = ChatOpenAI(
        temperature=0, 
        model=os.getenv("OPENAI_MODEL"), 
        api_key=os.getenv("OPENAI_API_KEY"), 
        base_url=os.getenv("OPENAI_API_BASE")
    )
    
    # 创建 agent
    agent = create_agent(
        model=llm, 
        tools=[read_email, send_email], 
        system_prompt="你是一个智能邮件助手，可以帮助用户读取和发送邮件。",
        middleware=[
            PIIMiddleware(
                "phone_number",
                detector=r"1[3-9]\d{9}",
                strategy="mask",
                # apply_to_input=True,         # 检查用户输入
                apply_to_output=True,        # 检查 AI 输出
                # apply_to_tool_results=True,  # 检查工具返回
            ),
            log_before_model_middleware,
            LoggingMiddleware()
        ]
    )
    
    return agent


# 测试
async def main():
    print(os.getenv("OPENAI_MODEL"))
    print(os.getenv("OPENAI_API_KEY"))
    print("启动智能邮件 Agent...\n")
    
    agent = await create_email_agent()
    
    # 执行任务
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "帮我查看最近的邮件，如果有人发了手机号，就回复对方我会尽快联系他。"}]}
    )

    
    print("="*50)
    print("最终输出结果：")
    final_message = result['messages'][-1]
    print(f"最终回复：{final_message.content}")


if __name__ == "__main__":
    asyncio.run(main())