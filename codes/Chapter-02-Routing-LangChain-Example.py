# Copyright (c) 2025 Marco Fago
#
# This code is licensed under the MIT License.
# See the LICENSE file in the repository for the full license text.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

import os
from dotenv import load_dotenv
load_dotenv()

# Colab 代码链接: https://colab.research.google.com/drive/1Yh3eUcvajJfgTFKhEQga6bJ3yyKodAmg

# 安装依赖
# pip install langchain langgraph google-cloud-aiplatform langchain-google-genai google-adk deprecated pydantic

# --- Configuration ---
# Ensure your API key environment variable is set (e.g., GOOGLE_API_KEY)
# 确保你的 API 密钥环境变量已设置 (如 GOOGLE_API_KEY)
try:
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    # print(f"Language model initialized: {llm.model}")
    llm = ChatOpenAI(temperature=0, model=os.getenv("OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
    print(f"Language model initialized: {llm.model_name}")
except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None

# --- Define Simulated Sub-Agent Handlers (equivalent to ADK sub_agents) ---
# --- 定义模拟的子智能体处理器 (等同于 ADK 中的 sub_agents) ---

def booking_handler(request: str) -> str:
    """Simulates the Booking Agent handling a request."""
    print("--- DELEGATING TO BOOKING HANDLER ---")
    return f"Booking Handler processed request: '{request}'. Result: Simulated booking action."

def info_handler(request: str) -> str:
    """Simulates the Info Agent handling a request."""
    print("--- DELEGATING TO INFO HANDLER ---")
    return f"Info Handler processed request: '{request}'. Result: Simulated information retrieval."

def unclear_handler(request: str) -> str:
    """Handles requests that couldn't be delegated."""
    print("--- HANDLING UNCLEAR REQUEST ---")
    return f"Coordinator could not delegate request: '{request}'. Please clarify."

# --- Define Coordinator Router Chain (equivalent to ADK coordinator's instruction) ---
# This chain decides which handler to delegate to.
# --- 定义协调员的路由链 (等同于 ADK 协调员的指令) ---
# 这个链负责决定将任务委派给哪个处理器。
coordinator_router_prompt = ChatPromptTemplate.from_messages([
    ("system", """Analyze the user's request and determine which specialist handler should process it.
     - If the request is related to booking flights or hotels, output 'booker'.
     - For all other general information questions, output 'info'.
     - If the request is unclear or doesn't fit either category, output 'unclear'.
     ONLY output one word: 'booker', 'info', or 'unclear'."""),
    ("user", "{request}")
])

if llm:
    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

# --- Define the Delegation Logic (equivalent to ADK's Auto-Flow based on sub_agents) ---
# Use RunnableBranch to route based on the router chain's output.
# --- 定义委派逻辑 (等同于 ADK 基于 sub_agents 的自动流) ---
# 使用 RunnableBranch 根据路由链的输出进行路由。

# Define the branches for the RunnableBranch
# 为 RunnableBranch 定义分支
branches = {
    "booker": RunnablePassthrough.assign(output=lambda x: booking_handler(x['request']['request'])),
    "info": RunnablePassthrough.assign(output=lambda x: info_handler(x['request']['request'])),
    "unclear": RunnablePassthrough.assign(output=lambda x: unclear_handler(x['request']['request'])),
}

# Create the RunnableBranch. It takes the output of the router chain
# and routes the original input ('request') to the corresponding handler.
# 创建 RunnableBranch。它会接收路由链的输出，
# 并将原始输入 ('request') 路由到相应的处理器。
delegation_branch = RunnableBranch(
    (lambda x: x['decision'].strip() == 'booker', branches["booker"]), # Added .strip()
    (lambda x: x['decision'].strip() == 'info', branches["info"]),     # Added .strip()
    branches["unclear"] # Default branch for 'unclear' or any other output
)

# Combine the router chain and the delegation branch into a single runnable
# The router chain's output ('decision') is passed along with the original input ('request')
# to the delegation_branch.
# 将路由链和委派分支组合成一个可执行单元
# 路由链的输出 ('decision') 会连同原始输入 ('request') 一起
# 传递给 delegation_branch。
coordinator_agent = {
    "decision": coordinator_router_chain,
    "request": RunnablePassthrough()
} | delegation_branch | (lambda x: x['output']) # Extract the final output

# --- Example Usage ---
# --- 使用示例 ---
def main():
    if not llm:
        print("\nSkipping execution due to LLM initialization failure.")
        return

    print("--- Running with a booking request ---")
    request_a = "Book me a flight to London."
    print(f"Request A: {request_a}")
    result_a = coordinator_agent.invoke({"request": request_a})
    print(f"Final Result A: {result_a}")

    print("\n--- Running with an info request ---")
    request_b = "What is the capital of Italy?"
    print(f"Request B: {request_b}")
    result_b = coordinator_agent.invoke({"request": request_b})
    print(f"Final Result B: {result_b}")

    print("\n--- Running with an unclear request ---")
    request_c = "Tell me about quantum physics."
    print(f"Request C: {request_c}")
    result_c = coordinator_agent.invoke({"request": request_c})
    print(f"Final Result C: {result_c}")

if __name__ == "__main__":
    main()