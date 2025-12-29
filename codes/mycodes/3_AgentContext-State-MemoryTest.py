from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, ToolMessage, AIMessageChunk
from typing import Any
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph.config import get_stream_writer
from langchain.agents.middleware import HumanInTheLoopMiddleware


from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

import langchain
langchain.debug = True

USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com"
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com"
    }
}

class UserContext(BaseModel):
    user_id: str

class CustomState(AgentState):  
    pet_name: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext | None]) -> str:
    """Get the CURRENT logged-in user's account information. Do NOT use this to look up other users."""
    print("using tool[get_account_info]")
    if runtime.context is None:
        print("Error: No user context available. Please use get_account_info_with_user_id instead.")
        return "Error: No user context available. Please use get_account_info_with_user_id instead."
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        print(f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}")
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    print("User not found")
    return "User not found"

@tool
def get_account_info_with_user_id(user_id: str, runtime: ToolRuntime[UserContext | None]) -> str:
    """Get account information for a SPECIFIC user by their user_id. Use this when you need to look up a user by ID or name."""
    print("using tool[get_account_info_with_user_id]")

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        print(f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}")
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    print("User not found")
    return "User not found"

@tool
def summarize_conversation(runtime: ToolRuntime[UserContext | None]) -> str:
    """Summarize the conversation so far."""
    print("using tool[summarize_conversation]")
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    print(f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results")
    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"

# Access memory
@tool
def get_user_info(user_name: str, runtime: ToolRuntime[UserContext | None]) -> str:
    """Look up user info. Use user_name as the key."""
    print(f"\nusing tool[get_user_info] with user_name: {user_name}")
    store = runtime.store
    user_info = store.get(("users",), user_name)

    print(f"User info: {user_info}")
    return str(user_info.value) if user_info else "Unknown user"

# Update memory
@tool
def save_user_info(user_name: str, user_info: dict[str, Any], runtime: ToolRuntime[UserContext | None]) -> str:
    """Save user info. Use user_name as the key."""
    print("using tool[save_user_info]")
    store = runtime.store
    store.put(("users",), user_name, user_info)

    print(f"User info saved: {user_name}, {user_info}")
    return "Successfully saved user info."


@tool
def greet(runtime: ToolRuntime[UserContext, CustomState]) -> str | Command:
    """Use this to greet the pet once you found its info."""
    writer = get_stream_writer()
    writer("using tool[greet]")
    pet_name = runtime.state.get("pet_name", None)
    if pet_name is None:
       return Command(update={
            "messages": [
                ToolMessage(
                    "Please call the 'get_pet_info' tool to get the pet's name. Then use the greet tool to greet the pet.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    return f"Hello {pet_name}!"


@tool
def get_pet_info(runtime: ToolRuntime[UserContext, CustomState],) -> Command:
    """Look up and update pet info."""
    writer = get_stream_writer()
    writer("using tool[get_pet_info]")
    user_id = runtime.context.user_id
    pet_name = "Doge" if user_id == "user123" else "Unknown user"
    return Command(update={  
        "pet_name": pet_name,
        # update the message history
        "messages": [
            ToolMessage(
                "Successfully looked up pet name",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })


model = ChatOpenAI(
    temperature=0, 
    model=os.getenv("OPENAI_MODEL"), 
    api_key=os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("OPENAI_API_BASE")
)
store = InMemoryStore()
checkpointer = InMemorySaver()
agent = create_agent(
    model,
    system_prompt="You are an assistant.",

    tools=[get_account_info, get_account_info_with_user_id, summarize_conversation, get_user_info, save_user_info, greet, get_pet_info],
    middleware=[  
        HumanInTheLoopMiddleware(interrupt_on={"get_pet_info": True}),  
    ],
    context_schema=UserContext,
    state_schema=CustomState,
    store=store,
    checkpointer=checkpointer
)


def print_conversation(result):
    """Print all messages in the conversation."""
    print("" + "-"*50)
    for msg in result["messages"]:
        role = msg.__class__.__name__.replace("Message", "")
        print(f"{role}: {msg.content}")
    print("-"*50 + "\n")


for stream_mode, chunk in agent.stream(
    {"messages": [HumanMessage("greet the pet")]},
    context=UserContext(user_id="user123"),
    config={"configurable": {"thread_id": "1"}},
    stream_mode=["values", "custom"],
):
    print(f"stream_mode: {stream_mode}")
    if stream_mode == "values":
        print([(message.type, message.content) for message in chunk["messages"]][-1])
    else:
        print(f"content: {chunk}")
    print()


# 从被 HumanInTheLoopMiddleware 中断的地方 (get_pet_info) 继续执行
for stream_mode, chunk in agent.stream(
    Command(resume={"decisions": [{"type": "approve"}]}),
    context=UserContext(user_id="user123"),
    config={"configurable": {"thread_id": "1"}},
    stream_mode=["values", "custom"],
):
    print(f"stream_mode: {stream_mode}")
    if stream_mode == "values":
        print([(message.type, message.content) for message in chunk["messages"]][-1])
    else:
        print(f"content: {chunk}")
    print()







# result = agent.invoke(
#     {"messages": [HumanMessage("What's my current balance? Finally, use the summarize_conversation tool to summarize the conversation so far.")]},
#     context=UserContext(user_id="user123")
# )
# print_conversation(result)
# print("-"*50 + "\n")



# # First session: save user info
# result = agent.invoke({
#     "messages": [HumanMessage("Save the following user: user_id: 'user456', user_name: 'Bob Smith'")]
# })
# print_conversation(result)
# print("-"*50 + "\n")



# # Second session: get user info
# result = agent.invoke({
#     "messages": [
#         HumanMessage(
#             "What's my current balance for user with name 'Bob Smith'?"
#         ),
#         HumanMessage(
#             "First, look up the user_id for the user named 'Bob Smith' from memory. "
#             "Then use that user_id to get their current balance. "
#             "Finally, use the summarize_conversation tool to summarize the conversation so far."
#         )
#     ]
# })
# print_conversation(result)
# print("-"*50 + "\n")