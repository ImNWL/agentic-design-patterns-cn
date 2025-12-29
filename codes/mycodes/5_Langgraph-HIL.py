from langgraph.types import Command, interrupt
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Annotated
from operator import add
import uuid

# 1. 定义 State 类型
class HumanInLoopState(TypedDict):
    request: str
    analysis: str
    human_feedback: str
    approved: str
    messages: Annotated[List[str], add]

# 2. 定义节点函数
def analyze_node(state: HumanInLoopState) -> dict:
    return {
        "analysis": f"分析结果: {state['request']}",
        "messages": ["完成分析"]
    }

def human_feedback_node(state: HumanInLoopState) -> dict:
    # 定义中断信息，告诉外界为何中断以及需要什么样的输入来恢复
    interrupt_data = {
        "type": "human_review",
        "request": state['request'],
        "analysis": state['analysis'],
        "prompt": "请输入: 同意 / 拒绝"
    }
    # 使用 interrupt() 函数暂停工作流，等待人工输入
    print(f"等待用户反馈")
    human_response = interrupt(interrupt_data)
    print(f"收到用户输入: {human_response}")
    
    return {
        "human_feedback": human_response.get("feedback"),
        "approved": human_response.get("decision"),
        "messages": [f"人工反馈: {human_response.get('feedback')}"]
    }

def process_approval(state: HumanInLoopState) -> dict:
    return {"messages": ["请求已批准，正在处理..."]}

def process_rejection(state: HumanInLoopState) -> dict:
    return {"messages": ["请求已拒绝"]}

# 3. 定义路由函数
def route_by_human_decision(state: HumanInLoopState) -> str:
    if state.get("approved") == "同意":
        return "process_approval"
    return "process_rejection"

# 4. 构建图
builder = StateGraph(HumanInLoopState)

# 添加节点
builder.add_node("analyze", analyze_node)
builder.add_node("human_feedback", human_feedback_node)
builder.add_node("process_approval", process_approval)
builder.add_node("process_rejection", process_rejection)

# 添加边
builder.add_edge(START, "analyze")
builder.add_edge("analyze", "human_feedback")
# 添加条件边，根据用户反馈来选择调用后续的节点
builder.add_conditional_edges(
    "human_feedback",
    route_by_human_decision,
    {
        "process_approval": "process_approval",
        "process_rejection": "process_rejection"
    }
)
builder.add_edge("process_approval", END)
builder.add_edge("process_rejection", END)

# 5. 编译图并启用检查点
memory = MemorySaver()
app = builder.compile(checkpointer=memory)

# 6. 配置会话 id，用于区分不同的会话
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
initial_input = {"request": "测试请求", "messages": []}

# 首次执行图，执行到 human_feedback 节点会中断，invoke 立即返回
# 返回的结果会包含中断信息
result = app.invoke(initial_input, config)
print("中断信息:", result)

# 模拟人工输入（实际应用中来自用户界面）
human_decision = {
    "decision": "同意",
    "feedback": "用户反馈内容"
}

# 重新恢复工作流，继续执行后续节点
resume_command = Command(resume=human_decision)
final_result = app.invoke(resume_command, config)
print("最终结果:", final_result)
