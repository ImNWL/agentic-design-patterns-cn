from typing import Annotated, List, TypedDict
import operator
from langgraph.types import Send, Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# ============================================================
# 示例 1: Send - 用于路由函数中，动态分发并行任务
# ============================================================

# 整个图的全局状态
class OverallState(TypedDict):
    input_text: str
    # 使用 operator.add 来汇总并行的结果
    results: Annotated[List[str], operator.add]

# 每个并行任务的状态（发送给翻译节点的参数）
class TranslationTask(TypedDict):
    text: str
    target_lang: str


# 1. 路由函数：决定要翻译成哪些语言，返回 Send 对象列表
def route_to_translations(state: OverallState):
    languages = ["English", "Japanese", "French"]
    # 关键点：返回一个 Send 对象列表，每个对象对应一个并行任务
    return [Send("translate_node", {"text": state["input_text"], "target_lang": lang}) 
            for lang in languages]

# 2. 翻译工作节点：被并行调用的节点
def translate_node(state: TranslationTask):
    # 这里模拟翻译逻辑
    translated = f"[{state['target_lang']}]: {state['text']}"
    # 返回结果，会自动被 OverallState 中的 operator.add 收集
    return {"results": [translated]}

builder = StateGraph(OverallState)

# 只需要添加翻译节点
builder.add_node("translate_node", translate_node)

# 从 START 使用条件边进行分发（路由函数返回 Send 列表）
builder.add_conditional_edges(START, route_to_translations)
# 翻译完后直接汇总到结束
builder.add_edge("translate_node", END)

memory = MemorySaver()
send_graph = builder.compile(checkpointer=memory)


# ============================================================
# 示例 2: Command - 用于节点内部，同时更新状态并控制流向
# ============================================================

class WorkflowState(TypedDict):
    task: str
    status: str
    result: str

# 节点 1: 处理任务，根据结果决定下一步
def process_task(state: WorkflowState):
    task = state["task"]
    
    if "urgent" in task.lower():
        # 使用 Command: 同时更新状态 + 指定跳转到 urgent_handler
        return Command(
            update={"status": "urgent_detected"},
            goto="urgent_handler"
        )
    else:
        # 普通任务走正常流程
        return Command(
            update={"status": "normal"},
            goto="normal_handler"
        )

# 节点 2a: 紧急任务处理
def urgent_handler(state: WorkflowState):
    return {"result": f"🚨 紧急处理: {state['task']}"}

# 节点 2b: 普通任务处理  
def normal_handler(state: WorkflowState):
    return {"result": f"✅ 常规处理: {state['task']}"}

builder2 = StateGraph(WorkflowState)
builder2.add_node("process_task", process_task)
builder2.add_node("urgent_handler", urgent_handler)
builder2.add_node("normal_handler", normal_handler)

builder2.add_edge(START, "process_task")
# 注意：使用 Command 时不需要 add_conditional_edges，流向由节点内部决定
builder2.add_edge("urgent_handler", END)
builder2.add_edge("normal_handler", END)

command_graph = builder2.compile()


# ============================================================
# 测试运行
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("示例 1: Send - 并行翻译")
    print("=" * 50)

    send_config = {"configurable": {"thread_id": "send_thread"}}
    result1 = send_graph.invoke({"input_text": "你好世界", "results": []}, config=send_config)
    print("翻译结果:")
    for r in result1["results"]:
        print(f"  {r}")

    # 查看所有的步骤，注意，checkpoints的第一个值是Graph执行的最后一个节点（顺序是反的）
    checkpoints = list(send_graph.get_state_history(send_config))
    for i, checkpoint in enumerate(checkpoints):
        print(f"步骤 {i}: 下一节点 {checkpoint.next}, 状态值 {checkpoint.values}")

    # 获取一个检查点，更新状态
    checkpoint = checkpoints[2]
    # update_state 返回新的 config
    new_config = send_graph.update_state(
        checkpoint.config,
        {
            "input_text": "世界你好", 
            "results": []
        }
    )
    # 从更新后的检查点继续执行，使用 new_config
    result1 = send_graph.invoke(None, config=new_config)
    print("\n从更新后的检查点恢复执行:")
    for r in result1["results"]:
        print(f"  {r}")
    

    
    print("\n" + "=" * 50)
    print("示例 2: Command - 条件路由")
    print("=" * 50)
    
    # 测试普通任务
    result2 = command_graph.invoke({"task": "写周报", "status": "", "result": ""})
    print(f"普通任务: {result2['result']}")
    
    # 测试紧急任务
    result3 = command_graph.invoke({"task": "Urgent: 服务器宕机", "status": "", "result": ""})
    print(f"紧急任务: {result3['result']}")


# ============================================================
# Send vs Command 对比总结
# ============================================================
# 特性              | Send                    | Command
# ----------------- | ----------------------- | -----------------------
# 使用位置          | 路由函数中（条件边）    | 节点内部
# 主要用途          | 动态分发并行任务        | 更新状态 + 控制流向
# 能否更新状态      | ❌ 只能传递参数         | ✅ 通过 update 参数
# 能否指定下一节点  | ✅ 第一个参数           | ✅ 通过 goto 参数
# 能否并行          | ✅ 返回列表即可         | ✅ goto 可以是列表