import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Initialize the Chat LLM. We use a powerful model like gpt-4o for better reasoning.
# A lower temperature is used for more deterministic and focused outputs.
# 使用 gpt-4o 或其他模型，并设置较低的温度值以获得更稳定的输出
llm = ChatOpenAI(temperature=0.1, model=os.getenv("OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))


def run_reflection_loop():
    """
    Demonstrates a multi-step AI reflection loop to progressively improve a Python function.
    展示了通过多步骤反思循环，逐步改进 Python 函数的方法。
    """

    # --- The Core Task ---
    # --- 核心任务的提示词 ---
    task_prompt = """
    Your task is to create a Python function named `calculate_factorial`.
    This function should do the following:
    1.  Accept a single integer `n` as input.
    2.  Calculate its factorial (n!).
    3.  Include a clear docstring explaining what the function does.
    4.  Handle edge cases: The factorial of 0 is 1.
    5.  Handle invalid input: Raise a ValueError if the input is a negative number.
    """

    # --- The Reflection Loop ---
    # --- 反思循环 ---
    max_iterations = 3
    current_code = ""
    # We will build a conversation history to provide context in each step.
    # 构建对话历史，为每一步提供必要的上下文信息。
    message_history = [HumanMessage(content=task_prompt)]

    for i in range(max_iterations):
        print("\n" + "="*25 + f" REFLECTION LOOP: ITERATION {i + 1} " + "="*25)

        # --- 1. GENERATE / REFINE STAGE ---
        # In the first iteration, it generates. In subsequent iterations, it refines.
        # 在第一次迭代时，生成初始代码；在后续迭代时，基于上一步的反馈优化代码。
        if i == 0:
            print("\n>>> STAGE 1: GENERATING initial code...")
            # The first message is just the task prompt.
            # 第一次迭代时，只需要任务提示词。
            response = llm.invoke(message_history)
            current_code = response.content
        else:
            print("\n>>> STAGE 1: REFINING code based on previous critique...")
            # The message history now contains the task, the last code, and the last critique.
            # We instruct the model to apply the critiques.
            # 后续迭代时，除了任务提示词，还包含上一步的代码和反馈。
            # 然后要求模型根据反馈意见优化代码。
            message_history.append(HumanMessage(content="Please refine the code using the critiques provided."))
            response = llm.invoke(message_history)
            current_code = response.content

        print("\n--- Generated Code (v" + str(i + 1) + ") ---\n")
        message_history.append(response) # Add the generated code to history

        # --- 2. REFLECT STAGE ---
        # --- 反思阶段 ---
        print("\n>>> STAGE 2: REFLECTING on the generated code...")

        # Create a specific prompt for the reflector agent.
        # This asks the model to act as a senior code reviewer.
        # 创建一个特定的提示词，要求模型扮演高级软件工程师的角色，对代码进行仔细的审查。
        reflector_prompt = [
            SystemMessage(content="""
                You are a senior software engineer and an expert in Python.
                Your role is to perform a meticulous code review.
                Critically evaluate the provided Python code based on the original task requirements.
                Look for bugs, style issues, missing edge cases, and areas for improvement.
                Provide a bulleted list of your critiques.
            """),
            # SystemMessage(content="""
            #     You are a senior software engineer and an expert in Python.
            #     Your role is to perform a meticulous code review.
            #     Critically evaluate the provided Python code based on the original task requirements.
            #     Look for bugs, style issues, missing edge cases, and areas for improvement.
            #     If the code is perfect and meets all requirements, respond with the single phrase 'CODE_IS_PERFECT'.
            #     Otherwise, provide a bulleted list of your critiques.
            # """),
            HumanMessage(content=f"Original Task:\n{task_prompt}\n\nCode to Review:\n{current_code}")
        ]

        critique_response = llm.invoke(reflector_prompt)
        critique = critique_response.content

        # --- 3. STOPPING CONDITION ---
        # 如果代码完美符合要求，则结束反思循环。
        if "CODE_IS_PERFECT" in critique:
            print("\n--- Critique ---\nNo further critiques found. The code is satisfactory.")
            break

        print("\n--- Critique ---\n" + critique)
        # Add the critique to the history for the next refinement loop.
        message_history.append(HumanMessage(content=f"Critique of the previous code:\n{critique}"))

    print("\n" + "="*30 + " FINAL RESULT " + "="*30)
    print("\nFinal refined code after the reflection process:\n")
    # print(current_code)


if __name__ == "__main__":
    run_reflection_loop()

