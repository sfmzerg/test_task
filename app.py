from datetime import datetime, timezone
from typing import Sequence
from langgraph.graph import END, MessageGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool


llm = ChatOllama(model="mistral")


@tool
def get_current_time() -> dict:
    """Return the current UTC time in ISO-8601 format.

    Returns:
        dict: A dictionary with the current UTC time in ISO-8601 format.
        Example: {"utc": "2025-05-21T06:42:00Z"}
    """
    now = datetime.now(timezone.utc)
    return {"utc": now.strftime("%Y-%m-%dT%H:%M:%SZ")}



tools = [get_current_time]
tool_node = ToolNode(tools)


workflow = MessageGraph()
workflow.add_node("agent", lambda messages: llm.invoke(messages))
workflow.add_node("action", tool_node)
workflow.set_entry_point("agent")



def route_messages(state: Sequence[HumanMessage | AIMessage]):
    if not state:
        return END

    last_message = state[-1]
    if isinstance(last_message, HumanMessage):
        content = last_message.content.lower()
        if any(keyword in content for keyword in ["time", "what time", "current time", "clock"]):
            return "action"
    return END


workflow.add_conditional_edges("agent", route_messages)
workflow.add_edge("action", "agent")


app = workflow.compile()



def run_chat(message: str):
    try:
        response = app.invoke([HumanMessage(content=message)])
        if isinstance(response[-1], AIMessage):
            return response[-1].content
        return "I couldn't generate a response."
    except Exception as e:
        return f"An error occurred: {str(e)}"