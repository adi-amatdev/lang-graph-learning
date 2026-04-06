# Creating tools and logging in tool-messages
# Creating a reAct graph



from dataclasses import dataclass

from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import SystemMessage

from langchain_ollama import ChatOllama

# reduces function like add_messages allows us to append new info into the state
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode



@dataclass
class AgentState:
    messages: Annotated[Sequence[BaseMessage],add_messages]


@tool
def add(a: int, b:int)-> int:
    """This is an addition function that adds 2 numbers. input params: a: int, b:int, returns : int"""

    return a + b

@tool
def sub(a: int, b:int)-> int:
    """This is an subtraction function that subtracts 2 numbers. input params: a: int, b:int, returns : int"""

    return a - b


@tool
def mul(a:int,b:int)-> int:
    """This is an multiplication function that multiplies 2 numbers. input params: a: int, b:int, returns : int"""
    return a*b


tools = [add,sub,mul]

model = ChatOllama(model='llama3.2:1b').bind_tools(tools)



def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are a genuis ai assitant, help me with my query"
    )
    response = model.invoke([system_prompt] + list(state.messages))

    return {"messages": [response]}


def should_continue(state: AgentState)-> AgentState:
    messages = state.messages
    last_msg = messages[-1]
    if not last_msg.tool_calls:
        return "end"
    else:
        return "continue"


 
graph = StateGraph(AgentState)

graph.add_node("agent_node",model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tool_node",tool_node)


graph.add_edge(START, "agent_node")
graph.add_conditional_edges(
    "agent_node",
    should_continue,
    {
        "end":END,
        "continue": "tool_node"
    }
)
graph.add_edge("tool_node","agent_node")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages":[("user","3 and 4(multiply)")]}
print_stream(app.stream(inputs, stream_mode="values"))

