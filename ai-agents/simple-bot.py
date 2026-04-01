from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


@dataclass
class AgentState:
    messages: list[HumanMessage] = field(default_factory=list)


llm = ChatOllama(
    model="gemma3:1b"
)

def greeting_node(state: AgentState)-> AgentState:
    
    resp = llm.invoke(state.messages)
    print(f'AI: {resp.content}')

    return state


graph = StateGraph(AgentState)

graph.add_node("greeting",greeting_node)

graph.add_edge(START,"greeting")
graph.add_edge("greeting",END)

app = graph.compile()


user_input = input('Message:')
app.invoke({"messages":[HumanMessage(content=user_input)]})
while user_input != 'exit':
    app.invoke({"messages":[HumanMessage(content=user_input)]})
    user_input = input('Message:')
 
