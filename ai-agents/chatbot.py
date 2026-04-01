from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama



@dataclass
class AgentState:
    messages: list[HumanMessage | AIMessage] = field(default_factory=list)


llm = ChatOllama(
    model="gemma3:1b"
)

def greeting_node(state: AgentState)-> AgentState:
    
    resp = llm.invoke(state.messages)
    state.messages.append(AIMessage(content=resp.content))
    print(f'AI: {resp.content}')

    return state


graph = StateGraph(AgentState)

graph.add_node("greeting",greeting_node)

graph.add_edge(START,"greeting")
graph.add_edge("greeting",END)

app = graph.compile()


conversation_history = []
user_input = input('Message:')
while user_input != 'exit':
    conversation_history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input('Message:')
 

with open('convo.txt','w') as file:
    file.write(f"\n Convo history: \n")

    for mesg in conversation_history:
        if isinstance(mesg, HumanMessage):
            file.write(f'\n you: {mesg.content} \n')

        elif isinstance(mesg,AIMessage) :
            file.write(f'\n AI: {mesg.content} \n')

    file.write('end of convo')


print("app closed")
