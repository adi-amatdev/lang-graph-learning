from langgraph.graph import StateGraph, START, END
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_ollama  import ChatOllama

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from dataclasses import dataclass



document_content=""

@dataclass
class AgentState:
    messages: Annotated[Sequence[BaseMessage],add_messages]



@tool
def update(content: str)-> str:
    """updates document with provided content"""
    global document_content 
    document_content = content

    return f"Successfully updated document, its contents are {document_content}"


@tool
def save(filename: str)-> str:
    """
        Save the current document as a text file and finish the process.
        Args:
            filename : name for the text file. 
    """

    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as f:
            f.write(document_content)
        print(f'\n docs has been saved to {filename}')
        return f"Document has been saved successfully to {filename}"

    except Exception as e:
        return f"Error saving document :{str(e)}"


tools = [update,save]


model = ChatOllama(model='llama3.2:1b').bind_tools(tools)


def agent_node (state: AgentState)-> AgentState:
    system_prompt = SystemMessage(

        content=f'''
            You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
            - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
            - If the user wants to save and finish, you need to use the 'save' tool.
            - Make sure to always show the current document state after modifications.
            
            The current document content is:{document_content}
        '''
    )

    if not state.messages:
        user_input = "Im ready to help you update the document. What would you like to create?"
        print(f'\n USER: {user_input}')

        user_mesg = HumanMessage(content=user_input)

    else:
        user_input = input(f'\n What would you like updated?')
        print(f'\n USER: {user_input}')
        user_mesg = HumanMessage(content=user_input)

    all_mesgs = [system_prompt] + list(state.messages) + [user_mesg]


    response = model.invoke(all_mesgs)

    print(f'\n AI: {response.content}')
    if hasattr(response,"tool_calls") and response.tool_calls:
        print(f'\n USING TOOLS: {tc["name"] for tc in response.tool_calls}')

    return {"messages": list(state.messages) + [user_mesg, response]}



def should_continue(state: AgentState)-> AgentState:
    """Determine if the conversation should continue or not"""

    mesgs = state.messages

    if not mesgs:
        return "continue"

    for message in reversed(mesgs):
        if isinstance(message, ToolMessage) and "saved" in message.content.lower() and "document" in message.content.lower():
            return "end"

    return "continue"


def print_messages(messages):
    """function to print stuff in a readable format"""

    if not messages:
        return

    for msg in messages[-3:]:
        if isinstance(msg, ToolMessage):
            print(f'\n TOOL RESULT: {msg.content}')

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools",ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_edge("agent","tools")


graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue":"agent",
        "end":END
    }
)

app = graph.compile()

def run_document_agent():
    print('\n ==============Drafter============')
    state = {"messages":[]}

    for step in app.stream(state,stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n =======Drafter finished============")

if __name__ == "__main__":
    run_document_agent()