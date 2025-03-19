# import streamlit as st
# from langgraph.graph import StateGraph, END
# from pydantic import BaseModel
# from typing import Dict, Any
#
# # Define a state schema using Pydantic
# class ChatState(BaseModel):
#     user_input: str = ""
#     bot_response: str = ""
#
# # Bot responses for historical monuments
# def get_monument_info(query):
#     knowledge_base = {
#         "taj mahal": "The Taj Mahal, located in Agra, India, is a UNESCO World Heritage Site and a symbol of love.",
#         "pyramids of giza": "The Pyramids of Giza in Egypt are ancient wonders built as tombs for the pharaohs.",
#         "eiffel tower": "The Eiffel Tower in Paris, France, is an iconic iron lattice tower completed in 1889."
#     }
#
#     # Normalize query for matching
#     query = query.lower().strip()
#
#     for key, value in knowledge_base.items():
#         if key in query:  # Match key within query
#             return value
#     return "I'm sorry, I only know about historical monuments. Can you ask about another one?"
#
# # Define chatbot flow using LangGraph
# graph = StateGraph(state_schema=ChatState)
#
# def start_node(state: ChatState) -> Dict[str, Any]:
#     print("DEBUG: In start_node")
#     return {"user_input": "", "bot_response": "Hey, I am a historical agent AI. You can ask me about historical monuments."}
#
# def handle_monument_query(state: ChatState) -> Dict[str, Any]:
#     query = state.user_input.strip().lower()  # Ensure input is normalized
#     print(f"DEBUG: User asked: {query}")  # Debugging output
#
#     response = get_monument_info(query)
#     print(f"DEBUG: Bot response: {response}")  # Debugging output
#
#     return {"user_input": query, "bot_response": response}
#
# # Add nodes to the graph
# graph.add_node("start", start_node)
# graph.add_node("handle_monument_query", handle_monument_query)
#
# # Define edges (Flow of conversation)
# graph.set_entry_point("start")
# graph.add_edge("start", "handle_monument_query")
# graph.add_edge("handle_monument_query", END)
#
# # Compile the graph
# graph_compiled = graph.compile()
#
# def main():
#     st.title("🗿 Historical AI Chatbot")
#     st.write("Ask me about historical monuments!")
#
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
#
#     user_input = st.text_input("You:", "", key="input")
#
#     if st.button("Send") and user_input.strip():  # Ensure input is not empty
#         # Pass input correctly
#         state = ChatState(user_input=user_input.strip())
#         response = graph_compiled.invoke(state)
#
#         # Append user and bot messages to chat history
#         st.session_state.chat_history.append(("You", user_input))
#         st.session_state.chat_history.append(("Bot", response['bot_response']))
#
#     for user, msg in st.session_state.chat_history:
#         st.write(f"**{user}:** {msg}")
#
# if __name__ == "__main__":
#     main()

from secret_api_key import groq_api_key
import streamlit as st
from typing import Annotated  # Import the Annotated class for type hints with additional metadata
from typing_extensions import TypedDict  # Import the TypedDict class for defining custom typed dictionaries
from langgraph.graph import StateGraph  # Import the StateGraph class for creating state graphs
from langgraph.graph.message import add_messages  # Import the add_messages function for adding messages to a list
from langchain_groq import ChatGroq

# Streamlit page configuration
st.set_page_config(page_title="Chatbot", page_icon="🤖")

# Title of the app
st.title("LangGraph Chatbot")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")


class State(TypedDict):
    messages: Annotated[list, add_messages]


# Create a StateGraph instance
graph_builder = StateGraph(State)


# Define the chatbot function
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# Add chatbot to the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

# Compile the graph
graph = graph_builder.compile()


# Function to stream graph updates
def stream_graph_updates(user_input: str):
    initial_state = {"messages": [("user", user_input)]}
    responses = []
    for event in graph.stream(initial_state):
        for value in event.values():
            responses.append(value["messages"][-1].content)
    return responses


# Initialize session state to store conversation history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []


# Sidebar for user input
def chatbot_sidebar():
    st.sidebar.title("Chat with the Assistant")

    # Input box for user message on the sidebar
    user_input = st.sidebar.text_input("Your message", key="input", placeholder="Type your message here...")

    # Submit button in the sidebar
    if st.sidebar.button("Send"):
        submit_message(user_input)


# Function to submit the message and generate response
def submit_message(user_input):
    if user_input:
        # Append user input to conversation history
        st.session_state['messages'].append(f"You: {user_input}")

        # Get chatbot response
        responses = stream_graph_updates(user_input)

        # Append chatbot responses to conversation history
        for response in responses:
            st.session_state['messages'].append(f"Assistant: {response}")


# Main page for displaying chat history
def display_chat():
    st.write("### Conversation:")
    for message in st.session_state['messages']:
        st.write(message)


# Run the sidebar and main chat display
chatbot_sidebar()  # Input and submit button on the sidebar
display_chat()  # Display conversation on the main page
