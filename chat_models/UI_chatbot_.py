import streamlit as st
from dotenv import load_dotenv
import os
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load env
load_dotenv()

# Initialize model
model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.9,
    max_tokens=150
)

# Page config
st.set_page_config(page_title="AI Mood Chatbot", page_icon="🤖", layout="centered")

# Title
st.title("🤖 AI Mood Chatbot")
st.markdown("### Choose your AI personality & start chatting")

# Sidebar for mode selection
st.sidebar.title("⚙️ Settings")
mode_option = st.sidebar.selectbox(
    "Choose AI Mood",
    ["😡 Angry", "😢 Sad", "😄 Funny"]
)

# Map modes
if mode_option == "😡 Angry":
    mode = "You are an angry AI agent. Reply aggressively and sarcastically."
elif mode_option == "😢 Sad":
    mode = "You are a sad AI agent. Reply in a depressed and emotional tone."
else:
    mode = "You are a funny AI agent. Reply humorously and make jokes."

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=mode)]

# Display chat history
for msg in st.session_state.messages[1:]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Chat input
prompt = st.chat_input("Type your message...")

if prompt:
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # Get response
    response = model.invoke(st.session_state.messages)

    # Add AI response
    st.session_state.messages.append(AIMessage(content=response.content))
    st.chat_message("assistant").write(response.content)

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = [SystemMessage(content=mode)]
    st.rerun()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit + LangChain + Mistral AI")