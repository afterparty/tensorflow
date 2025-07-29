import streamlit as st
import os
from dotenv import load_dotenv
from typing import Dict, Any, List
import json
from datetime import datetime

from agent_core import AgenticChatAgent
from utils import initialize_session_state, display_chat_message

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="Agentic Chat App",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Agentic Chat with LangGraph & AWS Bedrock")
    st.sidebar.title("Configuration")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.subheader("Agent Settings")
        
        # Display current configuration
        agent_id = os.getenv("BEDROCK_AGENT_ID", "Not configured")
        agent_alias = os.getenv("BEDROCK_AGENT_ALIAS_ID", "Not configured")
        region = os.getenv("AWS_REGION", "us-east-1")
        
        st.info(f"**Agent ID:** {agent_id}")
        st.info(f"**Agent Alias:** {agent_alias}")
        st.info(f"**Region:** {region}")
        
        # Clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.agent_memory = {}
            st.rerun()
        
        # Show memory state
        if st.session_state.agent_memory:
            st.subheader("Agent Memory")
            with st.expander("View Memory State"):
                st.json(st.session_state.agent_memory)
    
    # Initialize agent
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = AgenticChatAgent()
            st.success("Agent initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(user_message)
        display_chat_message(user_message)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.process_message(
                        prompt,
                        memory=st.session_state.agent_memory
                    )
                    
                    # Update memory
                    st.session_state.agent_memory = response.get("memory", {})
                    
                    # Display response
                    assistant_message = {
                        "role": "assistant",
                        "content": response["content"],
                        "timestamp": datetime.now().isoformat(),
                        "sources": response.get("sources", []),
                        "tools_used": response.get("tools_used", [])
                    }
                    
                    st.write(response["content"])
                    
                    # Show sources if available
                    if response.get("sources"):
                        with st.expander("Sources"):
                            for source in response["sources"]:
                                st.write(f"- {source}")
                    
                    # Show tools used if available
                    if response.get("tools_used"):
                        with st.expander("Tools Used"):
                            for tool in response["tools_used"]:
                                st.write(f"- {tool}")
                    
                    st.session_state.messages.append(assistant_message)
                    
                except Exception as e:
                    error_message = f"Error processing message: {str(e)}"
                    st.error(error_message)
                    
                    error_response = {
                        "role": "assistant",
                        "content": error_message,
                        "timestamp": datetime.now().isoformat(),
                        "error": True
                    }
                    st.session_state.messages.append(error_response)

if __name__ == "__main__":
    main()