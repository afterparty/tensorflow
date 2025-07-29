import streamlit as st
from typing import Dict, Any
from datetime import datetime


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = {
            "session_start": datetime.now().isoformat(),
            "conversation_count": 0,
            "recent_context": []
        }
    
    if "agent" not in st.session_state:
        st.session_state.agent = None


def display_chat_message(message: Dict[str, Any]):
    """Display a chat message with appropriate formatting"""
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp")
    
    with st.chat_message(role):
        st.write(content)
        
        # Show timestamp if available
        if timestamp:
            st.caption(f"🕒 {datetime.fromisoformat(timestamp).strftime('%H:%M:%S')}")
        
        # Show additional info for assistant messages
        if role == "assistant":
            # Show sources if available
            sources = message.get("sources", [])
            if sources:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(sources, 1):
                        if source:
                            st.write(f"{i}. {source}")
            
            # Show tools used if available
            tools_used = message.get("tools_used", [])
            if tools_used:
                with st.expander("🔧 Tools Used"):
                    for tool in tools_used:
                        st.write(f"• {tool}")
            
            # Show error indicator if it's an error message
            if message.get("error"):
                st.error("⚠️ This message contains an error")


def format_memory_display(memory: Dict[str, Any]) -> Dict[str, Any]:
    """Format memory for display in the sidebar"""
    display_memory = {}
    
    if memory.get("conversation_count"):
        display_memory["Conversations"] = memory["conversation_count"]
    
    if memory.get("session_start"):
        start_time = datetime.fromisoformat(memory["session_start"])
        display_memory["Session Started"] = start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    if memory.get("last_updated"):
        last_time = datetime.fromisoformat(memory["last_updated"])
        display_memory["Last Updated"] = last_time.strftime("%Y-%m-%d %H:%M:%S")
    
    if memory.get("recent_context"):
        display_memory["Recent Interactions"] = len(memory["recent_context"])
    
    return display_memory


def validate_aws_config() -> Dict[str, str]:
    """Validate AWS configuration and return status"""
    import os
    
    config_status = {}
    
    # Check AWS credentials
    if os.getenv("AWS_ACCESS_KEY_ID"):
        config_status["AWS Access Key"] = "✅ Configured"
    else:
        config_status["AWS Access Key"] = "❌ Not configured"
    
    if os.getenv("AWS_SECRET_ACCESS_KEY"):
        config_status["AWS Secret Key"] = "✅ Configured"
    else:
        config_status["AWS Secret Key"] = "❌ Not configured"
    
    if os.getenv("AWS_REGION"):
        config_status["AWS Region"] = f"✅ {os.getenv('AWS_REGION')}"
    else:
        config_status["AWS Region"] = "❌ Not configured (defaulting to us-east-1)"
    
    # Check Bedrock configuration
    if os.getenv("BEDROCK_AGENT_ID"):
        config_status["Bedrock Agent ID"] = "✅ Configured"
    else:
        config_status["Bedrock Agent ID"] = "❌ Not configured"
    
    if os.getenv("BEDROCK_AGENT_ALIAS_ID"):
        config_status["Bedrock Agent Alias"] = "✅ Configured"
    else:
        config_status["Bedrock Agent Alias"] = "❌ Not configured"
    
    return config_status


def export_conversation(messages: list) -> str:
    """Export conversation history as text"""
    export_text = "# Chat Conversation Export\n\n"
    export_text += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for i, message in enumerate(messages, 1):
        role = message["role"].title()
        content = message["content"]
        timestamp = message.get("timestamp", "")
        
        export_text += f"## Message {i} - {role}\n"
        if timestamp:
            export_text += f"**Time:** {timestamp}\n\n"
        export_text += f"{content}\n\n"
        
        # Add sources if available
        if message.get("sources"):
            export_text += "**Sources:**\n"
            for source in message["sources"]:
                export_text += f"- {source}\n"
            export_text += "\n"
        
        export_text += "---\n\n"
    
    return export_text