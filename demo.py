#!/usr/bin/env python3
"""
Command-line demo of the Agentic Chat App
"""

import os
import json
from dotenv import load_dotenv
from agent_core import AgenticChatAgent

def print_separator():
    print("\n" + "="*60 + "\n")

def print_response(response):
    """Pretty print agent response"""
    print(f"🤖 Agent: {response['content']}")
    
    if response.get('tools_used'):
        print(f"\n🔧 Tools used: {', '.join(response['tools_used'])}")
    
    if response.get('sources'):
        print(f"\n📚 Sources:")
        for i, source in enumerate(response['sources'], 1):
            if source:
                print(f"   {i}. {source}")
    
    if response.get('search_results'):
        print(f"\n🔍 Search results:")
        for i, result in enumerate(response['search_results'][:2], 1):
            print(f"   {i}. {result['title']}")
            print(f"      {result['snippet'][:100]}...")

def run_demo():
    """Run interactive demo"""
    load_dotenv()
    
    print("🚀 Agentic Chat App - Command Line Demo")
    print_separator()
    
    # Initialize agent
    print("Initializing agent...")
    try:
        agent = AgenticChatAgent()
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return
    
    # Demo queries
    demo_queries = [
        "Hello! What can you help me with?",
        "Search for the latest news about artificial intelligence",
        "What is LangGraph?",
        "Tell me about recent developments in renewable energy",
        "What did we discuss about AI earlier?"
    ]
    
    memory = {}
    
    print("\n🎯 Running demo queries...")
    
    for i, query in enumerate(demo_queries, 1):
        print_separator()
        print(f"Query {i}: {query}")
        print("\nProcessing...")
        
        try:
            response = agent.process_message(query, memory=memory)
            memory = response.get('memory', {})
            
            print_response(response)
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    print_separator()
    
    # Interactive mode
    print("🎮 Interactive mode - Type your questions (or 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = agent.process_message(user_input, memory=memory)
            memory = response.get('memory', {})
            
            print()
            print_response(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_demo()