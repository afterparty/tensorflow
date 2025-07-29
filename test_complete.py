#!/usr/bin/env python3
"""
Comprehensive test of the Agentic Chat App functionality
"""

import os
import sys
from dotenv import load_dotenv

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    try:
        from agent_core import AgenticChatAgent, search_internet
        from bedrock_config import BedrockAgentManager
        from utils import initialize_session_state, display_chat_message
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    print("\n🧪 Testing environment...")
    load_dotenv()
    
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"⚠️  Missing environment variables: {missing}")
        print("📝 App will run in fallback mode")
    else:
        print("✅ Environment variables configured")
    
    return True

def test_agent_initialization():
    """Test agent initialization"""
    print("\n🧪 Testing agent initialization...")
    try:
        from agent_core import AgenticChatAgent
        agent = AgenticChatAgent()
        print("✅ Agent initialized successfully")
        return agent
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return None

def test_basic_conversation(agent):
    """Test basic conversation functionality"""
    print("\n🧪 Testing basic conversation...")
    if not agent:
        return False
    
    try:
        response = agent.process_message("Hello, this is a test message.")
        
        print(f"✅ Agent responded: {response['content'][:100]}...")
        print(f"📊 Tools used: {response['tools_used']}")
        print(f"🧠 Memory updated: {'memory' in response}")
        
        return True
    except Exception as e:
        print(f"❌ Conversation test failed: {e}")
        return False

def test_search_functionality(agent):
    """Test web search functionality"""
    print("\n🧪 Testing web search...")
    if not agent:
        return False
    
    try:
        response = agent.process_message("Search for Python programming")
        
        has_search = 'web_search' in response.get('tools_used', [])
        has_sources = len(response.get('sources', [])) > 0
        
        print(f"✅ Search triggered: {has_search}")
        print(f"✅ Sources found: {has_sources}")
        print(f"📊 Sources: {response.get('sources', [])}")
        
        return has_search
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def test_memory_functionality(agent):
    """Test memory and context management"""
    print("\n🧪 Testing memory functionality...")
    if not agent:
        return False
    
    try:
        # First message
        response1 = agent.process_message("My name is Alice and I like cats.")
        memory1 = response1.get('memory', {})
        
        # Second message referencing context
        response2 = agent.process_message("What did I tell you about my preferences?", memory=memory1)
        memory2 = response2.get('memory', {})
        
        conversation_count = memory2.get('conversation_count', 0)
        has_context = len(memory2.get('recent_context', [])) > 0
        
        print(f"✅ Conversation count: {conversation_count}")
        print(f"✅ Context preserved: {has_context}")
        print(f"📊 Recent context entries: {len(memory2.get('recent_context', []))}")
        
        return conversation_count >= 2
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def test_bedrock_config():
    """Test Bedrock configuration utilities"""
    print("\n🧪 Testing Bedrock configuration...")
    try:
        from bedrock_config import BedrockAgentManager, create_sample_agent_config
        
        manager = BedrockAgentManager()
        config = create_sample_agent_config()
        
        print("✅ Bedrock manager created")
        print("✅ Sample config generated")
        print(f"📊 Config has {len(config)} fields")
        
        # Test connection (will likely fail in demo mode, but shouldn't crash)
        try:
            connection_result = manager.test_connection()
            print(f"📡 Connection test completed: {connection_result}")
        except Exception as e:
            print(f"⚠️  Connection test failed (expected in demo mode): {e}")
        
        return True
    except Exception as e:
        print(f"❌ Bedrock config test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Comprehensive Agentic Chat App Test")
    print("=" * 50)
    
    results = []
    
    # Test 1: Imports
    results.append(test_imports())
    
    # Test 2: Environment
    results.append(test_environment())
    
    # Test 3: Agent initialization
    agent = test_agent_initialization()
    results.append(agent is not None)
    
    # Test 4: Basic conversation
    results.append(test_basic_conversation(agent))
    
    # Test 5: Search functionality
    results.append(test_search_functionality(agent))
    
    # Test 6: Memory functionality
    results.append(test_memory_functionality(agent))
    
    # Test 7: Bedrock configuration
    results.append(test_bedrock_config())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "Imports",
        "Environment",
        "Agent initialization",
        "Basic conversation",
        "Search functionality",
        "Memory functionality",
        "Bedrock configuration"
    ]
    
    for i, (test_name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your agentic chat app is fully functional.")
        print("\n🚀 Ready to use:")
        print("   • Run command-line demo: python demo.py")
        print("   • Run web app: ./run_app.sh")
        print("   • Configure AWS Bedrock in .env for enhanced features")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)