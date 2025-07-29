#!/usr/bin/env python3
"""
Setup and configuration test script for the Agentic Chat App
"""

import os
import sys
import subprocess
from typing import Dict, Any
from dotenv import load_dotenv

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_environment_file():
    """Check if .env file exists and has required variables"""
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        print("📝 Creating .env from .env.example...")
        try:
            with open('.env.example', 'r') as example, open('.env', 'w') as env_file:
                env_file.write(example.read())
            print("✅ .env file created. Please edit it with your credentials.")
            return False
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    
    load_dotenv()
    
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION']
    optional_vars = ['BEDROCK_AGENT_ID', 'BEDROCK_AGENT_ALIAS_ID']
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_required:
        print(f"❌ Missing required environment variables: {', '.join(missing_required)}")
        return False
    
    print("✅ Environment file configured")
    if missing_optional:
        print(f"⚠️  Optional variables not set (app will use fallback mode): {', '.join(missing_optional)}")
    
    return True

def test_aws_connection():
    """Test AWS connection and Bedrock access"""
    print("\n🔗 Testing AWS connection...")
    
    try:
        from bedrock_config import BedrockAgentManager
        
        manager = BedrockAgentManager()
        result = manager.test_connection()
        
        if result['bedrock_runtime']:
            print("✅ Bedrock Runtime connection successful")
        else:
            print("❌ Bedrock Runtime connection failed")
        
        if result['bedrock_agent']:
            print("✅ Bedrock Agent connection successful")
        else:
            print("❌ Bedrock Agent connection failed")
        
        if result['agent_accessible']:
            print("✅ Configured agent is accessible")
        elif os.getenv('BEDROCK_AGENT_ID'):
            print("❌ Configured agent is not accessible")
        else:
            print("ℹ️  No agent configured (fallback mode will be used)")
        
        if result['error']:
            print(f"⚠️  Error details: {result['error']}")
        
        return result['bedrock_runtime'] or not os.getenv('BEDROCK_AGENT_ID')
    
    except Exception as e:
        print(f"❌ Error testing AWS connection: {e}")
        return False

def test_internet_search():
    """Test internet search functionality"""
    print("\n🔍 Testing internet search...")
    
    try:
        from agent_core import search_internet
        
        results = search_internet("test search")
        if results and len(results) > 0:
            print("✅ Internet search working")
            print(f"   Found {len(results)} results")
            return True
        else:
            print("❌ Internet search not working")
            return False
    
    except Exception as e:
        print(f"❌ Error testing internet search: {e}")
        return False

def test_agent_creation():
    """Test agent creation and basic functionality"""
    print("\n🤖 Testing agent creation...")
    
    try:
        from agent_core import AgenticChatAgent
        
        agent = AgenticChatAgent()
        
        # Test basic message processing
        response = agent.process_message("Hello, test message")
        
        if response and response.get('content'):
            print("✅ Agent creation and basic functionality working")
            print(f"   Response: {response['content'][:100]}...")
            return True
        else:
            print("❌ Agent not responding properly")
            return False
    
    except Exception as e:
        print(f"❌ Error testing agent: {e}")
        return False

def list_available_agents():
    """List available Bedrock agents if configured"""
    if not os.getenv('AWS_ACCESS_KEY_ID'):
        return
    
    print("\n📋 Listing available Bedrock agents...")
    
    try:
        from bedrock_config import BedrockAgentManager
        
        manager = BedrockAgentManager()
        agents = manager.list_available_agents()
        
        if agents:
            print(f"✅ Found {len(agents)} agents:")
            for agent in agents:
                print(f"   - {agent['agent_name']} ({agent['agent_id']}) - {agent['agent_status']}")
        else:
            print("ℹ️  No agents found or unable to list agents")
    
    except Exception as e:
        print(f"❌ Error listing agents: {e}")

def run_setup():
    """Run the complete setup process"""
    print("🚀 Agentic Chat App Setup\n")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check environment configuration
    if not check_environment_file():
        print("\n⚠️  Please edit the .env file with your AWS credentials and run setup again.")
        return False
    
    # Test AWS connection
    aws_ok = test_aws_connection()
    
    # Test internet search
    search_ok = test_internet_search()
    
    # Test agent creation
    agent_ok = test_agent_creation()
    
    # List available agents
    list_available_agents()
    
    print("\n" + "="*50)
    
    if aws_ok and search_ok and agent_ok:
        print("✅ Setup completed successfully!")
        print("\n🎉 You can now run the app with: streamlit run app.py")
        return True
    else:
        print("⚠️  Setup completed with some issues.")
        if not aws_ok:
            print("   - AWS/Bedrock connection issues (app will work in fallback mode)")
        if not search_ok:
            print("   - Internet search issues")
        if not agent_ok:
            print("   - Agent creation issues")
        print("\n🔧 You can still run the app, but some features might not work properly.")
        print("   Run: streamlit run app.py")
        return False

if __name__ == "__main__":
    success = run_setup()
    sys.exit(0 if success else 1)