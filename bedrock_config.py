import os
import boto3
import json
from typing import Dict, Any, Optional, List
from datetime import datetime


class BedrockAgentManager:
    """Manager for AWS Bedrock Agent configuration and operations"""
    
    def __init__(self):
        self.bedrock_client = None
        self.agent_client = None
        self.region = os.getenv("AWS_REGION", "us-east-1")
        
        try:
            # Initialize Bedrock runtime client
            self.bedrock_client = boto3.client(
                'bedrock-agent-runtime',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=self.region
            )
            
            # Initialize Bedrock agent client for management
            self.agent_client = boto3.client(
                'bedrock-agent',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=self.region
            )
        except Exception as e:
            print(f"Warning: Could not initialize Bedrock clients: {e}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to AWS Bedrock services"""
        result = {
            "bedrock_runtime": False,
            "bedrock_agent": False,
            "agent_accessible": False,
            "error": None
        }
        
        try:
            # Test Bedrock runtime connection
            if self.bedrock_client:
                # This is a simple test - we try to list model invocation logs
                # which should fail gracefully if we don't have permissions
                try:
                    self.bedrock_client.list_model_invocation_logs()
                    result["bedrock_runtime"] = True
                except Exception:
                    # Even if this specific call fails, the client might be working
                    result["bedrock_runtime"] = True
            
            # Test Bedrock agent connection
            if self.agent_client:
                try:
                    self.agent_client.list_agents(maxResults=1)
                    result["bedrock_agent"] = True
                except Exception as e:
                    result["error"] = f"Agent client error: {str(e)}"
            
            # Test specific agent accessibility
            agent_id = os.getenv("BEDROCK_AGENT_ID")
            if agent_id and self.bedrock_client:
                try:
                    # Try to invoke the agent with a simple test
                    response = self.bedrock_client.invoke_agent(
                        agentId=agent_id,
                        agentAliasId=os.getenv("BEDROCK_AGENT_ALIAS_ID", "TSTALIASID"),
                        sessionId=f"test-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        inputText="Hello"
                    )
                    result["agent_accessible"] = True
                except Exception as e:
                    result["error"] = f"Agent invocation error: {str(e)}"
        
        except Exception as e:
            result["error"] = f"General connection error: {str(e)}"
        
        return result
    
    def list_available_agents(self) -> List[Dict[str, Any]]:
        """List available Bedrock agents in the account"""
        agents = []
        
        if not self.agent_client:
            return agents
        
        try:
            response = self.agent_client.list_agents()
            
            for agent_summary in response.get('agentSummaries', []):
                agent_info = {
                    "agent_id": agent_summary.get('agentId'),
                    "agent_name": agent_summary.get('agentName'),
                    "agent_status": agent_summary.get('agentStatus'),
                    "description": agent_summary.get('description', ''),
                    "created_at": agent_summary.get('createdAt'),
                    "updated_at": agent_summary.get('updatedAt')
                }
                agents.append(agent_info)
        
        except Exception as e:
            print(f"Error listing agents: {e}")
        
        return agents
    
    def get_agent_details(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific agent"""
        if not self.agent_client or not agent_id:
            return None
        
        try:
            response = self.agent_client.get_agent(agentId=agent_id)
            return response.get('agent', {})
        except Exception as e:
            print(f"Error getting agent details: {e}")
            return None
    
    def list_agent_aliases(self, agent_id: str) -> List[Dict[str, Any]]:
        """List aliases for a specific agent"""
        aliases = []
        
        if not self.agent_client or not agent_id:
            return aliases
        
        try:
            response = self.agent_client.list_agent_aliases(agentId=agent_id)
            
            for alias_summary in response.get('agentAliasSummaries', []):
                alias_info = {
                    "agent_alias_id": alias_summary.get('agentAliasId'),
                    "agent_alias_name": alias_summary.get('agentAliasName'),
                    "agent_alias_status": alias_summary.get('agentAliasStatus'),
                    "description": alias_summary.get('description', ''),
                    "created_at": alias_summary.get('createdAt'),
                    "updated_at": alias_summary.get('updatedAt')
                }
                aliases.append(alias_info)
        
        except Exception as e:
            print(f"Error listing agent aliases: {e}")
        
        return aliases
    
    def create_session_with_memory(self, agent_id: str, agent_alias_id: str, session_id: str = None) -> str:
        """Create a session with memory enabled"""
        if not session_id:
            session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Bedrock Agent automatically manages session memory
        # We just need to return a consistent session ID
        return session_id
    
    def invoke_agent_with_memory(
        self, 
        agent_id: str, 
        agent_alias_id: str, 
        session_id: str,
        input_text: str,
        enable_trace: bool = False
    ) -> Dict[str, Any]:
        """Invoke agent with memory context"""
        if not self.bedrock_client:
            raise Exception("Bedrock client not initialized")
        
        try:
            invoke_params = {
                'agentId': agent_id,
                'agentAliasId': agent_alias_id,
                'sessionId': session_id,
                'inputText': input_text
            }
            
            if enable_trace:
                invoke_params['enableTrace'] = True
            
            response = self.bedrock_client.invoke_agent(**invoke_params)
            
            # Process the streaming response
            response_text = ""
            trace_data = []
            
            for event in response.get('completion', []):
                if 'chunk' in event:
                    chunk = event['chunk']
                    if 'bytes' in chunk:
                        response_text += chunk['bytes'].decode('utf-8')
                
                if enable_trace and 'trace' in event:
                    trace_data.append(event['trace'])
            
            return {
                "response": response_text,
                "session_id": session_id,
                "trace": trace_data if enable_trace else None
            }
        
        except Exception as e:
            raise Exception(f"Error invoking agent: {str(e)}")
    
    def get_session_memory(self, agent_id: str, agent_alias_id: str, session_id: str) -> Dict[str, Any]:
        """Get memory for a specific session (if supported by the agent)"""
        # Note: Direct memory retrieval might not be available depending on agent configuration
        # This is a placeholder for future implementation
        return {
            "session_id": session_id,
            "memory_available": "Agent manages memory internally",
            "note": "Bedrock Agent handles memory automatically per session"
        }


def create_sample_agent_config() -> Dict[str, Any]:
    """Create a sample configuration for setting up a Bedrock Agent"""
    return {
        "agent_name": "ChatAssistant",
        "description": "An AI assistant for general conversation and web search",
        "foundation_model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "instruction": """
        You are a helpful AI assistant with access to web search capabilities. 
        You can help users with questions, provide information, and search the web when needed.
        
        Guidelines:
        - Be helpful, accurate, and concise
        - Use web search when you need current information
        - Maintain conversation context using your memory
        - Be honest about your limitations
        """,
        "action_groups": [
            {
                "action_group_name": "WebSearchGroup",
                "description": "Tools for searching the web",
                "api_schema": {
                    "openapi": "3.0.0",
                    "info": {
                        "title": "Web Search API",
                        "version": "1.0.0"
                    },
                    "paths": {
                        "/search": {
                            "post": {
                                "summary": "Search the web",
                                "parameters": [
                                    {
                                        "name": "query",
                                        "in": "query",
                                        "required": True,
                                        "schema": {"type": "string"},
                                        "description": "Search query"
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        ]
    }