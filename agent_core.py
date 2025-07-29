import os
import boto3
import json
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

import requests
from urllib.parse import quote


class AgentState(TypedDict):
    messages: List[BaseMessage]
    memory: Dict[str, Any]
    current_query: str
    search_results: Optional[List[Dict]]
    tools_used: List[str]
    sources: List[str]


class SearchResult(BaseModel):
    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    snippet: str = Field(description="Snippet/description of the search result")


@tool
def search_internet(query: str) -> List[SearchResult]:
    """Search the internet for information using DuckDuckGo API"""
    try:
        # Using DuckDuckGo instant answer API as a simple search
        url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        
        # Get abstract if available
        if data.get('Abstract'):
            results.append(SearchResult(
                title=data.get('AbstractText', 'Search Result'),
                url=data.get('AbstractURL', ''),
                snippet=data.get('Abstract', '')
            ))
        
        # Get related topics
        for topic in data.get('RelatedTopics', [])[:3]:
            if isinstance(topic, dict) and topic.get('Text'):
                results.append(SearchResult(
                    title=topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else 'Related Topic',
                    url=topic.get('FirstURL', ''),
                    snippet=topic.get('Text', '')
                ))
        
        # Fallback: simple web search simulation
        if not results:
            results.append(SearchResult(
                title=f"Search results for: {query}",
                url="https://duckduckgo.com",
                snippet=f"Found information related to: {query}"
            ))
        
        return results
    
    except Exception as e:
        return [SearchResult(
            title="Search Error",
            url="",
            snippet=f"Error searching for '{query}': {str(e)}"
        )]


class AgenticChatAgent:
    def __init__(self):
        self.bedrock_client = None
        self.agent_id = os.getenv("BEDROCK_AGENT_ID")
        self.agent_alias_id = os.getenv("BEDROCK_AGENT_ALIAS_ID")
        self.session_id = None
        
        # Initialize AWS Bedrock client
        try:
            self.bedrock_client = boto3.client(
                'bedrock-agent-runtime',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )
        except Exception as e:
            print(f"Warning: Could not initialize Bedrock client: {e}")
        
        # Initialize LangGraph
        self.graph = self._create_graph()
        self.app = self.graph.compile()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        def process_input(state: AgentState) -> AgentState:
            """Process the input message and update state"""
            current_query = state["current_query"]
            state["tools_used"].append("input_processing")
            return state
        
        def decide_action(state: AgentState) -> str:
            """Decide whether to search or respond directly"""
            query = state["current_query"].lower()
            
            # Keywords that suggest need for web search
            search_keywords = [
                "search", "find", "latest", "recent", "current", "news", 
                "what is", "who is", "when did", "where is", "how to",
                "weather", "stock", "price", "update"
            ]
            
            if any(keyword in query for keyword in search_keywords):
                return "search"
            else:
                return "respond"
        
        def search_web(state: AgentState) -> AgentState:
            """Search the web for information"""
            query = state["current_query"]
            search_results = search_internet.invoke(query)
            
            state["search_results"] = [
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet
                }
                for result in search_results
            ]
            state["tools_used"].append("web_search")
            state["sources"].extend([result.url for result in search_results if result.url])
            
            return state
        
        def generate_response(state: AgentState) -> AgentState:
            """Generate response using Bedrock Agent or fallback"""
            try:
                if self.bedrock_client and self.agent_id and self.agent_alias_id:
                    response = self._call_bedrock_agent(state)
                else:
                    response = self._generate_fallback_response(state)
                
                # Update memory with conversation context
                memory = state["memory"]
                memory["last_query"] = state["current_query"]
                memory["last_response"] = response
                memory["conversation_count"] = memory.get("conversation_count", 0) + 1
                memory["last_updated"] = datetime.now().isoformat()
                
                # Store recent context
                if "recent_context" not in memory:
                    memory["recent_context"] = []
                
                memory["recent_context"].append({
                    "query": state["current_query"],
                    "response": response,
                    "timestamp": datetime.now().isoformat(),
                    "search_used": bool(state.get("search_results"))
                })
                
                # Keep only last 5 interactions in memory
                if len(memory["recent_context"]) > 5:
                    memory["recent_context"] = memory["recent_context"][-5:]
                
                state["memory"] = memory
                
                # Add the response message
                ai_message = AIMessage(content=response)
                state["messages"].append(ai_message)
                
            except Exception as e:
                error_response = f"I apologize, but I encountered an error: {str(e)}"
                ai_message = AIMessage(content=error_response)
                state["messages"].append(ai_message)
            
            return state
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("process_input", process_input)
        workflow.add_node("search_web", search_web)
        workflow.add_node("generate_response", generate_response)
        
        # Add edges
        workflow.set_entry_point("process_input")
        workflow.add_conditional_edges(
            "process_input",
            decide_action,
            {
                "search": "search_web",
                "respond": "generate_response"
            }
        )
        workflow.add_edge("search_web", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow
    
    def _call_bedrock_agent(self, state: AgentState) -> str:
        """Call AWS Bedrock Agent with context"""
        try:
            # Prepare the input with context from memory and search results
            input_text = state["current_query"]
            
            # Add search context if available
            if state.get("search_results"):
                search_context = "\n".join([
                    f"Search result: {result['title']} - {result['snippet']}"
                    for result in state["search_results"][:3]
                ])
                input_text = f"Context from web search:\n{search_context}\n\nUser question: {input_text}"
            
            # Add memory context
            memory = state["memory"]
            if memory.get("recent_context"):
                recent_context = memory["recent_context"][-2:]  # Last 2 interactions
                context_text = "\n".join([
                    f"Previous: {ctx['query']} -> {ctx['response'][:100]}..."
                    for ctx in recent_context
                ])
                input_text = f"Recent conversation:\n{context_text}\n\n{input_text}"
            
            # Generate session ID if not exists
            if not self.session_id:
                self.session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            response = self.bedrock_client.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=self.session_id,
                inputText=input_text
            )
            
            # Extract response from Bedrock
            response_text = ""
            for event in response['completion']:
                if 'chunk' in event:
                    chunk = event['chunk']
                    if 'bytes' in chunk:
                        response_text += chunk['bytes'].decode('utf-8')
            
            return response_text or "I received your message but couldn't generate a response."
            
        except Exception as e:
            raise Exception(f"Bedrock Agent error: {str(e)}")
    
    def _generate_fallback_response(self, state: AgentState) -> str:
        """Generate a fallback response when Bedrock is not available"""
        query = state["current_query"]
        search_results = state.get("search_results", [])
        memory = state["memory"]
        
        # Create a contextual response
        response_parts = []
        
        # Acknowledge the query
        response_parts.append(f"I understand you're asking about: {query}")
        
        # Include search results if available
        if search_results:
            response_parts.append("\nBased on my web search, here's what I found:")
            for i, result in enumerate(search_results[:2], 1):
                response_parts.append(f"{i}. {result['title']}: {result['snippet']}")
        
        # Add memory context if relevant
        if memory.get("conversation_count", 0) > 0:
            response_parts.append(f"\nThis is our {memory['conversation_count'] + 1} interaction in this session.")
        
        # Add helpful note about capabilities
        if not search_results:
            response_parts.append("\nI can help you with questions and search the web for current information. Try asking me to search for something specific!")
        
        return "\n".join(response_parts)
    
    def process_message(self, message: str, memory: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a user message and return response with updated memory"""
        
        # Initialize state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "memory": memory or {},
            "current_query": message,
            "search_results": None,
            "tools_used": [],
            "sources": []
        }
        
        # Run the graph
        final_state = self.app.invoke(initial_state)
        
        # Extract the AI response
        ai_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
        response_content = ai_messages[-1].content if ai_messages else "I couldn't generate a response."
        
        return {
            "content": response_content,
            "memory": final_state["memory"],
            "sources": final_state["sources"],
            "tools_used": final_state["tools_used"],
            "search_results": final_state.get("search_results", [])
        }