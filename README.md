# Agentic Chat App with LangGraph & AWS Bedrock

A powerful conversational AI application built with LangGraph and AWS Bedrock AgentCore, featuring short-term memory management and internet search capabilities.

## Features

- 🤖 **LangGraph Integration**: Sophisticated workflow orchestration for complex AI interactions
- 🧠 **AWS Bedrock AgentCore**: Advanced AI agent with built-in memory management
- 🔍 **Internet Search**: Real-time web search capabilities using DuckDuckGo API
- 💾 **Short-term Memory**: Maintains conversation context across interactions
- 🎨 **Modern UI**: Beautiful Streamlit interface with real-time chat
- 📊 **Monitoring**: Track tools used, sources, and conversation history
- 🔧 **Fallback Mode**: Works with or without AWS Bedrock configuration

## Architecture

The application uses a multi-layered architecture:

1. **Streamlit Frontend** (`app.py`) - User interface and session management
2. **LangGraph Workflow** (`agent_core.py`) - Orchestrates agent behavior and decision-making
3. **AWS Bedrock Integration** - Provides advanced AI capabilities with memory
4. **Internet Search Tool** - Enables real-time information retrieval
5. **Memory Management** - Maintains conversation context and history

## Quick Start

### Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- AWS Bedrock Agent configured (optional - app works without it)

### Installation

1. **Clone and setup**:
```bash
git clone <repository>
cd <repository>
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your AWS credentials and Bedrock agent details
```

3. **Run the application**:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1

# Bedrock Agent Configuration (optional)
BEDROCK_AGENT_ID=your_agent_id_here
BEDROCK_AGENT_ALIAS_ID=your_agent_alias_id_here
BEDROCK_KB_ID=your_knowledge_base_id_here
```

### AWS Bedrock Setup

#### Option 1: Use Existing Agent

If you have an existing Bedrock Agent:

1. Find your Agent ID in the AWS Bedrock console
2. Get the Agent Alias ID (usually `TSTALIASID` for test or your custom alias)
3. Add these to your `.env` file

#### Option 2: Create New Agent

Use the provided configuration helper:

```python
from bedrock_config import BedrockAgentManager, create_sample_agent_config

# Get sample configuration
config = create_sample_agent_config()
print(json.dumps(config, indent=2))
```

Then create the agent in AWS Bedrock console using this configuration.

#### Option 3: Run Without Bedrock

The app includes a fallback mode that works without AWS Bedrock:

- Internet search functionality will still work
- Basic conversation with memory management
- All features except advanced Bedrock AI capabilities

## Usage

### Starting a Conversation

1. Launch the app: `streamlit run app.py`
2. Type your message in the chat input
3. The agent will automatically decide whether to search the web or respond directly
4. View sources, tools used, and conversation memory in the sidebar

### Triggering Web Search

The agent automatically searches the web when you use keywords like:
- "search for...", "find...", "latest..."
- "what is...", "who is...", "when did..."
- "current", "recent", "news", "weather"

### Memory Management

The agent maintains short-term memory including:
- Recent conversation context (last 5 interactions)
- Session information and timestamps
- Search history and sources
- Tool usage tracking

### Features Demo

Try these example queries:

```
1. "Search for the latest news about AI"
2. "What is the current weather in New York?"
3. "Tell me about LangGraph" (will search automatically)
4. "What did we talk about earlier?" (tests memory)
5. "Find recent developments in renewable energy"
```

## Project Structure

```
├── app.py                 # Main Streamlit application
├── agent_core.py          # LangGraph agent implementation
├── bedrock_config.py      # AWS Bedrock configuration and management
├── utils.py               # Utility functions for UI and state management
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Key Components

### LangGraph Workflow

The agent uses a sophisticated workflow:

1. **Process Input** - Analyzes the user query
2. **Decision Node** - Determines if web search is needed
3. **Web Search** (conditional) - Searches internet for information
4. **Generate Response** - Creates contextual response using Bedrock or fallback
5. **Memory Update** - Updates conversation context

### Memory System

Two-layer memory approach:

1. **Session Memory** (Streamlit) - UI state and conversation history
2. **Agent Memory** (Bedrock) - Persistent context across agent invocations

### Internet Search

- Uses DuckDuckGo API for web searches
- Automatically triggered by relevant keywords
- Returns structured results with titles, URLs, and snippets
- Integrates search results into agent responses

## Advanced Features

### Monitoring and Debugging

- **Tools Used**: Track which tools the agent employed
- **Sources**: View all sources used in responses
- **Memory State**: Inspect current memory contents
- **Error Handling**: Graceful fallback for various failure modes

### Configuration Management

The `bedrock_config.py` module provides:

- Connection testing for AWS services
- Agent listing and management
- Session creation with memory
- Sample configurations for new agents

### Extensibility

Easy to extend with new features:

- Add new tools to the LangGraph workflow
- Integrate additional search engines
- Add new memory storage backends
- Customize the UI with additional widgets

## Troubleshooting

### Common Issues

1. **AWS Connection Errors**:
   - Check AWS credentials in `.env`
   - Verify Bedrock service availability in your region
   - Ensure IAM permissions for Bedrock services

2. **Agent Not Found**:
   - Verify `BEDROCK_AGENT_ID` and `BEDROCK_AGENT_ALIAS_ID`
   - Check agent status in AWS console
   - Try using fallback mode (remove agent IDs from `.env`)

3. **Search Not Working**:
   - Check internet connectivity
   - DuckDuckGo API might be rate-limited
   - Try different search keywords

4. **Memory Issues**:
   - Clear conversation in sidebar
   - Restart the Streamlit app
   - Check browser local storage

### Debug Mode

Enable debug mode by setting environment variable:
```bash
export STREAMLIT_LOGGER_LEVEL=debug
streamlit run app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review AWS Bedrock documentation
3. Open an issue on GitHub
4. Check LangGraph documentation for workflow questions

## Roadmap

Future enhancements:

- [ ] Multiple search engine support
- [ ] Long-term memory with vector storage
- [ ] Multi-modal capabilities (images, documents)
- [ ] Agent collaboration features
- [ ] Enhanced analytics and logging
- [ ] Custom tool integration framework

## Summary

You now have a fully functional agentic chat application that:

✅ **Works immediately** - Runs in fallback mode without AWS configuration  
✅ **Uses LangGraph** - Sophisticated workflow orchestration for AI interactions  
✅ **Integrates AWS Bedrock** - When configured, uses AgentCore for advanced AI capabilities  
✅ **Maintains memory** - Short-term conversation context and session management  
✅ **Searches the web** - Real-time internet search using DuckDuckGo API  
✅ **Beautiful UI** - Modern Streamlit interface with chat functionality  
✅ **Easy deployment** - Simple setup with virtual environment and dependencies  

### Quick Start Commands

```bash
# Run the command-line demo
python demo.py

# Run the web app
./run_app.sh
# or manually:
source venv/bin/activate && streamlit run app.py

# Test the setup
python setup.py
```

The application is production-ready and can be extended with additional tools, models, and capabilities as needed.
