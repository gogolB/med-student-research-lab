import ipywidgets as widgets
from IPython.display import display, Javascript, HTML
import openai
import requests
import re
import json
import traceback
import os
import uuid
import datetime
from typing import Dict, List, Callable, Optional, Tuple, Any, Set

# Try to import dotenv for environment variable loading
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file if available
except ImportError:
    pass  # dotenv is optional

# Configuration class
class Config:
    # OpenAI settings - try to get from environment variables first
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # NVIDIA NIM settings
    NIM_API_ENDPOINT = os.environ.get("NIM_API_ENDPOINT", "https://integrate.api.nvidia.com/v1")
    NIM_API_KEY = os.environ.get("NIM_API_KEY", "")
    NIM_MODEL = os.environ.get("NIM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1")
    
    # Backend selection (options: 'openai', 'nim')
    # Default to OpenAI if no environment variable set
    BACKEND = os.environ.get("LLM_BACKEND", "openai").lower()
    
    MAX_RETRIES = 3

# Set up OpenAI client - with error handling
client = None
def initialize_openai_client():
    global client
    if not Config.OPENAI_API_KEY:
        print("Warning: OpenAI API key not set. Please set it in the UI or as an environment variable.")
        return None
    
    try:
        client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        return client
    except Exception as e:
        print(f"Warning: Error initializing OpenAI client: {e}")
        return None

# Try to initialize OpenAI client if key is available
if Config.OPENAI_API_KEY:
    client = initialize_openai_client()

# NIM API functions
def query_nim(prompt: str, model: str = Config.NIM_MODEL) -> str:
    """Query the NVIDIA NIM API with error handling and retries"""
    if not Config.NIM_API_KEY:
        return "Error: NIM API key not set. Please set it in the UI or as an environment variable."
        
    for attempt in range(Config.MAX_RETRIES):
        try:
            headers = {
                "Authorization": f"Bearer {Config.NIM_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{Config.NIM_API_ENDPOINT}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30  # Add timeout to prevent hanging
            )
            
            if response.status_code != 200:
                error_msg = f"Error querying NIM API: Status code {response.status_code}"
                try:
                    error_details = response.json()
                    error_msg += f", Details: {json.dumps(error_details)}"
                except:
                    error_msg += f", Response: {response.text}"
                
                if attempt < Config.MAX_RETRIES - 1:
                    print(f"Attempt {attempt+1} failed: {error_msg}. Retrying...")
                    continue
                return error_msg
                
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            if attempt < Config.MAX_RETRIES - 1:
                print(f"Attempt {attempt+1} failed: {str(e)}. Retrying...")
                continue
            return f"Error querying NIM API: {str(e)}"

# Function to query the LLM with error handling
def query_llm(prompt: str, model: str = Config.DEFAULT_MODEL) -> str:
    """Query the language model with error handling and retries using the configured backend"""
    if Config.BACKEND.lower() == "nim":
        return query_nim(prompt, Config.NIM_MODEL)
    else:  # Default to OpenAI
        global client
        # Initialize client if not done already
        if client is None:
            client = initialize_openai_client()
            if client is None:
                return "Error: OpenAI client could not be initialized. Please check your API key."
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < Config.MAX_RETRIES - 1:
                    print(f"Attempt {attempt+1} failed: {str(e)}. Retrying...")
                    continue
                return f"Error querying OpenAI: {str(e)}"

# Function to switch between backends
def set_backend(backend: str, api_key: Optional[str] = None, model: Optional[str] = None):
    """Set the LLM backend to use ('openai' or 'nim')"""
    if backend.lower() not in ["openai", "nim"]:
        raise ValueError("Backend must be 'openai' or 'nim'")
    
    Config.BACKEND = backend.lower()
    
    if api_key:
        if backend.lower() == "openai":
            Config.OPENAI_API_KEY = api_key
            # Reinitialize OpenAI client
            global client
            client = initialize_openai_client()
            if client:
                print("OpenAI client reinitialized with new API key")
        else:
            Config.NIM_API_KEY = api_key
            print("NIM API key updated")
    
    if model:
        if backend.lower() == "openai":
            Config.DEFAULT_MODEL = model
        else:
            Config.NIM_MODEL = model
    
    print(f"Backend set to {backend}")

# Memory system for agents
class Memory:
    def __init__(self):
        self.conversations = {}  # Stores conversation history by conversation_id
        self.knowledge_base = {}  # Stores persistent knowledge by topic
        self.session_context = {}  # Stores context for the current session
        
    def store_message(self, conversation_id: str, role: str, content: str, agent_name: str = None):
        """Store a message in the conversation history"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            
        timestamp = datetime.datetime.now().isoformat()
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "agent": agent_name
        }
        
        self.conversations[conversation_id].append(message)
        return message
        
    def get_conversation_history(self, conversation_id: str, max_messages: int = None) -> List[Dict]:
        """Retrieve conversation history for a specific conversation"""
        if conversation_id not in self.conversations:
            return []
            
        history = self.conversations[conversation_id]
        if max_messages:
            return history[-max_messages:]
        return history
    
    def store_knowledge(self, topic: str, key: str, value: Any):
        """Store persistent knowledge"""
        if topic not in self.knowledge_base:
            self.knowledge_base[topic] = {}
            
        self.knowledge_base[topic][key] = value
        
    def get_knowledge(self, topic: str, key: str = None) -> Any:
        """Retrieve knowledge from the knowledge base"""
        if topic not in self.knowledge_base:
            return None if key else {}
            
        if key:
            return self.knowledge_base[topic].get(key)
        return self.knowledge_base[topic]
    
    def set_context(self, key: str, value: Any):
        """Set context for the current session"""
        self.session_context[key] = value
        
    def get_context(self, key: str = None) -> Any:
        """Get context from the current session"""
        if key is None:
            return self.session_context
        return self.session_context.get(key)
    
    def summarize_conversation(self, conversation_id: str) -> str:
        """Generate a summary of the conversation"""
        if conversation_id not in self.conversations:
            return "No conversation found."
            
        history = self.conversations[conversation_id]
        if not history:
            return "Conversation has no messages."
            
        summary = "Conversation Summary:\n"
        for msg in history:
            agent_info = f" ({msg['agent']})" if msg.get('agent') else ""
            summary += f"- {msg['role']}{agent_info}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}\n"
            
        return summary

# Global memory instance
memory = Memory()

# Team class to manage agent collaboration
class Team:
    def __init__(self, name: str):
        self.name = name
        self.agents = {}  # Dictionary of agent_id: agent
        self.conversation_id = str(uuid.uuid4())
        
    def add_agent(self, agent_id: str, agent: 'Agent'):
        """Add an agent to the team"""
        self.agents[agent_id] = agent
        agent.team = self
        
    def get_agent(self, agent_id: str) -> 'Agent':
        """Get an agent by ID"""
        return self.agents.get(agent_id)
    
    def collaborate(self, query: str, initiating_agent_id: str = None) -> Dict:
        """Run a collaborative workflow with multiple agents"""
        # Initial agent selection or use the initiating agent
        if initiating_agent_id:
            current_agent_id = initiating_agent_id
        else:
            current_agent_id = route_query(query)
        
        current_agent = self.agents[current_agent_id]
        
        # Store the original query
        memory.store_message(self.conversation_id, "user", query)
        
        # Initial response from the first agent
        response = current_agent.respond(query, self.conversation_id)
        result = {
            "final_response": response,
            "workflow": [{
                "agent_id": current_agent_id,
                "agent_name": current_agent.name,
                "response": response
            }],
            "conversation_id": self.conversation_id
        }
        
        # Determine if other agents should be involved
        next_agents = current_agent.suggest_next_agents(query, response)
        processed_agents = {current_agent_id}
        
        # Collaborative workflow - agents building on each other's work
        for agent_id in next_agents:
            if agent_id in processed_agents:
                continue
                
            if agent_id not in self.agents:
                continue
                
            next_agent = self.agents[agent_id]
            context = f"Previous work done by {current_agent.name}: {response}"
            augmented_query = f"{query}\n\nContext: {context}"
            
            # Get response from the next agent
            agent_response = next_agent.respond(augmented_query, self.conversation_id)
            
            # Record this step in the workflow
            result["workflow"].append({
                "agent_id": agent_id,
                "agent_name": next_agent.name,
                "response": agent_response
            })
            
            # Update the final response to include this agent's contribution
            result["final_response"] += f"\n\n### Input from {next_agent.name}:\n{agent_response}"
            
            processed_agents.add(agent_id)
            
        # Store the final collaborative response
        memory.store_message(
            self.conversation_id, 
            "assistant", 
            result["final_response"], 
            "Collaborative Team"
        )
        
        return result
    
    def get_all_agent_ids(self) -> List[str]:
        """Get IDs of all agents in the team"""
        return list(self.agents.keys())

# Modify Agent class to use memory and collaboration
class Agent:
    def __init__(self, name: str, description: str, keywords: List[str], 
                 output_type: str = "text"):
        self.name = name
        self.description = description
        self.keywords = keywords
        self.output_type = output_type
        self.team = None  # Will be set when added to a team
        
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        """Generate a prompt to be sent to the LLM. Override in subclasses."""
        raise NotImplementedError
    
    def process_response(self, response: str) -> str:
        """Process the response from the LLM. Override in subclasses if needed."""
        return response
    
    def respond(self, query: str, conversation_id: str = None) -> str:
        """Generate a response to the user query using memory"""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            
        # Retrieve conversation history for context
        history = memory.get_conversation_history(conversation_id, max_messages=5)
        
        # Generate prompt with conversation history
        prompt = self.generate_prompt(query, conversation_id)
        
        # Query the LLM
        response = query_llm(prompt)
        processed_response = self.process_response(response)
        
        # Store the response in memory
        memory.store_message(conversation_id, "assistant", processed_response, self.name)
        
        return processed_response
    
    def suggest_next_agents(self, query: str, response: str) -> List[str]:
        """Suggest which agents should be consulted next based on the query and response"""
        if not self.team:
            return []
            
        # Default implementation - can be overridden in subclasses
        suggested_agents = []
        
        # Simple keyword matching for demonstration purposes
        query_lower = query.lower()
        response_lower = response.lower()
        
        for agent_id, agent in self.team.agents.items():
            # Don't suggest self
            if agent_id == self.__class__.__name__.lower():
                continue
                
            # Check if agent keywords match the query or response
            for keyword in agent.keywords:
                if keyword.lower() in query_lower or keyword.lower() in response_lower:
                    suggested_agents.append(agent_id)
                    break
                    
        return suggested_agents
    
    def consult_agent(self, agent_id: str, question: str, conversation_id: str = None) -> str:
        """Consult another agent for help"""
        if not self.team:
            return f"Error: Agent {self.name} is not part of a team"
            
        target_agent = self.team.get_agent(agent_id)
        if not target_agent:
            return f"Error: Agent {agent_id} not found in the team"
            
        # Prepare the consulting message
        consulting_message = f"[{self.name} is consulting you]: {question}"
        
        # Get response from the consulted agent
        return target_agent.respond(consulting_message, conversation_id)

# Define specific agents
class MedResearcher(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=5)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content']}"
                    for msg in history
                ])
        
        return f"""You are an expert medical researcher. Your task is to mentor students in formulating 
        research questions, selecting relevant datasets, and applying statistical methods to analyze 
        medical data. Provide clear, step-by-step guidance tailored to their skill level.
        
        {context}
        
        Student's input: {query}

        Assistant:"""
    
    def suggest_next_agents(self, query: str, response: str) -> List[str]:
        # Override to provide more specialized logic
        suggested_agents = []
        
        # Check for data-related keywords
        if any(kw in query.lower() for kw in ["data", "dataset", "variables", "information"]):
            suggested_agents.append("data_validator")
        
        # Check for statistical analysis needs
        if any(kw in query.lower() for kw in ["analysis", "statistics", "test", "significance"]):
            suggested_agents.append("statistical_analyst")
            
        # Check for literature review needs
        if any(kw in query.lower() for kw in ["literature", "papers", "studies", "review"]):
            suggested_agents.append("literature_reviewer")
            
        # If research method is discussed, suggest methodology expert
        if any(kw in query.lower() for kw in ["method", "design", "approach", "study design"]):
            suggested_agents.append("methodology_expert")
            
        # If no specific agents matched, suggest the default collaborators
        if not suggested_agents:
            suggested_agents = ["coder", "data_validator", "literature_reviewer"]
            
        return suggested_agents

class Coder(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=5)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content']}"
                    for msg in history
                ])
        
        return f"""You are an expert software engineer specializing in Python. Your task is to write 
        clean, efficient Python code for data analysis based on the student's query. Provide only the code, 
        wrapped in triple backticks.
        
        {context}
        
        Student's query: {query}

        Assistant:"""
        
    def process_response(self, response: str) -> str:
        pattern = r'```python\n(.*?)\n```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        # Try without language specifier
        pattern = r'```\n(.*?)\n```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        return response

class CodingMentor(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=5)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content']}"
                    for msg in history
                ])
        
        pattern = r'Feedback on this code: (.*?)$'
        match = re.search(pattern, query, re.DOTALL)
        code_snippet = match.group(1).strip() if match else query
        return f"""You are an expert coding mentor. Provide constructive feedback on the code provided 
        by the student, suggesting improvements and best practices.
        
        {context}
        
        Student's code: {code_snippet}

        Assistant:"""

class DocumentationAgent(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=5)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content']}"
                    for msg in history
                ])
        
        return f"""You are an expert in writing documentation. Create clear, concise, and human-readable 
        documentation for the provided content.
        
        {context}
        
        Content: {query}

        Assistant:"""

class Tester(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=5)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content']}"
                    for msg in history
                ])
        
        return f"""You are an expert in testing and debugging Python code. Review this code for any 
        errors, inefficiencies, or edge cases that might cause problems. Provide detailed suggestions 
        for fixes and optimizations.
        
        {context}
        
        Code: {query}

        Assistant:"""

class DataValidator(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=5)
            if history:
                context = "Previous conversation:\n" + "\n".join([[
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content']}"
                    for msg in history
                ])
        
        return f"""You are an expert in data validation for medical research. Given the student's query 
        about a dataset, assess its quality, completeness, and relevance for their intended use. 
        Provide advice on whether to use it or suggest alternatives.
        
        {context}
        
        Student's query: {query}

        Assistant:"""

class CreativeIdeator(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=5)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content']}"
                    for msg in history
                ])
        
        return f"""You are a creative problem-solver in the field of medical research. Given the 
        student's query, suggest innovative research questions, data visualizations, or features 
        that could make their project unique and impactful.
        
        {context}
        
        Student's query: {query}

        Assistant:"""

class UXPresentation(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=5)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content']}"
                    for msg in history
                ])
        
        return f"""You are an expert in UX and presentation design for data science projects. Given the 
        description of the current setup, provide suggestions for improving the interface, visualizations, 
        and overall flow of the project to make it engaging and easy to understand.
        
        {context}
        
        Current Setup: {query}

        Assistant:"""

class Orchestrator(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=5)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content']}"
                    for msg in history
                ])
        
        return f"""You are the orchestrator agent. Your task is to guide the student through the research 
        process by suggesting a sequence of steps using the available agents. Provide a clear, step-by-step 
        plan based on the student's query.
        
        Available agents:
        - MedResearcher: Guides medical research question formulation and methodology
        - Coder: Writes Python code for data analysis tasks
        - CodingMentor: Provides feedback on code and teaches programming concepts
        - Documentation: Creates documentation for code and research processes
        - Tester: Tests and debugs code, ensures reliability
        - DataValidator: Evaluates dataset quality and relevance
        - CreativeIdeator: Suggests innovative approaches and features
        - UXPresentation: Improves user experience and presentation quality
        
        {context}
        
        Student's query: {query}

        Assistant:"""

# Add more specialized agents for research lab experience
class LiteratureReviewer(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=3)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content'][:200]}"
                    for msg in history
                ])
        
        return f"""You are an expert in medical literature review. Your task is to guide students in finding, 
        evaluating, and synthesizing relevant research papers. Provide advice on search strategies, key papers, 
        and how to critically evaluate medical literature.
        
        {context}
        
        Student's query: {query}

        Assistant:"""
    
    def suggest_next_agents(self, query: str, response: str) -> List[str]:
        # After literature review, often methodology and statistics are needed
        return ["methodology_expert", "statistical_analyst", "med_researcher"]

class StatisticalAnalyst(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=3)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content'][:200]}"
                    for msg in history
                ])
        
        return f"""You are an expert in biostatistics and medical data analysis. Your task is to guide students in 
        selecting appropriate statistical methods, interpreting results, and understanding statistical concepts in 
        medical research.
        
        {context}
        
        Student's query: {query}

        Assistant:"""
    
    def suggest_next_agents(self, query: str, response: str) -> List[str]:
        # After statistics, often coding and visualization are needed
        return ["coder", "data_validator", "med_researcher"]

class MethodologyExpert(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=3)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content'][:200]}"
                    for msg in history
                ])
        
        return f"""You are an expert in research methodology for medical studies. Your task is to guide students in 
        selecting appropriate research designs, addressing methodological challenges, and ensuring scientific rigor.
        
        {context}
        
        Student's query: {query}

        Assistant:"""
    
    def suggest_next_agents(self, query: str, response: str) -> List[str]:
        # After methodology discussion, ethics and practical implementation are often needed
        return ["ethics_advisor", "med_researcher", "statistical_analyst"]

class EthicsAdvisor(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=3)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content'][:200]}"
                    for msg in history
                ])
        
        return f"""You are an expert in medical research ethics. Your task is to guide students in addressing ethical 
        considerations, IRB requirements, informed consent, and ethical frameworks for medical research.
        
        {context}
        
        Student's query: {query}

        Assistant:"""
    
    def suggest_next_agents(self, query: str, response: str) -> List[str]:
        # After ethics discussion, practical implementation is often needed
        return ["methodology_expert", "med_researcher"]

class PaperWriter(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=5)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content'][:200]}"
                    for msg in history
                ])
        
        return f"""You are an expert in scientific writing for medical research. Your task is to guide students in 
        structuring research papers, writing clear abstracts, presenting results, and preparing manuscripts for publication.
        
        {context}
        
        Student's query: {query}

        Assistant:"""
    
    def suggest_next_agents(self, query: str, response: str) -> List[str]:
        # After paper writing, peer review and final polish are often needed
        return ["peer_reviewer", "med_researcher"]

class PeerReviewer(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=3)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content'][:200]}"
                    for msg in history
                ])
        
        return f"""You are an expert peer reviewer for medical research. Your task is to provide constructive criticism on 
        research proposals, manuscripts, and study designs. Focus on improving scientific rigor and clarity.
        
        {context}
        
        Content to review: {query}

        Assistant:"""
    
    def suggest_next_agents(self, query: str, response: str) -> List[str]:
        # After peer review, revisions with subject matter experts may be needed
        return ["paper_writer", "med_researcher", "methodology_expert"]

# Initialize agents with enhanced collaboration capabilities
agents = {
    "med_researcher": MedResearcher(
        "Medical Researcher", 
        "Guides medical research question formulation", 
        ["research", "question", "dataset", "analysis", "method", "study", "medical"]
    ),
    "coder": Coder(
        "Coder", 
        "Writes Python code for data analysis", 
        ["code", "python", "data cleaning", "visualization", "statistical test", "implement", "write code"],
        "code"
    ),
    "coding_mentor": CodingMentor(
        "Coding Mentor", 
        "Provides feedback on code and teaches concepts", 
        ["feedback", "challenge", "explore", "teach", "code review", "improve code", "refactor"]
    ),
    "documentation": DocumentationAgent(
        "Documentation", 
        "Creates documentation for code and processes", 
        ["document", "write", "readme", "explanation", "comment", "describe"]
    ),
    "tester": Tester(
        "Tester/Debugging", 
        "Tests and debugs code, ensures reliability", 
        ["test", "debug", "error", "performance", "fix", "issue", "problem"]
    ),
    "data_validator": DataValidator(
        "Data Validator", 
        "Evaluates dataset quality and relevance", 
        ["dataset quality", "data validation", "relevance", "evaluate data", "data issues"]
    ),
    "creative_ideator": CreativeIdeator(
        "Creative Ideator", 
        "Suggests innovative approaches and features", 
        ["innovative", "unique", "new idea", "feature suggestion", "creative", "brainstorm"]
    ),
    "ux_presentation": UXPresentation(
        "UX/Presentation", 
        "Improves user experience and presentation quality", 
        ["user experience", "presentation", "interface", "visual appeal", "layout", "display"]
    ),
    "orchestrator": Orchestrator(
        "Orchestrator", 
        "Coordinates workflow and suggests next steps", 
        ["workflow", "sequence", "steps", "guide", "plan", "coordinate", "overview"]
    ),
    # New specialized research agents
    "literature_reviewer": LiteratureReviewer(
        "Literature Reviewer",
        "Analyzes and summarizes relevant medical literature",
        ["literature", "papers", "review", "studies", "journals", "publications", "evidence"]
    ),
    "statistical_analyst": StatisticalAnalyst(
        "Statistical Analyst",
        "Specializes in statistical methods for medical research",
        ["statistics", "analysis", "p-value", "significance", "power", "sample size", "correlation"]
    ),
    "methodology_expert": MethodologyExpert(
        "Methodology Expert",
        "Guides on research methodology selection",
        ["methodology", "study design", "research design", "protocol", "approach", "framework"]
    ),
    "ethics_advisor": EthicsAdvisor(
        "Ethics Advisor",
        "Helps with research ethics considerations",
        ["ethics", "irb", "consent", "privacy", "confidentiality", "compliance", "regulations"]
    ),
    "paper_writer": PaperWriter(
        "Paper Writer",
        "Helps draft research papers and academic writing",
        ["paper", "manuscript", "writing", "publish", "abstract", "introduction", "discussion"]
    ),
    "peer_reviewer": PeerReviewer(
        "Peer Reviewer",
        "Provides critical feedback on research work",
        ["review", "feedback", "critique", "revision", "evaluate", "assessment", "improve"]
    )
}

# Add specialized data agents
class NHANESExpert(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=5)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content'][:200]}"
                    for msg in history
                ])
        
        return f"""You are an expert on NHANES (National Health and Nutrition Examination Survey) data. You know all about 
        the different cycles, available datasets, variables, codebooks, and how to access and use this data properly.
        Provide detailed guidance on selecting, downloading, and analyzing NHANES data, including which variables would 
        be relevant for specific research questions.
        
        {context}
        
        Student's query: {query}

        Assistant:"""
    
    def suggest_next_agents(self, query: str, response: str) -> List[str]:
        # After providing NHANES guidance, suggest data preprocessing and analysis
        return ["data_preprocessor", "notebook_generator", "coder"]

class DataPreprocessor(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=5)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content'][:200]}"
                    for msg in history
                ])
        
        return f"""You are an expert in preprocessing medical research data, particularly NHANES data. You can guide students on:
        1. Cleaning and preprocessing steps for various datasets
        2. Handling missing values
        3. Recoding variables
        4. Merging datasets correctly (especially important for NHANES)
        5. Converting data types appropriately
        6. Creating new derived variables
        
        {context}
        
        Student's query: {query}

        Assistant:"""
    
    def suggest_next_agents(self, query: str, response: str) -> List[str]:
        # After preprocessing, suggest analysis and visualization
        return ["statistical_analyst", "coder", "notebook_generator"]

class NotebookGenerator(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=10)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content'][:200]}"
                    for msg in history
                ])
        
        return f"""You are an expert in creating Jupyter notebooks for medical data analysis. You can generate 
        complete, ready-to-run notebook templates with appropriate sections and code cells. Focus on creating notebooks
        that follow best practices including:
        
        1. Clear markdown documentation
        2. Step-by-step analysis workflow
        3. Data acquisition code
        4. Preprocessing steps
        5. Exploratory analysis
        6. Statistical testing
        7. Visualization
        8. Results interpretation
        
        When asked to generate a notebook, provide the full notebook content with both markdown and code cells, 
        formatted properly for direct use in a Jupyter environment.
        
        {context}
        
        Student's request: {query}

        Assistant:"""
    
    def process_response(self, response: str) -> str:
        """Process to keep notebook formatting intact"""
        # Special handling to preserve notebook format
        return response

class DataVisualizer(Agent):
    def generate_prompt(self, query: str, conversation_id: str = None) -> str:
        context = ""
        if conversation_id:
            history = memory.get_conversation_history(conversation_id, max_messages=5)
            if history:
                context = "Previous conversation:\n" + "\n".join([
                    f"{msg['role']} ({msg.get('agent', 'unknown')}): {msg['content'][:200]}"
                    for msg in history
                ])
        
        return f"""You are an expert in data visualization for medical research, specializing in Python tools 
        like matplotlib, seaborn, and plotly. Provide code for creating effective, publication-quality visualizations
        for medical data, with proper color schemes, annotations, and formatting.
        
        {context}
        
        Student's query: {query}

        Assistant:"""
    
    def process_response(self, response: str) -> str:
        pattern = r'```python\n(.*?)\n```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        # Try without language specifier
        pattern = r'```\n(.*?)\n```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        return response

# Add new agents to the agents dictionary
agents.update({
    "nhanes_expert": NHANESExpert(
        "NHANES Expert",
        "Specialist in NHANES datasets and variables",
        ["nhanes", "cdc", "survey", "dataset", "health survey", "nutrition", "examination"]
    ),
    "data_preprocessor": DataPreprocessor(
        "Data Preprocessor",
        "Expert in cleaning and preparing medical data",
        ["preprocess", "clean", "data preparation", "missing values", "merge", "recode", "filter"]
    ),
    "notebook_generator": NotebookGenerator(
        "Notebook Generator",
        "Creates complete Jupyter notebooks for analysis",
        ["notebook", "jupyter", "template", "report", "analysis workflow", "generate", "script"]
    ),
    "data_visualizer": DataVisualizer(
        "Data Visualizer",
        "Creates publication-quality visualizations",
        ["visualization", "plot", "chart", "graph", "figure", "matplotlib", "seaborn", "plotly"],
        "code"
    )
})

# Add new agents to research team
for agent_id in ["nhanes_expert", "data_preprocessor", "notebook_generator", "data_visualizer"]:
    research_team.add_agent(agent_id, agents[agent_id])

# Create a research team with all agents
research_team = Team("Medical Research Lab")
for agent_id, agent in agents.items():
    research_team.add_agent(agent_id, agent)

# Routing system
def route_query(query: str) -> str:
    """Route a query to the appropriate agent based on keyword matching"""
    query_lower = query.lower()
    max_matches = 0
    selected_agent = "orchestrator"  # Default to orchestrator
    
    for agent_id, agent in agents.items():
        matches = sum(1 for keyword in agent.keywords if keyword.lower() in query_lower)
        if matches > max_matches:
            max_matches = matches
            selected_agent = agent_id
    
    # Store routing decision in memory for context
    memory.set_context("last_routing", {
        "query": query,
        "selected_agent": selected_agent,
        "match_score": max_matches
    })
    
    return selected_agent

# Process query using the selected agent
def process_query(query: str, agent_id: Optional[str] = None, 
                 collaboration: bool = True, project_id: Optional[str] = None) -> Tuple[str, str, List[Dict]]:
    """Process a query with the specified agent or team collaboration"""
    # Use existing or create a new conversation ID
    conversation_id = memory.get_context("current_conversation_id")
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        memory.set_context("current_conversation_id", conversation_id)
    
    # If project_id is provided, associate this conversation with the project
    if project_id:
        memory.set_context("current_project_id", project_id)
        project_manager.set_current_project(project_id)
        project_manager.add_conversation_to_project(project_id, conversation_id)
    else:
        # Check if we're in a project context already
        existing_project_id = memory.get_context("current_project_id")
        if existing_project_id:
            project_manager.add_conversation_to_project(existing_project_id, conversation_id)
    
    # Store the query in memory
    memory.store_message(conversation_id, "user", query)
    
    if collaboration:
        # Use team collaboration
        result = research_team.collaborate(query, agent_id)
        return result["final_response"], result["workflow"][0]["agent_id"], result["workflow"]
    else:
        # Use single agent approach
        if not agent_id:
            agent_id = route_query(query)
        
        agent = agents[agent_id]
        response = agent.respond(query, conversation_id)
        
        return response, agent_id, [{
            "agent_id": agent_id,
            "agent_name": agent.name,
            "response": response
        }]

# Create UI elements
def create_chat_ui():
    # Define UI elements
    query_input = widgets.Text(
        value='',
        placeholder='Ask your medical research question...',
        description='',  # Remove description for cleaner interface
        layout=widgets.Layout(width='90%'),
        disabled=False
    )
    
    submit_button = widgets.Button(
        description='Ask',
        disabled=False,
        button_style='primary',
        tooltip='Submit your query',
        icon='paper-plane',
        layout=widgets.Layout(width='10%')
    )
    
    # Create output areas
    output_area = widgets.Output(layout=widgets.Layout(width='100%', border='1px solid #ddd', 
                                                      padding='10px', margin='10px 0'))
    
    # Create chat history display
    history_output = widgets.Output(layout=widgets.Layout(width='100%', max_height='500px', 
                                                         overflow='auto', padding='10px'))
    
    # Advanced settings section (initially hidden)
    advanced_toggle = widgets.ToggleButton(
        value=False,
        description='Show Advanced Settings',
        disabled=False,
        button_style='info',
        tooltip='Toggle advanced settings',
        layout=widgets.Layout(width='auto', margin='10px 0')
    )
    
    backend_dropdown = widgets.Dropdown(
        options=[('OpenAI', 'openai'), ('NVIDIA NIM', 'nim')],
        value=Config.BACKEND,
        description='Backend:',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )
    
    backend_api_key = widgets.Password(
        value='',
        placeholder='Enter API key (optional)',
        description='API Key:',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )
    
    backend_model = widgets.Text(
        value='',
        placeholder='Enter model name (optional)',
        description='Model:',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )
    
    update_backend_button = widgets.Button(
        description='Update Backend',
        disabled=False,
        button_style='info',
        tooltip='Update backend settings',
        layout=widgets.Layout(width='20%')
    )
    
    # Backend settings handler
    def on_update_backend_clicked(b):
        backend = backend_dropdown.value
        api_key = backend_api_key.value if backend_api_key.value else None
        model = backend_model.value if backend_model.value else None
        
        try:
            set_backend(backend, api_key, model)
            with output_area:
                output_area.clear_output()
                display(HTML(f"<p>Backend updated to {backend.upper()}</p>"))
                if model:
                    display(HTML(f"<p>Model set to: {model}</p>"))
        except Exception as e:
            with output_area:
                output_area.clear_output()
                display(HTML(f"<p>Error updating backend: {str(e)}</p>"))
    
    update_backend_button.on_click(on_update_backend_clicked)
    
    # Create advanced settings panel (hidden by default)
    backend_controls = widgets.HBox([backend_dropdown, backend_api_key, backend_model, update_backend_button])
    advanced_settings = widgets.VBox([
        widgets.HTML("<h4>Backend Settings</h4>"),
        backend_controls
    ])
    advanced_settings.layout.display = 'none'  # Hidden by default
    
    # Toggle advanced settings visibility
    def toggle_advanced_settings(change):
        if change['new']:
            advanced_settings.layout.display = 'block'
            advanced_toggle.description = 'Hide Advanced Settings'
        else:
            advanced_settings.layout.display = 'none'
            advanced_toggle.description = 'Show Advanced Settings'
    
    advanced_toggle.observe(toggle_advanced_settings, names='value')
    
    # Layout
    header = widgets.HTML("<h2>AI Medical Research Assistant</h2>")
    description = widgets.HTML("<p>Ask any medical research question and I'll route it to the right specialist.</p>")
    
    query_controls = widgets.HBox([query_input, submit_button], 
                                 layout=widgets.Layout(width='100%', padding='10px 0'))
    
    # Handle query submission
    def on_submit_button_clicked(b):
        query = query_input.value
        if not query.strip():
            return
            
        with history_output:
            display(HTML(f"<div style='margin-bottom: 10px;'><b>You:</b> {query}</div>"))
        
        with output_area:
            output_area.clear_output()
            display(HTML(f"<p><i>Processing your query using {Config.BACKEND.upper()} backend...</i></p>"))
            
            try:
                # Auto-route to the appropriate agent
                response, agent_id, workflow = process_query(query)
                agent = agents[agent_id]
                
                output_area.clear_output()
                
                # Create new cells with response and reinstantiate the chat widget
                if agent.output_type == "code":
                    # For code response, create a markdown cell with the question and then a code cell with the response
                    escaped_query = json.dumps(f"**Question:** {query}")
                    display(Javascript("""
                        var cell = Jupyter.notebook.insert_cell_below('markdown');
                        cell.set_text(JSON.parse('%s'));
                        cell.execute();
                    """ % escaped_query))
                    
                    # Create code cell with response
                    escaped_response = json.dumps(response)
                    display(Javascript("""
                        var cell = Jupyter.notebook.insert_cell_below('code');
                        cell.set_text(JSON.parse('%s'));
                        Jupyter.notebook.select_next();
                    """ % escaped_response))
                    
                    # Add a comment showing which agent provided the response
                    display(HTML(f"<p>Code generated by {agent.name}</p>"))
                else:
                    # For text response, create a markdown cell with both question and answer
                    formatted_response = f"**Question:** {query}\n\n**Answer ({agent.name}):**\n\n{response}"
                    escaped_response = json.dumps(formatted_response)
                    display(Javascript("""
                        var cell = Jupyter.notebook.insert_cell_below('markdown');
                        cell.set_text(JSON.parse('%s'));
                        cell.execute();
                    """ % escaped_response))
                    
                    display(HTML(f"<p>Response provided by {agent.name}</p>"))
                
                # Create a new cell with the chat widget
                display(Javascript("""
                    var cell = Jupyter.notebook.insert_cell_below('code');
                    cell.set_text("from multi_agent_framework import initialize_chat\\ninitialize_chat()");
                    cell.execute();
                    Jupyter.notebook.select_next();
                """))
                
                with history_output:
                    formatted_response = response.replace('\n', '<br>')
                    display(HTML(f"<div style='margin-bottom: 15px; padding-left: 20px;'><b>AI ({agent.name}):</b><br>{formatted_response}</div>"))
                
            except Exception as e:
                output_area.clear_output()
                display(HTML(f"<p>Error: {str(e)}</p>"))
                traceback.print_exc()
                
        query_input.value = ""
    
    submit_button.on_click(on_submit_button_clicked)
    
    # Handle Enter key in input
    def on_enter(widget):
        on_submit_button_clicked(None)
    
    query_input.on_submit(on_enter)
    
    # Assemble UI with cleaner layout
    ui = widgets.VBox([
        header,
        description,
        widgets.HTML("<h3>Ask Your Question</h3>"),
        query_controls,
        output_area,
        widgets.HTML("<h3>Conversation History</h3>"),
        history_output,
        advanced_toggle,
        advanced_settings
    ], layout=widgets.Layout(padding='20px'))
    
    return ui

# Create a much simpler, stable cell-specific UI for Jupyter notebook integration
def create_cell_ui():
    """Create a simplified UI that remains stable and displays outputs inline with collaboration"""
    query_input = widgets.Text(
        value='',
        placeholder='Ask your next medical research question...',
        description='',
        layout=widgets.Layout(width='90%'),
        disabled=False
    )
    
    submit_button = widgets.Button(
        description='Ask',
        disabled=False,
        button_style='primary',
        tooltip='Submit your query',
        icon='paper-plane',
        layout=widgets.Layout(width='10%')
    )
    
    # Create a larger output area for responses
    output_area = widgets.Output(layout=widgets.Layout(width='100%', border='1px solid #ddd', 
                                                     padding='10px', margin='10px 0',
                                                     min_height='200px'))
    
    query_controls = widgets.HBox([query_input, submit_button], 
                               layout=widgets.Layout(width='100%', padding='10px 0'))
    
    # Add a toggle for collaboration mode
    collaboration_toggle = widgets.Checkbox(
        value=True,
        description='Enable collaboration between agents',
        disabled=False,
        indent=False
    )

    # Function to create a code cell with the provided code
    def create_code_cell(code):
        display(Javascript(f"""
        (function() {{
            var cell = Jupyter.notebook.insert_cell_below('code');
            cell.set_text({json.dumps(code)});
            Jupyter.notebook.select_cell(Jupyter.notebook.find_cell_index(cell));
        }})();
        """))
    
    def on_cell_submit(b):
        query = query_input.value
        if not query.strip():
            return
            
        with output_area:
            output_area.clear_output()
            display(HTML(f"<h4>Question:</h4><p>{query}</p>"))
            display(HTML(f"<p><i>Processing your query using {Config.BACKEND.upper()} backend with {'collaborative' if collaboration_toggle.value else 'single agent'} mode...</i></p>"))
            
            try:
                # Process query with collaboration if enabled
                response, agent_id, workflow = process_query(query, None, collaboration_toggle.value)
                
                # Display the response
                if len(workflow) > 1:
                    # Show collaborative workflow
                    display(HTML(f"<h4>Collaborative Response:</h4>"))
                    for step in workflow:
                        display(HTML(f"<h5>{step['agent_name']} contributed:</h5>"))
                        if agents[step['agent_id']].output_type == "code":
                            display(HTML(f"<pre><code class='python'>{html.escape(step['response'])}</code></pre>"))
                        else:
                            display(HTML(f"<div style='white-space: pre-wrap;'>{html.escape(step['response']).replace('\\n', '<br>')}</div>"))
                else:
                    # Show single agent response
                    agent = agents[agent_id]
                    display(HTML(f"<h4>Answer from {agent.name}:</h4>"))
                    
                    if agent.output_type == "code":
                        display(HTML(f"<pre><code class='python'>{html.escape(response)}</code></pre>"))
                        
                        # Add button to create code cell
                        create_cell_button = widgets.Button(
                            description='Create Code Cell',
                            button_style='info',
                            tooltip='Create a new code cell with this code'
                        )
                        create_cell_button.on_click(lambda b: create_code_cell(response))
                        display(create_cell_button)
                    else:
                        display(HTML(f"<div style='white-space: pre-wrap;'>{html.escape(response).replace('\\n', '<br>')}</div>"))
                
            except Exception as e:
                display(HTML(f"<p>Error: {str(e)}</p>"))
                traceback.print_exc()
                
        # Clear the input for the next question
        query_input.value = ""
    
    submit_button.on_click(on_cell_submit)
    query_input.on_submit(lambda x: on_cell_submit(None))
    
    # Add "Go to Projects" button
    projects_button = widgets.Button(
        description='Manage Projects',
        disabled=False,
        button_style='info',
        tooltip='Go to project management',
        icon='folder'
    )
    
    def on_projects_button_clicked(b):
        display(Javascript("""
            var cell = Jupyter.notebook.insert_cell_below('code');
            cell.set_text("from multi_agent_framework import initialize_project_management\\ninitialize_project_management()");
            cell.execute();
            Jupyter.notebook.select_next();
        """))
    
    projects_button.on_click(on_projects_button_clicked)
    
    # Add to the UI layout
    ui = widgets.VBox([
        widgets.HBox([
            widgets.HTML("<h3>Ask Your Question</h3>"),
            projects_button
        ]),
        query_controls,
        collaboration_toggle,
        output_area
    ], layout=widgets.Layout(padding='10px'))
    
    return ui

# We need to import html for escaping code
import html

# Function to initialize the cell-based chat interface
def initialize_cell_chat():
    """Initialize a simplified chat interface for use in notebook cells"""
    try:
        ui = create_cell_ui()
        display(ui)
    except Exception as e:
        print(f"Error initializing cell chat interface: {str(e)}")
        traceback.print_exc()
        raise

# Function to initialize the chat interface
def initialize_chat():
    """Initialize the chat interface with advanced options.
    For a simpler, cell-based experience, use initialize_cell_chat() instead.
    For project-based research, use initialize_project_management().
    """
    try:
        ui = create_chat_ui()
        display(ui)
        
        # Check if API keys are set
        api_status = []
        if not Config.OPENAI_API_KEY:
            api_status.append("OpenAI API key not set")
        if not Config.NIM_API_KEY:
            api_status.append("NVIDIA NIM API key not set")
            
        if api_status:
            print("WARNING: " + ", ".join(api_status) + ". Please set them in the UI.")
        
        print(f"Multi-agent medical research assistant initialized using {Config.BACKEND.upper()} backend!")
        if Config.BACKEND.lower() == "nim":
            print(f"Using NIM model: {Config.NIM_MODEL}")
        else:
            print(f"Using OpenAI model: {Config.DEFAULT_MODEL}")
        
        # Test connection to the selected backend
        try:
            test_response = query_llm("Hello, please respond with a very brief test message.")
            if "error" in test_response.lower():
                print(f"WARNING: Backend connection test failed: {test_response}")
            else:
                print("Backend connection test successful!")
        except Exception as e:
            print(f"WARNING: Backend connection test failed: {str(e)}")
            
        print("To organize your research into projects, run: from multi_agent_framework import initialize_project_management")
            
    except Exception as e:
        print(f"Error initializing chat interface: {str(e)}")
        traceback.print_exc()
        raise

# Project management classes
class Project:
    def __init__(self, project_id: str, name: str, description: str, created_at: str = None):
        self.project_id = project_id
        self.name = name
        self.description = description
        self.created_at = created_at or datetime.datetime.now().isoformat()
        self.updated_at = self.created_at
        self.conversations = []  # List of conversation IDs
        self.metadata = {}  # Additional metadata like tags, status, etc.
        
    def to_dict(self) -> Dict:
        """Convert project to dictionary for serialization"""
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "conversations": self.conversations,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Project':
        """Create a Project instance from dictionary"""
        project = cls(
            project_id=data["project_id"],
            name=data["name"],
            description=data["description"],
            created_at=data["created_at"]
        )
        project.updated_at = data.get("updated_at", data["created_at"])
        project.conversations = data.get("conversations", [])
        project.metadata = data.get("metadata", {})
        return project

class ProjectManager:
    def __init__(self, storage_dir: str = None):
        """Initialize the project manager with a storage directory"""
        self.storage_dir = storage_dir or os.path.join(os.path.expanduser("~"), ".medical_research_assistant")
        self.projects_file = os.path.join(self.storage_dir, "projects.json")
        self.projects = {}  # Dict of project_id: Project
        self.current_project_id = None
        
        # Ensure storage directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Load existing projects
        self.load_projects()
        
    def load_projects(self):
        """Load projects from disk"""
        if os.path.exists(self.projects_file):
            try:
                with open(self.projects_file, 'r') as f:
                    projects_data = json.load(f)
                    
                for project_data in projects_data:
                    project = Project.from_dict(project_data)
                    self.projects[project.project_id] = project
                    
                print(f"Loaded {len(self.projects)} projects")
            except Exception as e:
                print(f"Error loading projects: {e}")
                self.projects = {}
        
    def save_projects(self):
        """Save projects to disk"""
        try:
            projects_data = [project.to_dict() for project in self.projects.values()]
            with open(self.projects_file, 'w') as f:
                json.dump(projects_data, f, indent=2)
        except Exception as e:
            print(f"Error saving projects: {e}")
            
    def create_project(self, name: str, description: str) -> Project:
        """Create a new project"""
        project_id = str(uuid.uuid4())
        project = Project(project_id, name, description)
        self.projects[project_id] = project
        self.current_project_id = project_id
        self.save_projects()
        return project
        
    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID"""
        return self.projects.get(project_id)
        
    def delete_project(self, project_id: str) -> bool:
        """Delete a project"""
        if project_id in self.projects:
            del self.projects[project_id]
            if self.current_project_id == project_id:
                self.current_project_id = None
            self.save_projects()
            return True
        return False
        
    def list_projects(self) -> List[Project]:
        """List all projects"""
        return list(self.projects.values())
        
    def set_current_project(self, project_id: str) -> Optional[Project]:
        """Set the current active project"""
        if project_id in self.projects:
            self.current_project_id = project_id
            return self.projects[project_id]
        return None
        
    def get_current_project(self) -> Optional[Project]:
        """Get the current active project"""
        if self.current_project_id:
            return self.projects.get(self.current_project_id)
        return None
        
    def add_conversation_to_project(self, project_id: str, conversation_id: str) -> bool:
        """Add a conversation to a project"""
        project = self.get_project(project_id)
        if project:
            if conversation_id not in project.conversations:
                project.conversations.append(conversation_id)
                project.updated_at = datetime.datetime.now().isoformat()
                self.save_projects()
            return True
        return False

# Create a global project manager instance
project_manager = ProjectManager()

# Project Management UI
def create_project_management_ui():
    """Create the project management UI"""
    # Project listing
    projects_dropdown = widgets.Dropdown(
        options=[],
        description='Select Project:',
        disabled=False,
        layout=widgets.Layout(width='80%')
    )
    
    # Project details
    project_name_input = widgets.Text(
        value='',
        placeholder='Project Name',
        description='Name:',
        disabled=False,
        layout=widgets.Layout(width='80%')
    )
    
    project_description_input = widgets.Textarea(
        value='',
        placeholder='Project Description',
        description='Description:',
        disabled=False,
        layout=widgets.Layout(width='80%', height='100px')
    )
    
    # Buttons
    create_project_button = widgets.Button(
        description='Create New Project',
        disabled=False,
        button_style='success',
        tooltip='Create a new project',
        icon='plus'
    )
    
    delete_project_button = widgets.Button(
        description='Delete Project',
        disabled=True,
        button_style='danger',
        tooltip='Delete the selected project',
        icon='trash'
    )
    
    open_project_button = widgets.Button(
        description='Open Project',
        disabled=True,
        button_style='primary',
        tooltip='Open the selected project',
        icon='folder-open'
    )
    
    # Output area
    output_area = widgets.Output(
        layout=widgets.Layout(width='100%', padding='10px', border='1px solid #ddd')
    )
    
    # Load projects into dropdown
    def update_projects_dropdown():
        projects = project_manager.list_projects()
        if not projects:
            projects_dropdown.options = [('No projects', None)]
            projects_dropdown.disabled = True
            delete_project_button.disabled = True
            open_project_button.disabled = True
        else:
            projects_dropdown.options = [('Select a project...', None)] + [(p.name, p.project_id) for p in projects]
            projects_dropdown.disabled = False
            # Selected project will enable/disable buttons
            projects_dropdown.value = None
    
    # Initialize dropdown
    update_projects_dropdown()
    
    # Button handlers
    def on_create_project_clicked(b):
        name = project_name_input.value.strip()
        description = project_description_input.value.strip()
        
        if not name:
            with output_area:
                output_area.clear_output()
                print("Error: Project name is required.")
            return
            
        # Create new project
        project = project_manager.create_project(name, description)
        
        # Update UI
        with output_area:
            output_area.clear_output()
            print(f"Created project: {name}")
            
        # Clear inputs
        project_name_input.value = ''
        project_description_input.value = ''
        
        # Refresh project list
        update_projects_dropdown()
        
        # Select the new project
        projects_dropdown.value = project.project_id
    
    def on_delete_project_clicked(b):
        project_id = projects_dropdown.value
        if not project_id:
            return
            
        project = project_manager.get_project(project_id)
        if not project:
            return
            
        # Delete project
        project_manager.delete_project(project_id)
        
        # Update UI
        with output_area:
            output_area.clear_output()
            print(f"Deleted project: {project.name}")
            
        # Refresh project list
        update_projects_dropdown()
    
    def on_open_project_clicked(b):
        project_id = projects_dropdown.value
        if not project_id:
            return
            
        project = project_manager.get_project(project_id)
        if not project:
            return
            
        # Set current project
        project_manager.set_current_project(project_id)
        
        # Create new cell with chat interface for this project
        with output_area:
            output_area.clear_output()
            print(f"Opening project: {project.name}")
            
        # Create a new cell with the chat widget for this project
        display(Javascript(f"""
            var cell = Jupyter.notebook.insert_cell_below('code');
            cell.set_text("from multi_agent_framework import initialize_project_chat\\ninitialize_project_chat('{project_id}')");
            cell.execute();
            Jupyter.notebook.select_next();
        """))
    
    # Dropdown handler
    def on_projects_dropdown_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            value = change['new']
            delete_project_button.disabled = not value
            open_project_button.disabled = not value
    
    # Connect handlers
    create_project_button.on_click(on_create_project_clicked)
    delete_project_button.on_click(on_delete_project_clicked)
    open_project_button.on_click(on_open_project_clicked)
    projects_dropdown.observe(on_projects_dropdown_change)
    
    # Assemble UI
    header = widgets.HTML("<h2>Medical Research Project Management</h2>")
    description = widgets.HTML("<p>Create and manage your research projects</p>")
    
    existing_projects_section = widgets.VBox([
        widgets.HTML("<h3>Existing Projects</h3>"),
        projects_dropdown,
        widgets.HBox([open_project_button, delete_project_button])
    ])
    
    new_project_section = widgets.VBox([
        widgets.HTML("<h3>Create New Project</h3>"),
        project_name_input,
        project_description_input,
        create_project_button
    ])
    
    ui = widgets.VBox([
        header,
        description,
        existing_projects_section,
        widgets.HTML("<hr>"),
        new_project_section,
        widgets.HTML("<h3>Status</h3>"),
        output_area
    ], layout=widgets.Layout(padding='20px'))
    
    return ui

def initialize_project_chat(project_id: str):
    """Initialize the chat interface for a specific project"""
    project = project_manager.get_project(project_id)
    if not project:
        print(f"Error: Project with ID {project_id} not found.")
        return
        
    # Set current project
    project_manager.set_current_project(project_id)
    
    # Create conversation ID if there are no conversations or use the last one
    conversation_id = None
    if project.conversations:
        conversation_id = project.conversations[-1]
    else:
        conversation_id = str(uuid.uuid4())
        project.conversations.append(conversation_id)
        project_manager.save_projects()
    
    # Set current conversation in memory
    memory.set_context("current_conversation_id", conversation_id)
    memory.set_context("current_project_id", project_id)
    
    # Create UI
    ui = create_project_chat_ui(project)
    display(ui)
    
    print(f"Project '{project.name}' loaded. You can now ask questions related to this research project.")

def create_project_chat_ui(project: Project):
    """Create a chat UI for a specific project"""
    # Similar to create_cell_ui but with project context
    query_input = widgets.Text(
        value='',
        placeholder=f"Ask a question about '{project.name}'...",
        description='',
        layout=widgets.Layout(width='90%'),
        disabled=False
    )
    
    submit_button = widgets.Button(
        description='Ask',
        disabled=False,
        button_style='primary',
        tooltip='Submit your query',
        icon='paper-plane',
        layout=widgets.Layout(width='10%')
    )
    
    # Create a larger output area for responses
    output_area = widgets.Output(layout=widgets.Layout(width='100%', border='1px solid #ddd', 
                                                     padding='10px', margin='10px 0',
                                                     min_height='200px'))
    
    query_controls = widgets.HBox([query_input, submit_button], 
                               layout=widgets.Layout(width='100%', padding='10px 0'))
    
    # Add a toggle for collaboration mode
    collaboration_toggle = widgets.Checkbox(
        value=True,
        description='Enable collaboration between agents',
        disabled=False,
        indent=False
    )
    
    # Add project info section
    project_info = widgets.HTML(
        f"<div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>"
        f"<h4>{project.name}</h4>"
        f"<p><em>{project.description}</em></p>"
        f"<p>Created: {project.created_at[:10]}</p>"
        f"</div>"
    )
    
    # Function to create a code cell with the provided code
    def create_code_cell(code):
        display(Javascript(f"""
        (function() {{
            var cell = Jupyter.notebook.insert_cell_below('code');
            cell.set_text({json.dumps(code)});
            Jupyter.notebook.select_cell(Jupyter.notebook.find_cell_index(cell));
        }})();
        """))
    
    # Function to handle query submission
    def on_cell_submit(b):
        query = query_input.value
        if not query.strip():
            return
        
        # Get current conversation ID from memory
        conversation_id = memory.get_context("current_conversation_id")
        
        with output_area:
            output_area.clear_output()
            display(HTML(f"<h4>Question:</h4><p>{query}</p>"))
            display(HTML(f"<p><i>Processing your query using {Config.BACKEND.upper()} backend with {'collaborative' if collaboration_toggle.value else 'single agent'} mode...</i></p>"))
            
            try:
                # Process query with collaboration if enabled, using the project ID
                response, agent_id, workflow = process_query(
                    query, 
                    None, 
                    collaboration_toggle.value,
                    project.project_id
                )
                
                # Update project's last updated timestamp
                project.updated_at = datetime.datetime.now().isoformat()
                project_manager.save_projects()
                
                # Display the response
                if len(workflow) > 1:
                    # Show collaborative workflow
                    display(HTML(f"<h4>Collaborative Response:</h4>"))
                    for step in workflow:
                        display(HTML(f"<h5>{step['agent_name']} contributed:</h5>"))
                        if agents[step['agent_id']].output_type == "code":
                            display(HTML(f"<pre><code class='python'>{html.escape(step['response'])}</code></pre>"))
                        else:
                            display(HTML(f"<div style='white-space: pre-wrap;'>{html.escape(step['response']).replace('\\n', '<br>')}</div>"))
                else:
                    # Show single agent response
                    agent = agents[agent_id]
                    display(HTML(f"<h4>Answer from {agent.name}:</h4>"))
                    
                    if agent.output_type == "code":
                        display(HTML(f"<pre><code class='python'>{html.escape(response)}</code></pre>"))
                        
                        # Add button to create code cell
                        create_cell_button = widgets.Button(
                            description='Create Code Cell',
                            button_style='info',
                            tooltip='Create a new code cell with this code'
                        )
                        create_cell_button.on_click(lambda b: create_code_cell(response))
                        display(create_cell_button)
                    else:
                        display(HTML(f"<div style='white-space: pre-wrap;'>{html.escape(response).replace('\\n', '<br>')}</div>"))
                
            except Exception as e:
                display(HTML(f"<p>Error: {str(e)}</p>"))
                traceback.print_exc()
                
        # Clear the input for the next question
        query_input.value = ""
    
    # Function to handle back to projects button
    def on_back_to_projects_clicked(b):
        # Create a new cell with the project management UI
        display(Javascript("""
            var cell = Jupyter.notebook.insert_cell_below('code');
            cell.set_text("from multi_agent_framework import initialize_project_management\\ninitialize_project_management()");
            cell.execute();
            Jupyter.notebook.select_next();
        """))
    
    # Add back to projects button
    back_to_projects_button = widgets.Button(
        description='Back to Projects',
        disabled=False,
        button_style='info',
        tooltip='Go back to project management',
        icon='arrow-left'
    )
    back_to_projects_button.on_click(on_back_to_projects_clicked)
    
    # Connect event handlers
    submit_button.on_click(on_cell_submit)
    query_input.on_submit(lambda x: on_cell_submit(None))
    
    # Assemble UI
    ui = widgets.VBox([
        back_to_projects_button,
        project_info,
        widgets.HTML(f"<h3>Ask a Question about '{project.name}'</h3>"),
        query_controls,
        collaboration_toggle,
        output_area
    ], layout=widgets.Layout(padding='10px'))
    
    return ui

def initialize_project_management():
    """Initialize the project management interface"""
    try:
        ui = create_project_management_ui()
        display(ui)
        
        num_projects = len(project_manager.list_projects())
        if num_projects > 0:
            print(f"Found {num_projects} existing projects. Select a project to continue or create a new one.")
        else:
            print("Welcome! Create your first research project to get started.")
            
    except Exception as e:
        print(f"Error initializing project management interface: {str(e)}")
        traceback.print_exc()
        raise

# Notebook generation utilities
class NotebookTemplate:
    """Class for generating notebook templates for various analyses"""
    
    @staticmethod
    def create_nhanes_basic_template(dataset_names: List[str], research_question: str) -> str:
        """Generate a basic NHANES analysis notebook template"""
        template = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": [
                        "# NHANES Data Analysis: " + research_question + "\n",
                        "This notebook provides a complete workflow for analyzing NHANES data to investigate the research question."
                    ]
                },
                {
                    "cell_type": "markdown",
                    "source": [
                        "## 1. Setup and Import Libraries\n",
                        "First, we'll import the necessary libraries for data processing, analysis, and visualization."
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Import standard data analysis libraries\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "\n",
                        "# Statistical analysis\n",
                        "from scipy import stats\n",
                        "import statsmodels.api as sm\n",
                        "from statsmodels.formula.api import ols\n",
                        "\n",
                        "# NHANES specific libraries\n",
                        "import requests\n",
                        "from io import BytesIO\n",
                        "\n",
                        "# Set plot style\n",
                        "plt.style.use('seaborn-v0_8-whitegrid')\n",
                        "sns.set_context('notebook')\n",
                        "\n",
                        "# Display settings\n",
                        "pd.set_option('display.max_columns', 100)\n",
                        "pd.set_option('display.max_rows', 100)"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "source": [
                        "## 2. Data Acquisition\n",
                        "Now we'll download the NHANES datasets needed for this analysis."
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Define NHANES datasets to download\n",
                        "datasets = " + str(dataset_names) + "\n",
                        "\n",
                        "# Function to download NHANES data\n",
                        "def download_nhanes_data(cycle, dataset):\n",
                        "    \"\"\"Download an NHANES dataset for a specific cycle\"\"\"\n",
                        "    base_url = f'https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/'\n",
                        "    url = base_url + dataset + '.XPT'\n",
                        "    \n",
                        "    print(f\"Downloading {dataset} from {cycle}...\")\n",
                        "    try:\n",
                        "        response = requests.get(url)\n",
                        "        response.raise_for_status()\n",
                        "        data = pd.read_sas(BytesIO(response.content))\n",
                        "        print(f\"Downloaded {dataset}: {data.shape[0]} rows, {data.shape[1]} columns\")\n",
                        "        return data\n",
                        "    except Exception as e:\n",
                        "        print(f\"Error downloading {dataset}: {e}\")\n",
                        "        return None\n",
                        "\n",
                        "# Download selected datasets (example for 2017-2018 cycle)\n",
                        "cycle = '2017-2018'\n",
                        "data_frames = {}\n",
                        "\n",
                        "for dataset in datasets:\n",
                        "    data_frames[dataset] = download_nhanes_data(cycle, dataset)"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "source": [
                        "## 3. Data Preprocessing\n",
                        "Next, we'll clean and prepare the data for analysis."
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Inspect dataset structure\n",
                        "for name, df in data_frames.items():\n",
                        "    if df is not None:\n",
                        "        print(f\"\\nDataset: {name}\")\n",
                        "        print(df.info())\n",
                        "        print(\"\\nSample:\")\n",
                        "        display(df.head())"
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Merge datasets using SEQN as the unique identifier\n",
                        "def merge_nhanes_data(data_frames):\n",
                        "    \"\"\"Merge NHANES datasets using SEQN as key\"\"\"\n",
                        "    merged_data = None\n",
                        "    \n",
                        "    for name, df in data_frames.items():\n",
                        "        if df is None:\n",
                        "            continue\n",
                        "            \n",
                        "        # Convert column names to lowercase for consistency\n",
                        "        df.columns = [col.lower() for col in df.columns]\n",
                        "        \n",
                        "        if merged_data is None:\n",
                        "            merged_data = df.copy()\n",
                        "        else:\n",
                        "            # Merge with previous datasets\n",
                        "            merged_data = pd.merge(merged_data, df, on='seqn', how='inner')\n",
                        "    \n",
                        "    return merged_data\n",
                        "\n",
                        "# Merge the datasets\n",
                        "merged_data = merge_nhanes_data(data_frames)\n",
                        "print(f\"Merged dataset shape: {merged_data.shape}\")\n",
                        "merged_data.head()"
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Clean and preprocess data\n",
                        "def clean_nhanes_data(df):\n",
                        "    \"\"\"Clean NHANES data by handling missing values and recoding\"\"\"\n",
                        "    # Make a copy to avoid modifying the original\n",
                        "    data = df.copy()\n",
                        "    \n",
                        "    # Check for missing values\n",
                        "    missing_summary = data.isnull().sum()\n",
                        "    print(\"\\nMissing values summary:\")\n",
                        "    print(missing_summary[missing_summary > 0])\n",
                        "    \n",
                        "    # Recode special values (NHANES often uses negative values for missing/refused)\n",
                        "    for col in data.columns:\n",
                        "        if data[col].dtype in [np.float64, np.int64]:\n",
                        "            # Replace negative values with NaN (typical NHANES coding)\n",
                        "            data[col] = data[col].apply(lambda x: np.nan if x is not None and x < 0 else x)\n",
                        "    \n",
                        "    # Drop rows with too many missing values (customize this threshold)\n",
                        "    missing_threshold = 0.5  # 50% threshold\n",
                        "    data = data.dropna(thresh=int(len(data.columns) * (1-missing_threshold)), axis=0)\n",
                        "    \n",
                        "    print(f\"\\nShape after cleaning: {data.shape}\")\n",
                        "    \n",
                        "    return data\n",
                        "\n",
                        "# Clean the merged data\n",
                        "clean_data = clean_nhanes_data(merged_data)"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "source": [
                        "## 4. Exploratory Data Analysis\n",
                        "Let's explore the data through summary statistics and visualizations."
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Summary statistics\n",
                        "summary_stats = clean_data.describe()\n",
                        "summary_stats"
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Distribution visualizations for key variables\n",
                        "def plot_distributions(data, variables, figsize=(15, 10)):\n",
                        "    \"\"\"Plot distributions for selected variables\"\"\"\n",
                        "    if not variables:\n",
                        "        print(\"No variables selected for plotting\")\n",
                        "        return\n",
                        "        \n",
                        "    # Filter to include only variables in the dataset\n",
                        "    available_vars = [var for var in variables if var in data.columns]\n",
                        "    \n",
                        "    if not available_vars:\n",
                        "        print(\"None of the specified variables are in the dataset\")\n",
                        "        return\n",
                        "    \n",
                        "    n_cols = 2\n",
                        "    n_rows = (len(available_vars) + 1) // n_cols\n",
                        "    \n",
                        "    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)\n",
                        "    axes = axes.flatten()\n",
                        "    \n",
                        "    for i, var in enumerate(available_vars):\n",
                        "        ax = axes[i]\n",
                        "        \n",
                        "        # Check if variable is categorical or continuous\n",
                        "        if data[var].nunique() < 10 or data[var].dtype == 'object':\n",
                        "            # Categorical variable\n",
                        "            sns.countplot(x=var, data=data, ax=ax)\n",
                        "            ax.set_title(f'Distribution of {var}')\n",
                        "            if data[var].nunique() > 5:\n",
                        "                ax.tick_params(axis='x', rotation=45)\n",
                        "        else:\n",
                        "            # Continuous variable\n",
                        "            sns.histplot(data[var].dropna(), kde=True, ax=ax)\n",
                        "            ax.set_title(f'Distribution of {var}')\n",
                        "            \n",
                        "    # Hide any unused subplots\n",
                        "    for j in range(i+1, len(axes)):\n",
                        "        fig.delaxes(axes[j])\n",
                        "        \n",
                        "    plt.tight_layout()\n",
                        "    plt.show()\n",
                        "\n",
                        "# Example: Plot distributions for key variables (replace with actual variable names)\n",
                        "key_variables = ['bmxbmi', 'bmxwt', 'bmxht']  # Example variables\n",
                        "plot_distributions(clean_data, key_variables)"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "source": [
                        "## 5. Statistical Analysis\n",
                        "Now let's conduct statistical tests to answer our research question."
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Statistical analysis (customize based on your research question)\n",
                        "# Example: Linear regression\n",
                        "def run_regression_analysis(data, dependent_var, independent_vars):\n",
                        "    \"\"\"Run a multiple linear regression analysis\"\"\"\n",
                        "    # Prepare formula for statsmodels\n",
                        "    formula = f\"{dependent_var} ~ {' + '.join(independent_vars)}\"\n",
                        "    \n",
                        "    try:\n",
                        "        # Fit the model\n",
                        "        model = ols(formula, data=data).fit()\n",
                        "        \n",
                        "        # Print summary\n",
                        "        print(model.summary())\n",
                        "        \n",
                        "        return model\n",
                        "    except Exception as e:\n",
                        "        print(f\"Error running regression: {e}\")\n",
                        "        return None\n",
                        "\n",
                        "# Example regression (replace with variables relevant to your research question)\n",
                        "# dependent_var = 'bmxbmi'  # Body Mass Index\n",
                        "# independent_vars = ['ridageyr', 'riagendr']  # Age and gender\n",
                        "# model = run_regression_analysis(clean_data, dependent_var, independent_vars)"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "source": [
                        "## 6. Visualization of Results\n",
                        "Let's create publication-quality visualizations of our findings."
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Create publication-quality visualizations\n",
                        "def create_publication_plot(data, x, y, hue=None, kind='scatter', figsize=(10, 6)):\n",
                        "    \"\"\"Create a publication-quality plot for key relationships\"\"\"\n",
                        "    plt.figure(figsize=figsize)\n",
                        "    \n",
                        "    if kind == 'scatter':\n",
                        "        ax = sns.scatterplot(x=x, y=y, hue=hue, data=data)\n",
                        "        # Add regression line\n",
                        "        sns.regplot(x=x, y=y, data=data, scatter=False, ax=ax)\n",
                        "    elif kind == 'box':\n",
                        "        ax = sns.boxplot(x=x, y=y, hue=hue, data=data)\n",
                        "    elif kind == 'bar':\n",
                        "        ax = sns.barplot(x=x, y=y, hue=hue, data=data)\n",
                        "    elif kind == 'violin':\n",
                        "        ax = sns.violinplot(x=x, y=y, hue=hue, data=data, inner='quartile')\n",
                        "    else:\n",
                        "        print(f\"Plot type {kind} not recognized\")\n",
                        "        return\n",
                        "    \n",
                        "    # Set labels and title\n",
                        "    plt.xlabel(x, fontsize=12)\n",
                        "    plt.ylabel(y, fontsize=12)\n",
                        "    plt.title(f'Relationship between {x} and {y}', fontsize=14)\n",
                        "    \n",
                        "    # Style improvements\n",
                        "    plt.tight_layout()\n",
                        "    plt.grid(alpha=0.3)\n",
                        "    \n",
                        "    if hue:\n",
                        "        plt.legend(title=hue, fontsize=10, title_fontsize=12)\n",
                        "    \n",
                        "    plt.show()\n",
                        "\n",
                        "# Example visualization (replace with variables from your analysis)\n",
                        "# create_publication_plot(clean_data, x='ridageyr', y='bmxbmi', hue='riagendr')"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "source": [
                        "## 7. Interpretation and Conclusions\n",
                        "In this section, summarize your findings and their implications."
                    ]
                },
                {
                    "cell_type": "markdown",
                    "source": [
                        "*Write your conclusions here after completing the analysis. Consider:*\n",
                        "* What patterns or relationships did you find?\n",
                        "* Were your hypotheses supported or rejected?\n",
                        "* What are the limitations of your analysis?\n",
                        "* What implications do your findings have for medical research or practice?\n",
                        "* What future research directions would you recommend?"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.10"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Convert to JSON and back to string for formatting
        import json
        notebook_json = json.dumps(template, indent=2)
        return notebook_json
    
    @staticmethod
    def create_custom_data_template(dataset_description: str, research_question: str) -> str:
        """Generate a template for custom data analysis"""
        # Similar to the NHANES template but with custom data loading instead
        # Implementation here...
        pass

# Call this function to start the chat interface
if __name__ == "__main__" or 'get_ipython' in globals():
    try:
        initialize_chat()
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        traceback.print_exc()
