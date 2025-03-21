from typing import Dict, Any, List, Optional, Type
from src.agents.base_agent import BaseAgent

class ResearchOrchestrator:
    """Coordinates multiple agents to assist with medical research."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.conversation_history: List[Dict[str, Any]] = []
    
    def register_agent(self, agent_id: str, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent_id] = agent
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a user query by routing to appropriate agents."""
        if context is None:
            context = {}
        
        # Determine which agent(s) should handle this query
        # This could be enhanced with more sophisticated routing
        task = self._create_task_from_query(query, context)
        relevant_agents = self._identify_relevant_agents(task)
        
        if not relevant_agents:
            return {"error": "No suitable agents found for this query"}
        
        # If only one agent is needed
        if len(relevant_agents) == 1:
            agent_id = list(relevant_agents.keys())[0]
            agent = relevant_agents[agent_id]
            result = agent.process_task(task)
            
            # Store in conversation history
            self._update_conversation_history(query, agent_id, result)
            
            return {
                "agent": agent_id,
                "response": result
            }
        
        # If multiple agents are needed, coordinate their collaboration
        return self._coordinate_multi_agent_response(task, relevant_agents)
    
    def _create_task_from_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a user query into a structured task."""
        # This is a simplified version - in practice, you might use NLP to extract
        # task type, parameters, etc. from the query
        
        task = {
            "query": query,
            "context": context,
            # Placeholder for task type inference
            "type": self._infer_task_type(query)
        }
        
        return task
    
    def _infer_task_type(self, query: str) -> str:
        """Infer the type of task from the query text."""
        # Simplified task type inference
        query_lower = query.lower()
        
        if "design" in query_lower and "study" in query_lower:
            return "design_study"
        elif "analysis" in query_lower or "analyze" in query_lower:
            return "suggest_analysis"
        elif "sample size" in query_lower:
            return "calculate_sample_size"
        elif "question" in query_lower or "hypothesis" in query_lower:
            return "refine_question"
        else:
            return "general_assistance"
    
    def _identify_relevant_agents(self, task: Dict[str, Any]) -> Dict[str, BaseAgent]:
        """Identify which agents can handle a given task."""
        task_type = task.get("type", "")
        
        # Map task types to agent IDs
        task_to_agent_map = {
            "design_study": ["researcher"],
            "suggest_analysis": ["statistician"],
            "calculate_sample_size": ["statistician"],
            "refine_question": ["researcher"],
            "interpret_results": ["statistician", "clinician"],
            "general_assistance": list(self.agents.keys())
        }
        
        relevant_agent_ids = task_to_agent_map.get(task_type, [])
        return {agent_id: self.agents[agent_id] for agent_id in relevant_agent_ids if agent_id in self.agents}
    
    def _coordinate_multi_agent_response(self, task: Dict[str, Any], relevant_agents: Dict[str, BaseAgent]) -> Dict[str, Any]:
        """Coordinate responses from multiple agents."""
        agent_responses = {}
        
        for agent_id, agent in relevant_agents.items():
            response = agent.process_task(task)
            agent_responses[agent_id] = response
        
        # Combine responses (this could be enhanced with more sophisticated integration)
        combined_response = self._synthesize_responses(task, agent_responses)
        
        # Update conversation history
        self._update_conversation_history(task["query"], "multiple", combined_response)
        
        return {
            "agents": list(relevant_agents.keys()),
            "individual_responses": agent_responses,
            "synthesized_response": combined_response
        }
    
    def _synthesize_responses(self, task: Dict[str, Any], agent_responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize responses from multiple agents into a coherent response."""
        # This is a placeholder for more sophisticated response synthesis
        return {
            "task": task.get("type", "unknown"),
            "contributions": [
                {"agent": agent_id, "key_points": list(response.keys())}
                for agent_id, response in agent_responses.items()
            ],
            "summary": "Multiple agents have provided perspectives on this question."
        }
    
    def _update_conversation_history(self, query: str, agent_id: str, response: Dict[str, Any]) -> None:
        """Update the conversation history with a new interaction."""
        self.conversation_history.append({
            "query": query,
            "agent": agent_id,
            "response": response,
            "timestamp": "current_time_here"  # In actual implementation, use datetime
        })
