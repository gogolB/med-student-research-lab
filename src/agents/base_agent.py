from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode
import json
import sys
import os

# Add the project root to sys.path if necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.telemetry.setup import setup_telemetry

class BaseAgent(ABC):
    """Base class for all research agents in the system."""
    
    def __init__(self, name: str, expertise: List[str]):
        self.name = name
        self.expertise = expertise
        self.memory: List[Dict[str, Any]] = []
        
        # Set up OpenTelemetry tracing
        self.tracer, self.meter = setup_telemetry(f"agent.{name}")
        
    @abstractmethod
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task and return results."""
        with self.tracer.start_as_current_span("process_task") as span:
            # Add task details as span attributes
            span.set_attribute("agent.name", self.name)
            span.set_attribute("task.type", task.get("type", "unknown"))
            span.set_attribute("task.details", json.dumps(task, default=str))
            
            try:
                # Actual implementation must be provided by subclasses
                result = self._process_task_impl(task)
                
                # Log the result
                span.set_attribute("result.success", True)
                span.set_attribute("result.data", json.dumps(result, default=str)[:1000])  # Truncate if too large
                return result
            except Exception as e:
                # Log the error
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
                span.set_attribute("result.success", False)
                span.set_attribute("error.message", str(e))
                raise
    
    @abstractmethod
    def _process_task_impl(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Actual implementation of task processing to be provided by subclasses."""
        pass
    
    def add_to_memory(self, item: Dict[str, Any]) -> None:
        """Store information in the agent's memory."""
        with self.tracer.start_as_current_span("add_to_memory") as span:
            span.set_attribute("agent.name", self.name)
            span.set_attribute("memory.item", json.dumps(item, default=str)[:1000])
            
            self.memory.append(item)
    
    def recall_from_memory(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant information from memory based on a query."""
        with self.tracer.start_as_current_span("recall_from_memory") as span:
            span.set_attribute("agent.name", self.name)
            span.set_attribute("memory.query", json.dumps(query, default=str))
            
            # Simple implementation - can be enhanced with vector search
            results = []
            for item in self.memory:
                if all(item.get(k) == v for k, v in query.items()):
                    results.append(item)
            
            span.set_attribute("memory.results_count", len(results))
            return results
    
    def collaborate(self, other_agent: 'BaseAgent', task: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with another agent on a task."""
        with self.tracer.start_as_current_span("collaborate") as span:
            span.set_attribute("agent.name", self.name)
            span.set_attribute("collaborator.name", other_agent.name)
            span.set_attribute("task.details", json.dumps(task, default=str)[:1000])
            
            # Default implementation for agent collaboration
            with self.tracer.start_as_current_span("my_processing") as my_span:
                my_result = self.process_task(task)
                my_span.set_attribute("result.data", json.dumps(my_result, default=str)[:1000])
            
            with self.tracer.start_as_current_span("collaborator_processing") as their_span:
                their_result = other_agent.process_task({**task, "context": my_result})
                their_span.set_attribute("result.data", json.dumps(their_result, default=str)[:1000])
            
            result = {
                "joint_result": {
                    "from": self.name,
                    "result": my_result,
                    "collaborator": other_agent.name,
                    "collaborator_result": their_result
                }
            }
            
            span.set_attribute("collaboration.result", json.dumps(result, default=str)[:1000])
            return result
