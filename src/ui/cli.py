import argparse
import json
import sys
from typing import Dict, Any

from src.orchestrator import ResearchOrchestrator
from src.agents.researcher import ResearchAgent
from src.agents.statistician import StatisticianAgent

class ResearchCLI:
    """Command-line interface for the medical research AI framework."""
    
    def __init__(self):
        self.orchestrator = ResearchOrchestrator()
        self._setup_agents()
        
    def _setup_agents(self) -> None:
        """Initialize and register all required agents."""
        # Create agents
        researcher = ResearchAgent()
        statistician = StatisticianAgent()
        
        # Register with orchestrator
        self.orchestrator.register_agent("researcher", researcher)
        self.orchestrator.register_agent("statistician", statistician)
    
    def run(self) -> None:
        """Run the CLI application."""
        parser = argparse.ArgumentParser(description="Medical Research AI Assistant")
        
        # Add commands
        subparsers = parser.add_subparsers(dest="command", help="Command to run")
        
        # Interactive mode
        interactive_parser = subparsers.add_parser("interactive", help="Start interactive session")
        
        # Direct query mode
        query_parser = subparsers.add_parser("query", help="Send a single query")
        query_parser.add_argument("text", help="The research question or query")
        
        # Study design command
        design_parser = subparsers.add_parser("design", help="Design a research study")
        design_parser.add_argument("question", help="The research question")
        design_parser.add_argument("--domain", help="Medical domain", default="general medicine")
        
        # Statistical analysis command
        stats_parser = subparsers.add_parser("stats", help="Get statistical advice")
        stats_parser.add_argument("question", help="The statistics question")
        stats_parser.add_argument("--data-type", help="Type of data", choices=["continuous", "categorical", "time-series"])
        stats_parser.add_argument("--outcome", help="Outcome type", choices=["continuous", "binary", "time-to-event"])
        
        args = parser.parse_args()
        
        if args.command == "interactive":
            self._run_interactive_mode()
        elif args.command == "query":
            response = self.orchestrator.process_query(args.text)
            self._pretty_print_response(response)
        elif args.command == "design":
            task = {
                "type": "design_study",
                "question": args.question,
                "domain": args.domain
            }
            researcher = self.orchestrator.get_agent("researcher")
            if researcher:
                response = researcher.process_task(task)
                self._pretty_print_response(response)
            else:
                print("Researcher agent not available.")
        elif args.command == "stats":
            task = {
                "type": "suggest_analysis",
                "question": args.question,
                "data_type": args.data_type,
                "outcome_type": args.outcome
            }
            statistician = self.orchestrator.get_agent("statistician")
            if statistician:
                response = statistician.process_task(task)
                self._pretty_print_response(response)
            else:
                print("Statistician agent not available.")
        else:
            parser.print_help()
    
    def _run_interactive_mode(self) -> None:
        """Run in interactive mode, accepting consecutive queries."""
        print("=== Medical Research AI Assistant ===")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'help' for assistance.")
        
        while True:
            try:
                query = input("\nHow can I help with your research? > ")
                
                if query.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break
                elif query.lower() == "help":
                    self._show_help()
                    continue
                
                # Process the query
                response = self.orchestrator.process_query(query)
                self._pretty_print_response(response)
                
            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _pretty_print_response(self, response: Dict[str, Any]) -> None:
        """Format and print a response nicely."""
        if "error" in response:
            print(f"Error: {response['error']}")
            return
        
        print("\n=== Response ===")
        
        if "agent" in response:
            print(f"Agent: {response['agent']}")
        
        if "agents" in response:
            print(f"Agents consulted: {', '.join(response['agents'])}")
        
        if "response" in response:
            self._print_nested_dict(response["response"])
        
        if "synthesized_response" in response:
            print("\nSynthesized response:")
            self._print_nested_dict(response["synthesized_response"])
    
    def _print_nested_dict(self, d: Dict[str, Any], indent: int = 0) -> None:
        """Print a nested dictionary with indentation."""
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{'  ' * indent}{key}:")
                self._print_nested_dict(value, indent + 1)
            elif isinstance(value, list):
                print(f"{'  ' * indent}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        self._print_nested_dict(item, indent + 1)
                    else:
                        print(f"{'  ' * (indent + 1)}- {item}")
            else:
                print(f"{'  ' * indent}{key}: {value}")
    
    def _show_help(self) -> None:
        """Show help information."""
        print("\n=== Help Information ===")
        print("You can ask questions about:")
        print("  - Research study design")
        print("  - Statistical analysis methods")
        print("  - Sample size calculations")
        print("  - Refining research questions")
        print("\nExample queries:")
        print("  - How do I design a study to compare two treatments for diabetes?")
        print("  - What statistical test should I use for comparing patient outcomes?")
        print("  - Can you help me calculate the sample size for my study?")
        print("  - How can I improve my research question about COVID-19 treatments?")

if __name__ == "__main__":
    cli = ResearchCLI()
    cli.run()
