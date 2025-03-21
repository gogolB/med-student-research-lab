import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from typing import Dict, Any

from src.orchestrator import ResearchOrchestrator
from src.agents.researcher import ResearchAgent
from src.agents.statistician import StatisticianAgent

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')

class ResearchWebUI:
    """Web interface for the medical research AI framework."""
    
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

# Initialize the web UI
web_ui = ResearchWebUI()

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    """Handle direct queries to the system."""
    if request.method == 'POST':
        query_text = request.form.get('query', '')
        if query_text:
            response = web_ui.orchestrator.process_query(query_text)
            return render_template('results.html', query=query_text, response=response)
    return render_template('query.html')

@app.route('/design', methods=['GET', 'POST'])
def design():
    """Handle study design requests."""
    if request.method == 'POST':
        question = request.form.get('question', '')
        domain = request.form.get('domain', 'general medicine')
        
        if question:
            task = {
                "type": "design_study",
                "question": question,
                "domain": domain
            }
            researcher = web_ui.orchestrator.get_agent("researcher")
            if researcher:
                response = researcher.process_task(task)
                return render_template('results.html', query=question, response=response)
            else:
                flash("Researcher agent not available.")
    
    return render_template('design.html')

@app.route('/stats', methods=['GET', 'POST'])
def stats():
    """Handle statistical analysis requests."""
    if request.method == 'POST':
        question = request.form.get('question', '')
        data_type = request.form.get('data_type', '')
        outcome = request.form.get('outcome', '')
        
        if question:
            task = {
                "type": "suggest_analysis",
                "question": question,
                "data_type": data_type,
                "outcome_type": outcome
            }
            statistician = web_ui.orchestrator.get_agent("statistician")
            if statistician:
                response = statistician.process_task(task)
                return render_template('results.html', query=question, response=response)
            else:
                flash("Statistician agent not available.")
    
    return render_template('stats.html')

@app.route('/api/query', methods=['POST'])
def api_query():
    """API endpoint for queries."""
    data = request.json
    query_text = data.get('query', '')
    if query_text:
        response = web_ui.orchestrator.process_query(query_text)
        return jsonify(response)
    return jsonify({"error": "No query provided"})

if __name__ == "__main__":
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
