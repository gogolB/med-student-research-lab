from flask import Flask, request, jsonify
import sys
import os
import json
import re
import traceback
import requests
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

# Import the multi-agent framework
from multi_agent_framework import agents, route_query, process_query, query_llm, Config

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok", "backend": Config.BACKEND})

@app.route('/query', methods=['POST'])
def handle_query():
    """Process a query and return the response"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing query parameter"}), 400
        
        query = data['query']
        
        # Process the query with the appropriate agent
        response, agent_id = process_query(query)
        agent = agents[agent_id]
        
        # Return the response and agent info
        return jsonify({
            "response": response,
            "agent": {
                "id": agent_id,
                "name": agent.name,
                "description": agent.description,
                "output_type": agent.output_type
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/set_backend', methods=['POST'])
def set_backend():
    """Update the backend configuration"""
    try:
        data = request.get_json()
        if not data or 'backend' not in data:
            return jsonify({"error": "Missing backend parameter"}), 400
        
        backend = data['backend']
        api_key = data.get('api_key')
        model = data.get('model')
        
        # Import set_backend function
        from multi_agent_framework import set_backend as update_backend
        
        # Update the backend
        update_backend(backend, api_key, model)
        
        return jsonify({"status": "ok", "backend": backend})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the Flask server
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Medical AI backend server on port {port}...")
    app.run(host='127.0.0.1', port=port, debug=False)
