from flask import Flask, render_template, request, jsonify
import sys
import os
import json
import traceback

# Add better error handling for imports
try:
    print("Attempting to add parent directory to sys.path...")
    parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    print(f"Parent directory: {parent_dir}")
    sys.path.append(parent_dir)
    
    print("Checking if multi_agent_framework.py exists...")
    framework_path = os.path.join(parent_dir, 'multi_agent_framework.py')
    if not os.path.exists(framework_path):
        print(f"ERROR: multi_agent_framework.py not found at {framework_path}")
        sys.exit(1)
    else:
        print(f"Found framework at {framework_path}")
    
    print("Importing from multi_agent_framework...")
    from multi_agent_framework import agents, route_query, process_query, query_llm, Config, project_manager
    print("Successfully imported multi_agent_framework modules")
except ImportError as e:
    print(f"ERROR: Failed to import from multi_agent_framework: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred during import: {e}")
    traceback.print_exc()
    sys.exit(1)

app = Flask(__name__)

# Check if templates directory exists
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
if not os.path.exists(templates_dir):
    print(f"WARNING: Templates directory not found at {templates_dir}")
    print("Creating templates directory...")
    os.makedirs(templates_dir, exist_ok=True)

# Check if index.html exists
index_path = os.path.join(templates_dir, 'index.html')
if not os.path.exists(index_path):
    print(f"WARNING: index.html not found at {index_path}")
    print("Creating a basic index.html...")
    with open(index_path, 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Medical Research AI</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #2c3e50; }
        .container { max-width: 800px; margin: 0 auto; }
        textarea { width: 100%; height: 100px; margin-bottom: 10px; }
        button { padding: 10px 15px; background: #3498db; color: white; border: none; cursor: pointer; }
        #response { margin-top: 20px; border: 1px solid #ddd; padding: 15px; min-height: 100px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Medical Research AI Assistant</h1>
        <textarea id="query" placeholder="Enter your medical research question here..."></textarea>
        <button onclick="submitQuery()">Submit Query</button>
        <div id="response"></div>
    </div>
    
    <script>
        function submitQuery() {
            const query = document.getElementById('query').value;
            const responseDiv = document.getElementById('response');
            responseDiv.innerHTML = "Processing...";
            
            fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            })
            .then(response => response.json())
            .then(data => {
                responseDiv.innerHTML = data.response.replace(/\\n/g, '<br>');
            })
            .catch(error => {
                responseDiv.innerHTML = "Error: " + error;
            });
        }
    </script>
</body>
</html>
        """)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def handle_query():
    """Process a query and return the response"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Check if a project_id was provided
    project_id = data.get('project_id')
    
    try:
        # Process the query, potentially in a project context
        response, agent_id, workflow = process_query(query, None, True, project_id)
        
        # Build a more detailed response that includes workflow information
        result = {
            "response": response,
            "agent_id": agent_id,
            "agent_name": agents[agent_id].name,
            "workflow": [
                {
                    "agent_id": step["agent_id"],
                    "agent_name": step["agent_name"],
                    "contribution": step["response"]
                } for step in workflow
            ]
        }
        
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/execute', methods=['POST'])
def execute_code():
    """Execute Python code and return the result"""
    try:
        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({"error": "Missing code parameter"}), 400
        
        code = data['code']
        
        # Create a temporary file for code execution
        temp_file = os.path.join(os.path.dirname(__file__), 'temp_exec.py')
        with open(temp_file, 'w') as f:
            f.write(code)
        
        # Capture output from code execution
        import subprocess
        result = subprocess.run(
            [sys.executable, temp_file], 
            capture_output=True, 
            text=True
        )
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Return the execution result
        return jsonify({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/backend', methods=['GET'])
def get_backend_info():
    """Get the current backend information"""
    return jsonify({
        "backend": Config.BACKEND,
        "model": Config.NIM_MODEL if Config.BACKEND == "nim" else Config.DEFAULT_MODEL
    })

@app.route('/api/backend', methods=['POST'])
def set_backend_config():
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
        
        return jsonify({
            "status": "ok", 
            "backend": backend,
            "model": Config.NIM_MODEL if backend == "nim" else Config.DEFAULT_MODEL
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# New routes for project management
@app.route('/api/projects', methods=['GET'])
def get_projects():
    """Get all projects"""
    try:
        projects = project_manager.list_projects()
        return jsonify([
            {
                "id": p.project_id,
                "name": p.name,
                "description": p.description,
                "created_at": p.created_at,
                "updated_at": p.updated_at,
                "conversations_count": len(p.conversations)
            } for p in projects
        ])
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/projects', methods=['POST'])
def create_project():
    """Create a new project"""
    data = request.json
    name = data.get('name', '').strip()
    description = data.get('description', '').strip()
    
    if not name:
        return jsonify({"error": "Project name is required"}), 400
    
    try:
        project = project_manager.create_project(name, description)
        return jsonify({
            "id": project.project_id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at,
            "updated_at": project.updated_at
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    """Get details of a specific project"""
    try:
        project = project_manager.get_project(project_id)
        if not project:
            return jsonify({"error": "Project not found"}), 404
            
        # Get conversation snippets for this project
        conversation_snippets = []
        for conv_id in project.conversations:
            history = memory.get_conversation_history(conv_id, max_messages=3)
            if history:
                snippet = {
                    "id": conv_id,
                    "preview": history[-1]["content"][:100] + "..." if len(history[-1]["content"]) > 100 else history[-1]["content"],
                    "last_message_time": history[-1]["timestamp"],
                    "message_count": len(history)
                }
                conversation_snippets.append(snippet)
        
        return jsonify({
            "id": project.project_id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "conversations": conversation_snippets,
            "metadata": project.metadata
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project"""
    try:
        success = project_manager.delete_project(project_id)
        if not success:
            return jsonify({"error": "Project not found"}), 404
            
        return jsonify({"success": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/projects/<project_id>/query', methods=['POST'])
def project_query(project_id):
    """Process a query in the context of a specific project"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        # Check if the project exists
        project = project_manager.get_project(project_id)
        if not project:
            return jsonify({"error": "Project not found"}), 404
        
        # Process the query in this project's context
        response, agent_id, workflow = process_query(query, None, True, project_id)
        
        # Build response
        result = {
            "response": response,
            "agent_id": agent_id,
            "agent_name": agents[agent_id].name,
            "workflow": [
                {
                    "agent_id": step["agent_id"],
                    "agent_name": step["agent_name"],
                    "contribution": step["response"]
                } for step in workflow
            ],
            "project": {
                "id": project.project_id,
                "name": project.name
            }
        }
        
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Medical Research AI Web Application...")
    print(f"Using {Config.BACKEND.upper()} backend")
    
    # Check if API keys are set
    if Config.BACKEND == "openai" and not Config.OPENAI_API_KEY:
        print("WARNING: OpenAI API key not set. Set it in the .env file or through the web interface.")
    elif Config.BACKEND == "nim" and not Config.NIM_API_KEY:
        print("WARNING: NVIDIA NIM API key not set. Set it in the .env file or through the web interface.")
    
    # Changed default port from 5000 to 8000
    port = int(os.environ.get('PORT', 8000))
    
    # Add support for HTTPS and HTTP options
    use_ssl = os.environ.get('USE_SSL', 'false').lower() == 'true'
    
    if use_ssl:
        import ssl
        # Create SSL context
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        
        # Check for certificate files
        cert_dir = os.path.join(os.path.dirname(__file__), 'certs')
        os.makedirs(cert_dir, exist_ok=True)
        
        cert_file = os.path.join(cert_dir, 'cert.pem')
        key_file = os.path.join(cert_dir, 'key.pem')
        
        # Check if cert files exist, if not, generate self-signed certs
        if not (os.path.exists(cert_file) and os.path.exists(key_file)):
            print("Generating self-signed certificates for HTTPS...")
            
            import subprocess
            cmd = [
                'openssl', 'req', '-x509', '-newkey', 'rsa:4096', 
                '-nodes', '-out', cert_file, '-keyout', key_file,
                '-days', '365', '-subj', '/CN=localhost'
            ]
            try:
                subprocess.run(cmd, check=True)
                print(f"Self-signed certificates generated in {cert_dir}")
            except Exception as e:
                print(f"Error generating certificates: {e}")
                print("Falling back to HTTP mode")
                use_ssl = False
        
        if use_ssl:
            context.load_cert_chain(cert_file, key_file)
            print(f"Running Flask app with HTTPS on https://localhost:{port}")
            app.run(host='0.0.0.0', port=port, debug=True, ssl_context=context)
        else:
            print(f"Running Flask app with HTTP on http://localhost:{port}")
            app.run(host='0.0.0.0', port=port, debug=True)
    else:
        print(f"Running Flask app with HTTP on http://localhost:{port}")
        app.run(host='0.0.0.0', port=port, debug=True)
