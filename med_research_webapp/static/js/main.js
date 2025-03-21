document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const conversationContainer = document.getElementById('conversation-container');
    const queryInput = document.getElementById('query-input');
    const submitButton = document.getElementById('submit-button');
    const backendSelect = document.getElementById('backend-select');
    const apiKeyInput = document.getElementById('api-key');
    const modelInput = document.getElementById('model');
    const updateBackendButton = document.getElementById('update-backend');
    const backendLabel = document.getElementById('backend-label');
    const exampleQuestions = document.querySelectorAll('.example-question');
    const codeResultModal = new bootstrap.Modal(document.getElementById('codeResultModal'));
    const codeResult = document.getElementById('code-result');
    
    // Get backend info on page load
    fetchBackendInfo();
    
    // Event listeners
    submitButton.addEventListener('click', handleSubmitQuery);
    queryInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmitQuery();
        }
    });
    
    updateBackendButton.addEventListener('click', updateBackendSettings);
    
    // Example questions click event
    exampleQuestions.forEach(question => {
        question.addEventListener('click', function(e) {
            e.preventDefault();
            queryInput.value = this.textContent;
            queryInput.focus();
        });
    });
    
    // Function to handle query submission
    function handleSubmitQuery() {
        const query = queryInput.value.trim();
        if (!query) return;
        
        // Add user message to conversation
        addMessage('user', query);
        
        // Clear input
        queryInput.value = '';
        
        // Show loading indicator
        const loadingIndicator = addLoadingIndicator();
        
        // Send query to API
        fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading indicator
            if (loadingIndicator) loadingIndicator.remove();
            
            if (data.error) {
                addMessage('error', `Error: ${data.error}`);
                return;
            }
            
            // Add AI response to conversation
            addMessage('assistant', data.response, data.agent);
            
            // Scroll to bottom
            scrollToBottom();
        })
        .catch(error => {
            // Remove loading indicator
            if (loadingIndicator) loadingIndicator.remove();
            
            // Show error
            addMessage('error', `Network error: ${error.message}`);
            console.error('Error:', error);
        });
    }
    
    // Function to add a message to the conversation
    function addMessage(role, content, agentInfo = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        let messageHTML = '';
        
        if (role === 'user') {
            messageHTML = `
                <div class="message-header">You</div>
                <div class="message-content">${escapeHTML(content)}</div>
            `;
        } else if (role === 'assistant') {
            // Process content based on agent type
            const isCode = agentInfo && agentInfo.output_type === 'code';
            const processedContent = isCode 
                ? formatCodeBlock(content)
                : marked.parse(content);
            
            messageHTML = `
                <div class="message-header">AI Assistant</div>
                <div class="message-content">${processedContent}</div>
            `;
            
            // Add agent info if available
            if (agentInfo) {
                messageHTML += `
                    <div class="agent-info">Answered by ${agentInfo.name} - ${agentInfo.description}</div>
                `;
                
                // Add execute button for code
                if (isCode) {
                    messageHTML += `
                        <div class="code-actions">
                            <button class="btn btn-sm btn-outline-primary execute-code-btn">Execute Code</button>
                            <button class="btn btn-sm btn-outline-secondary copy-code-btn">Copy Code</button>
                        </div>
                    `;
                }
            }
        } else if (role === 'error') {
            messageHTML = `
                <div class="message-header">Error</div>
                <div class="message-content">${escapeHTML(content)}</div>
            `;
        } else {
            // System message
            messageHTML = `
                <div class="message-content">${content}</div>
            `;
        }
        
        messageDiv.innerHTML = messageHTML;
        conversationContainer.appendChild(messageDiv);
        
        // Add event listeners for code buttons if they exist
        if (role === 'assistant' && agentInfo && agentInfo.output_type === 'code') {
            const executeBtn = messageDiv.querySelector('.execute-code-btn');
            const copyBtn = messageDiv.querySelector('.copy-code-btn');
            
            if (executeBtn) {
                executeBtn.addEventListener('click', function() {
                    executeCode(content);
                });
            }
            
            if (copyBtn) {
                copyBtn.addEventListener('click', function() {
                    navigator.clipboard.writeText(content)
                        .then(() => {
                            copyBtn.textContent = 'Copied!';
                            setTimeout(() => {
                                copyBtn.textContent = 'Copy Code';
                            }, 2000);
                        })
                        .catch(err => {
                            console.error('Error copying text:', err);
                        });
                });
            }
        }
        
        // Initialize syntax highlighting for code blocks
        if (role === 'assistant') {
            document.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }
        
        // Scroll to bottom
        scrollToBottom();
        
        return messageDiv;
    }
    
    // Function to add a loading indicator
    function addLoadingIndicator() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading-indicator';
        loadingDiv.innerHTML = '<div class="dot-typing"></div>';
        conversationContainer.appendChild(loadingDiv);
        
        // Scroll to bottom
        scrollToBottom();
        
        return loadingDiv;
    }
    
    // Function to format code blocks
    function formatCodeBlock(code) {
        return `<pre><code class="language-python">${escapeHTML(code)}</code></pre>`;
    }
    
    // Function to escape HTML
    function escapeHTML(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // Function to scroll conversation to bottom
    function scrollToBottom() {
        conversationContainer.scrollTop = conversationContainer.scrollHeight;
    }
    
    // Function to execute code
    function executeCode(code) {
        // Show loading in the modal
        codeResult.textContent = 'Executing code...';
        codeResultModal.show();
        
        // Send code to API
        fetch('/api/execute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ code: code })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                codeResult.textContent = `Error: ${data.error}`;
                return;
            }
            
            // Format and display the result
            let result = '';
            
            if (data.stdout) {
                result += data.stdout;
            }
            
            if (data.stderr) {
                result += `\n\nErrors/Warnings:\n${data.stderr}`;
            }
            
            if (data.returncode !== 0) {
                result += `\n\nProcess exited with code ${data.returncode}`;
            }
            
            codeResult.textContent = result || 'Code executed successfully with no output.';
        })
        .catch(error => {
            codeResult.textContent = `Error executing code: ${error.message}`;
            console.error('Error:', error);
        });
    }
    
    // Function to fetch backend info
    function fetchBackendInfo() {
        fetch('/api/backend')
        .then(response => response.json())
        .then(data => {
            backendLabel.textContent = data.backend.toUpperCase();
            backendSelect.value = data.backend;
            modelInput.placeholder = `Current: ${data.model}`;
        })
        .catch(error => {
            console.error('Error fetching backend info:', error);
        });
    }
    
    // Function to update backend settings
    function updateBackendSettings() {
        const backend = backendSelect.value;
        const apiKey = apiKeyInput.value;
        const model = modelInput.value;
        
        // Show loading
        updateBackendButton.disabled = true;
        updateBackendButton.textContent = 'Updating...';
        
        // Send settings to API
        fetch('/api/backend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                backend: backend,
                api_key: apiKey,
                model: model
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                addMessage('error', `Error updating backend: ${data.error}`);
                return;
            }
            
            // Update UI
            backendLabel.textContent = data.backend.toUpperCase();
            modelInput.placeholder = `Current: ${data.model}`;
            
            // Clear inputs
            apiKeyInput.value = '';
            modelInput.value = '';
            
            // Show success message
            addMessage('system', `Backend updated to ${data.backend.toUpperCase()} successfully.`);
        })
        .catch(error => {
            // Show error
            addMessage('error', `Error updating backend: ${error.message}`);
            console.error('Error:', error);
        })
        .finally(() => {
            // Re-enable button
            updateBackendButton.disabled = false;
            updateBackendButton.textContent = 'Update Settings';
        });
    }
});
