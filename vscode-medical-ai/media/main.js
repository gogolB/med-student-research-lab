(function() {
    // Get access to the VS Code API
    const vscode = acquireVsCodeApi();
    
    // DOM elements
    const questionInput = document.getElementById('question-input');
    const askButton = document.getElementById('ask-button');
    const conversationContainer = document.getElementById('conversation');
    
    // Keep track of messages
    let messages = [];
    
    // Load previous state if any
    const previousState = vscode.getState();
    if (previousState && previousState.messages) {
        messages = previousState.messages;
        renderConversation();
    }
    
    // Request API keys from extension
    vscode.postMessage({ command: 'getApiKeys' });
    
    // Handle click on ask button
    askButton.addEventListener('click', () => {
        askQuestion();
    });
    
    // Handle pressing Enter in the input field
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            askQuestion();
        }
    });
    
    // Function to ask a question
    function askQuestion() {
        const question = questionInput.value.trim();
        if (!question) return;
        
        // Add question to the conversation
        addMessage('user', question);
        
        // Clear input
        questionInput.value = '';
        
        // Add loading indicator
        const loadingId = addLoadingIndicator();
        
        // Send question to extension
        vscode.postMessage({
            command: 'askQuestion',
            text: question
        });
    }
    
    // Function to add a message to the conversation
    function addMessage(role, content, agentInfo = null) {
        messages.push({ role, content, agentInfo });
        
        // Update VS Code state
        vscode.setState({ messages });
        
        // Update the UI
        renderConversation();
    }
    
    // Function to add a loading indicator
    function addLoadingIndicator() {
        const loadingElement = document.createElement('div');
        loadingElement.className = 'message loading';
        loadingElement.innerHTML = '<div class="dot-typing"></div>';
        loadingElement.id = 'loading-' + Date.now();
        conversationContainer.appendChild(loadingElement);
        conversationContainer.scrollTop = conversationContainer.scrollHeight;
        
        return loadingElement.id;
    }
    
    // Function to remove the loading indicator
    function removeLoadingIndicator() {
        const loadingElements = document.querySelectorAll('.loading');
        loadingElements.forEach(el => el.remove());
    }
    
    // Function to render the conversation
    function renderConversation() {
        conversationContainer.innerHTML = '';
        
        messages.forEach(message => {
            const messageElement = document.createElement('div');
            messageElement.className = `message ${message.role}`;
            
            if (message.role === 'user') {
                messageElement.innerHTML = `<div class="message-header">You</div><div class="message-content">${escapeHtml(message.content)}</div>`;
            } else if (message.role === 'assistant') {
                let agentLabel = message.agentInfo ? `AI (${message.agentInfo.name})` : 'AI';
                let content = message.content;
                
                // Handle code blocks
                if (message.agentInfo && message.agentInfo.output_type === 'code') {
                    messageElement.innerHTML = `
                        <div class="message-header">${agentLabel}</div>
                        <div class="message-content">
                            <pre><code class="language-python">${escapeHtml(content)}</code></pre>
                            <button class="insert-code-button">Insert Code to Editor</button>
                        </div>
                    `;
                    
                    // Add event listener after a small delay (to ensure the element is in the DOM)
                    setTimeout(() => {
                        const button = messageElement.querySelector('.insert-code-button');
                        if (button) {
                            button.addEventListener('click', () => {
                                vscode.postMessage({
                                    command: 'insertCode',
                                    code: content
                                });
                            });
                        }
                    }, 0);
                } else {
                    // Regular text, handle markdown formatting
                    content = formatMarkdown(content);
                    messageElement.innerHTML = `<div class="message-header">${agentLabel}</div><div class="message-content">${content}</div>`;
                }
            }
            
            conversationContainer.appendChild(messageElement);
        });
        
        // Scroll to bottom
        conversationContainer.scrollTop = conversationContainer.scrollHeight;
    }
    
    // Function to escape HTML
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    // Function to format markdown-like text
    function formatMarkdown(text) {
        // Replace line breaks with <br>
        text = text.replace(/\n/g, '<br>');
        
        // Bold text
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Italic text
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Code spans
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        return text;
    }
    
    // Handle messages from the extension
    window.addEventListener('message', event => {
        const message = event.data;
        
        switch (message.command) {
            case 'response':
                // Remove loading indicator
                removeLoadingIndicator();
                
                // Add response to conversation
                addMessage('assistant', message.response, message.agent);
                break;
            
            case 'error':
                // Remove loading indicator
                removeLoadingIndicator();
                
                // Show error message
                vscode.window.showErrorMessage(message.message);
                break;
            
            case 'setQuestion':
                // Set the question in the input field
                questionInput.value = message.text;
                
                // Focus the input field
                questionInput.focus();
                break;
            
            case 'apiKeys':
                // Store backend info
                const backendLabel = document.getElementById('backend-label');
                if (backendLabel) {
                    backendLabel.textContent = `Using ${message.backend.toUpperCase()} backend`;
                }
                break;
        }
    });

    // Set initial focus on the input field
    questionInput.focus();
})();
