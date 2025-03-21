import * as vscode from 'vscode';
import axios from 'axios';

export class AgentProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'medicalAIChat';
    
    private _view?: vscode.WebviewView;
    
    constructor(
        private readonly _extensionUri: vscode.Uri,
    ) {}
    
    resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;
        
        webviewView.webview.options = {
            // Enable scripts in the webview
            enableScripts: true,
            localResourceRoots: [
                this._extensionUri
            ]
        };
        
        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);
        
        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage(
            async (message) => {
                switch (message.command) {
                    case 'askQuestion':
                        this._askQuestion(message.text);
                        break;
                    case 'insertCode':
                        this._insertCodeToEditor(message.code);
                        break;
                    case 'getApiKeys':
                        this._sendApiKeysToWebview();
                        break;
                }
            }
        );
    }
    
    private async _askQuestion(question: string) {
        if (!this._view) {
            return;
        }
        
        try {
            // Send the question to the Python backend server
            const response = await axios.post('http://localhost:5000/query', {
                query: question
            });
            
            // Send the response back to the webview
            this._view.webview.postMessage({ 
                command: 'response', 
                response: response.data.response,
                agent: response.data.agent
            });
        } catch (error) {
            console.error('Error querying AI:', error);
            this._view.webview.postMessage({ 
                command: 'error', 
                message: 'Failed to get response from AI backend. Check the logs for details.' 
            });
        }
    }
    
    private _insertCodeToEditor(code: string) {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            editor.edit(editBuilder => {
                // Insert the code at the current cursor position
                const position = editor.selection.active;
                editBuilder.insert(position, code);
            });
        } else {
            // If no editor is active, create a new file
            vscode.workspace.openTextDocument({ 
                content: code, 
                language: 'python' 
            }).then(doc => {
                vscode.window.showTextDocument(doc);
            });
        }
    }
    
    private _sendApiKeysToWebview() {
        if (!this._view) {
            return;
        }
        
        const config = vscode.workspace.getConfiguration('medicalResearchAI');
        const backend = config.get<string>('backend');
        
        this._view.webview.postMessage({
            command: 'apiKeys',
            backend: backend
        });
    }
    
    private _getHtmlForWebview(webview: vscode.Webview) {
        // Get styles
        const styleUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'style.css')
        );
        
        // Get script
        const scriptUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'main.js')
        );
        
        // Use a nonce to only allow specific scripts to be run
        const nonce = getNonce();
        
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
            <link href="${styleUri}" rel="stylesheet">
            <title>Medical AI Assistant</title>
        </head>
        <body>
            <h1>Medical Research AI</h1>
            <p>Ask any medical research question.</p>
            
            <div class="input-container">
                <input type="text" id="question-input" placeholder="Ask your question...">
                <button id="ask-button">Ask</button>
            </div>
            
            <div class="conversation-container">
                <div id="conversation"></div>
            </div>
            
            <div id="backend-info">
                <span id="backend-label"></span>
            </div>
            
            <script nonce="${nonce}" src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}

function getNonce() {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
