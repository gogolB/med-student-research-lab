import * as vscode from 'vscode';
import axios from 'axios';

export class MedicalAIPanel {
    public static currentPanel: MedicalAIPanel | undefined;
    private static readonly viewType = 'medicalAIChat';
    private static readonly title = 'Medical Research AI Assistant';
    
    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _disposables: vscode.Disposable[] = [];
    
    private static _onDidCreatePanel = new vscode.EventEmitter<MedicalAIPanel>();
    private static _onDidDisposePanel = new vscode.EventEmitter<void>();
    
    public static readonly onDidCreatePanel = MedicalAIPanel._onDidCreatePanel.event;
    public static readonly onDidDisposePanel = MedicalAIPanel._onDidDisposePanel.event;
    
    public static createOrShow(extensionUri: vscode.Uri) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;
            
        // If we already have a panel, show it
        if (MedicalAIPanel.currentPanel) {
            MedicalAIPanel.currentPanel._panel.reveal(column);
            return;
        }
        
        // Otherwise, create a new panel
        const panel = vscode.window.createWebviewPanel(
            MedicalAIPanel.viewType,
            MedicalAIPanel.title,
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [
                    vscode.Uri.joinPath(extensionUri, 'media'),
                    vscode.Uri.joinPath(extensionUri, 'dist')
                ]
            }
        );
        
        MedicalAIPanel.currentPanel = new MedicalAIPanel(panel, extensionUri);
        MedicalAIPanel._onDidCreatePanel.fire(MedicalAIPanel.currentPanel);
    }
    
    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
        this._panel = panel;
        this._extensionUri = extensionUri;
        
        // Set the webview's initial html content
        this._update();
        
        // Listen for when the panel is disposed
        // This happens when the user closes the panel or when the panel is closed programmatically
        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
        
        // Handle messages from the webview
        this._panel.webview.onDidReceiveMessage(
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
            },
            null,
            this._disposables
        );
    }
    
    public sendQuestion(question: string) {
        this._panel.webview.postMessage({ 
            command: 'setQuestion', 
            text: question 
        });
    }
    
    public dispose() {
        MedicalAIPanel.currentPanel = undefined;
        MedicalAIPanel._onDidDisposePanel.fire();
        
        // Clean up resources
        this._panel.dispose();
        
        while (this._disposables.length) {
            const x = this._disposables.pop();
            if (x) {
                x.dispose();
            }
        }
    }
    
    private async _askQuestion(question: string) {
        try {
            // Send questions to the Python backend server
            const response = await axios.post('http://localhost:5000/query', {
                query: question
            });
            
            // Send the response back to the webview
            this._panel.webview.postMessage({ 
                command: 'response', 
                response: response.data.response,
                agent: response.data.agent
            });
        } catch (error) {
            console.error('Error querying AI:', error);
            this._panel.webview.postMessage({ 
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
        const config = vscode.workspace.getConfiguration('medicalResearchAI');
        const backend = config.get<string>('backend');
        
        this._panel.webview.postMessage({
            command: 'apiKeys',
            backend: backend
        });
    }
    
    private _update() {
        const webview = this._panel.webview;
        
        // Set the HTML content
        webview.html = this._getHtmlForWebview(webview);
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
            <title>Medical Research AI Assistant</title>
        </head>
        <body>
            <h1>Medical Research AI Assistant</h1>
            <p>Ask any medical research question, and I'll route it to the right specialist.</p>
            
            <div class="input-container">
                <input type="text" id="question-input" placeholder="Ask your medical research question...">
                <button id="ask-button">Ask</button>
            </div>
            
            <div class="conversation-container">
                <div id="conversation"></div>
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
