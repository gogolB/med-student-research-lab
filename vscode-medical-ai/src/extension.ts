import * as vscode from 'vscode';
import * as path from 'path';
import * as child_process from 'child_process';
import { AgentProvider } from './sidebarProvider';
import { MedicalAIPanel } from './panel';

let activePanel: MedicalAIPanel | undefined;
let pythonProcess: child_process.ChildProcess | null = null;

export function activate(context: vscode.ExtensionContext) {
    console.log('Medical Research AI Assistant is now active');
    
    // Register the sidebar view provider
    const agentProvider = new AgentProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('medicalAIChat', agentProvider)
    );
    
    // Register command to start the AI assistant
    let startAgent = vscode.commands.registerCommand('medicalResearchAI.startAgent', () => {
        MedicalAIPanel.createOrShow(context.extensionUri);
        
        // Start Python backend if not already running
        startPythonBackend(context);
    });
    
    // Register command to ask a question directly
    let askQuestion = vscode.commands.registerCommand('medicalResearchAI.askQuestion', async () => {
        // Ensure the Python backend is running
        if (!pythonProcess) {
            startPythonBackend(context);
        }
        
        // Prompt user for question
        const question = await vscode.window.showInputBox({
            prompt: 'Ask your medical research question',
            placeHolder: 'e.g., How does the Mediterranean diet affect cardiovascular health?'
        });
        
        if (question) {
            // Open panel if not already open
            if (!activePanel) {
                MedicalAIPanel.createOrShow(context.extensionUri);
            }
            
            // Send question to panel
            if (activePanel) {
                activePanel.sendQuestion(question);
            }
        }
    });
    
    context.subscriptions.push(startAgent, askQuestion);
    
    // Store active panel reference when created
    MedicalAIPanel.onDidCreatePanel((panel) => {
        activePanel = panel;
    });
    
    // Remove active panel reference when disposed
    MedicalAIPanel.onDidDisposePanel(() => {
        activePanel = undefined;
    });
}

function startPythonBackend(context: vscode.ExtensionContext) {
    // Get Python executable path from configuration
    const config = vscode.workspace.getConfiguration('medicalResearchAI');
    const pythonPath = config.get<string>('pythonPath') || 'python';
    
    // Path to the Python server script
    const scriptPath = path.join(context.extensionPath, 'python', 'server.py');
    
    // Kill any existing Python process
    if (pythonProcess) {
        pythonProcess.kill();
    }
    
    // Start the Python backend server
    pythonProcess = child_process.spawn(pythonPath, [scriptPath]);
    
    // Log output from Python process
    pythonProcess.stdout?.on('data', (data) => {
        console.log(`Python Backend: ${data}`);
    });
    
    pythonProcess.stderr?.on('data', (data) => {
        console.error(`Python Backend Error: ${data}`);
    });
    
    pythonProcess.on('close', (code) => {
        console.log(`Python Backend process exited with code ${code}`);
        pythonProcess = null;
    });
}

export function deactivate() {
    // Kill Python process when extension is deactivated
    if (pythonProcess) {
        pythonProcess.kill();
        pythonProcess = null;
    }
}
