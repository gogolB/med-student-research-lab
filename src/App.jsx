import React from 'react';
import NotebookGenerator from './components/NotebookGenerator';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Research Notebook Generator</h1>
      </header>
      <main>
        <NotebookGenerator />
      </main>
    </div>
  );
}

export default App; 