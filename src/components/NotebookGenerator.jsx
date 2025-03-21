import React, { useState } from 'react';
import './NotebookGenerator.css';

function NotebookGenerator() {
  const [researchQuestion, setResearchQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/generate-notebook', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ researchQuestion }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate notebook');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'research_notebook.ipynb';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to generate notebook. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="notebook-generator">
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="research-question">Research Question:</label>
          <textarea
            id="research-question"
            value={researchQuestion}
            onChange={(e) => setResearchQuestion(e.target.value)}
            placeholder="Enter your research question here..."
            required
          />
        </div>
        <button type="submit" disabled={loading}>
          {loading ? 'Generating...' : 'Generate Notebook'}
        </button>
      </form>
      {error && <div className="error-message">{error}</div>}
    </div>
  );
}

export default NotebookGenerator; 