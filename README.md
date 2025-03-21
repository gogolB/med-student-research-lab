# MedResearch AI Assistant

An AI-powered research assistant that helps medical students conduct data science projects through a team of specialized AI agents.

## Features

- Chat-based interface for interacting with AI agents
- Multiple specialized agents (statistics, data analysis, visualization, etc.)
- Tool integration (web search, document processing, command line)
- Conversation memory for continuous project assistance
- Support for data upload and processing

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys (see `.env.example`)
4. Run the application: `python main.py`

## Usage

1. Start a new project with `/new project`
2. Upload data with `/upload`
3. Describe your research goals and ask questions
4. The AI agent team will guide you through your data science project

## Overview

This framework provides medical students with AI-powered assistance for designing and conducting research projects. It features multiple specialized agents with expertise in different aspects of medical research, from study design to statistical analysis.

## Features

- **Multi-agent system**: Access to specialized expertise in research methodology, statistics, and clinical domains
- **Research workflow support**: Guidance through each step of the research process
- **Interactive interface**: Simple command-line interface for querying the system
- **Collaborative agents**: Agents can work together to solve complex research problems
- **Metrics and logging**: Integrated with SignOz for monitoring and performance tracking

## Getting Started

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd research-ai

# Install dependencies
pip install -r requirements.txt
```

### Docker Deployment

```bash
# Build and start services with Docker Compose
docker-compose up -d

# Access the interactive CLI within the container
docker exec -it research-ai python -m src.ui.cli interactive

# Or use the CLI directly
docker exec -it research-ai python -m src.ui.cli query "How should I design a cohort study for diabetes patients?"
```

### SignOz Monitoring

The application is integrated with SignOz for metrics and logging:

- SignOz Dashboard: http://localhost:3301
- Metrics, traces, and logs are automatically collected
- View performance metrics and debug issues using the SignOz interface

### Running the Application Locally

```bash
# Start an interactive session
python -m src.ui.cli interactive

# Or ask a specific question
python -m src.ui.cli query "How should I design a cohort study for diabetes patients?"

# Get help with statistical analysis
python -m src.ui.cli stats "Which test should I use to compare treatment outcomes?"
```

## Available Agents

1. **Research Agent**: Specializes in research design, methodology, and research planning
2. **Statistician Agent**: Provides expertise in statistical analysis, sample size calculations, and result interpretation

## Example Usage

```bash
# Design a study
python -m src.ui.cli design "How does medication adherence affect blood pressure control?" --domain "cardiology"

# Get statistical advice
python -m src.ui.cli stats "How do I analyze survey data?" --data-type "categorical" --outcome "binary"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
