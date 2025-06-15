---
title: GAIA Agent Project
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.34.0
app_file: app.py
pinned: false
---

# GAIA Agent Project

AI agent for the GAIA benchmark, built for the Hugging Face Agents Course Certificate of Excellence.

## Overview

This project implements an AI agent that can solve tasks from the GAIA (General AI Assistants) benchmark. The agent uses xAI's Grok API for reasoning and includes tools for web search, file handling, and mathematical calculations.

## Goal

Achieve ≥30% score on the GAIA benchmark to earn the Certificate of Excellence from the Hugging Face Agents Course.

## Project Structure

```
├── agent.py          # Main GAIA agent implementation
├── tools.py          # Tool implementations (web search, file handling)
├── evaluate.py       # Evaluation script and scoring
├── test_agent.py     # Test suite for verification
├── requirements.txt  # Python dependencies
├── README.md         # This file
├── .gitignore        # Git ignore rules
└── submission.jsonl  # Generated submission file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. API Configuration

The agent uses xAI's Grok API. The API key is already configured in the code for this project.

### 3. Optional: SerpAPI for Enhanced Web Search

For better web search results, you can sign up for SerpAPI:
1. Visit https://serpapi.com/ and create an account
2. Get your API key
3. Update the `serpapi_key` in `agent.py`

## Usage

### Quick Test

Run the test suite to verify everything is working:

```bash
python test_agent.py
```

### Full Evaluation

Run the full evaluation on sample tasks:

```bash
python evaluate.py
```

Run with maximum number of tasks limit:

```bash
python evaluate.py --max-tasks 10
```

Run with custom dataset:

```bash
python evaluate.py --dataset path/to/gaia_dataset.jsonl
```

## Components

### Agent (`agent.py`)

- **GAIAAgent**: Main agent class that processes GAIA tasks
- **call_grok()**: Interface to xAI Grok API with retry logic
- **process_task()**: Main task processing pipeline
- **extract_final_answer()**: Extracts formatted answers from responses

### Tools (`tools.py`)

- **web_search()**: Web search with SerpAPI fallback to DuckDuckGo
- **read_file()**: Handles text, CSV, and image files
- **execute_code()**: Safe Python code execution (limited)
- **calculate_simple_math()**: Basic mathematical calculations

### Evaluation (`evaluate.py`)

- **evaluate_agent()**: Main evaluation function
- **load_gaia_dataset()**: Loads GAIA dataset from JSON/JSONL
- **normalize_answer()**: Normalizes answers for comparison
- **create_sample_dataset()**: Creates sample tasks for testing

## Features

- ✅ xAI Grok API integration with retry logic
- ✅ Web search capabilities (SerpAPI + DuckDuckGo fallback)
- ✅ Multi-format file handling (text, CSV, images)
- ✅ OCR support for image-based tasks (with pytesseract)
- ✅ Safe code execution environment
- ✅ Comprehensive evaluation system
- ✅ JSONL submission format generation
- ✅ Progress tracking and scoring

## GAIA Task Types

The agent handles different GAIA task levels:

- **Level 1**: Simple questions requiring basic knowledge
- **Level 2**: Multi-step reasoning tasks
- **Level 3**: Complex tasks involving files, images, or code

## Sample Tasks

The evaluation includes sample tasks like:

- Basic arithmetic: "What is 15 + 27?"
- General knowledge: "What is the capital of France?"
- Date calculations: "How many days are in a leap year?"
- Multi-step math: "What is 2 * 6 * 7?"
- Historical facts: "What year did World War II end?"

## Scoring

- Target: ≥30% accuracy for Certificate of Excellence
- Current leaderboard top score: ~76%
- Evaluation provides detailed per-task feedback
- Generates `submission.jsonl` in required format

## Troubleshooting

### API Issues
- Verify internet connection
- Check API key validity
- Monitor rate limits

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- For OCR: Install system dependency `tesseract-ocr`

### File Reading Issues
- Check file paths and permissions
- Verify file formats are supported

## Development

### Testing
Run the test suite before making changes:
```bash
python test_agent.py
```

### Adding New Tools
1. Implement the tool function in `tools.py`
2. Import and use in `agent.py`
3. Add tests in `test_agent.py`

### Improving Performance
- Optimize prompts for better reasoning
- Add more sophisticated web search
- Enhance file processing capabilities
- Implement better answer extraction

## Submission

1. Run evaluation: `python evaluate.py`
2. Upload `submission.jsonl` to the Hugging Face leaderboard
3. Verify score ≥30% for certificate eligibility

## Resources

- [GAIA Benchmark](https://github.com/gaia-benchmark/GAIA)
- [xAI API Documentation](https://x.ai/api)
- [Hugging Face Agents Course](https://huggingface.co/docs)
- [SerpAPI](https://serpapi.com/)

## License

This project is created for educational purposes as part of the Hugging Face Agents Course.

---

**Good luck achieving the 30% score for your Certificate of Excellence! 🎉**