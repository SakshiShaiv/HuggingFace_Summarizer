# HuggingFace_Summarizer
An abstractive dialogue &amp; text summarization system built by fine-tuning the Hugging-Face T5 Transformer using PyTorch, with FastAPI-based inference.

## Overview
This project generates concise and meaningful summaries from everyday conversations by understanding their context and extracting key information.

## Features
- Fine-tuned T5-small Transformer model.
- Text preprocessing and tokenization.
- FastAPI backend for model inference.
- Web interface built with HTML, CSS, and JavaScript.

## Tech Stack
- **Language:** Python
- **Deep Learning:** PyTorch
- **NLP:** Hugging-Face Transformers
- **Model:** T5-small
- **Dataset:** Samsum
- **Backend:** FastAPI
- **Frontend:** HTML, CSS, JavaScript

## Project Structure
```text
├── app.py
├── index.html
├── text_summarizer.ipynb
├── samsum dataset/
├── final_model/
├── .gitignore
└── README.md
```

## Example
**Input:**

```text
Amanda: Are we meeting for lunch today?
Jerry: Yes! How about 1 PM?
Amanda: Sounds good. See you there!
```
**Output:**

```text
Amanda and Jerry agree to meet for lunch at 1 PM.
```

*Example output is illustrative; actual generated summaries may vary.*

## Future Improvements
- Improve summarization quality through further fine-tuning.
- Support longer conversations.
- Deploy the application on a cloud platform.
- Enhance the web interface.
