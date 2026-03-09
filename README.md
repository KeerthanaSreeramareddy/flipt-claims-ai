# FliptRx AI Claims Assistant

This project is an AI-powered pharmacy claims assistant built using Python and Google Gemini.

Features:
- Uses embeddings for claim search
- Identifies paid, rejected, and duplicate claims
- Generates responses using Gemini AI
- Simple chat interface using ipywidgets

Technologies Used:
- Python
- Sentence Transformers
- Google Gemini
- NumPy
- ipywidgets
- 
## Architecture
The system follows a Retrieval-Augmented Generation (RAG) architecture:
User Query  
↓  
AI Query Assistant  
↓  
Intent Extraction using LLM  
↓  
Embedding Generation (SentenceTransformers)  
↓  
Vector Search (RAG Pipeline)  
↓  
Context Retrieval from claims.json  
↓  
LLM Reasoning (Gemini / Llama / Mistral)  
↓  
Claim classification and explanation

## Example Queries
Example 1
Show me last 2 paid claims
Example 2
Show me last 3 rejected claims

## Real World Application
This AI agent is designed for Pharmacy Benefit Managers (PBMs) to automate pharmacy claim analysis.
Instead of manually reviewing thousands of claims, the system allows administrators to ask natural language questions and receive insights such as:
- Paid claims
- Rejected claims
- Duplicate claims
- Rejection reasons
- Suggested resolution steps

## Future Improvements
Future work includes:
- Scaling the dataset to 5000+ synthetic claims
- Deploying the AI agent using AWS Bedrock
- Using models like Amazon Titan or Claude
- Ensuring HIPAA compliance for healthcare data
