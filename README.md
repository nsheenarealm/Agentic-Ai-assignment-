# 🎤 MOCKMATE – AI Mock Interview Coach

MOCKMATE is an Agentic AI-powered interview preparation platform that helps students practice company-specific interviews. It uses multiple AI agents and Retrieval-Augmented Generation (RAG) to conduct mock interviews, evaluate responses, and provide personalized feedback.

## ✨ Features

- Company and role-specific interview questions.
- AI Interviewer Agent for interactive interviews.
- AI Evaluator Agent for answer assessment.
- AI Coach Agent for personalized feedback.
- AI Report Agent for performance summaries.
- ChromaDB-powered RAG for relevant question retrieval.
- Intent detection and conversation memory.
- Downloadable PDF performance reports.

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Language:** Python
- **LLM:** Google Gemini
- **Framework:** LangChain and LCEL
- **Vector Database:** ChromaDB
- **Embeddings:** Sentence Transformers
- **Data:** CSV
- **Reports:** PDF Generation

## 🧠 Agent Workflow

1. Select a company and job role.
2. Retrieve relevant interview questions using RAG.
3. Conduct an interactive mock interview.
4. Evaluate answers based on correctness, depth, and communication.
5. Generate personalized feedback and an improvement plan.
6. Download the final performance report.

## ⚙️ Installation

```bash
git clone https://github.com/nsheenarealm/Agentic-Ai-assignment-.git
cd Agentic-Ai-assignment-
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Configure your API key in the `.env` file using `.env.example` as a reference.

## ▶️ Run the Application

```bash
streamlit run app.py
```

## 🎯 Project Objective

To make interview preparation smarter and more personalized by combining Agentic AI, Large Language Models, LangChain, and Retrieval-Augmented Generation.

## 👩‍💻 Project Details

- **Project:** MOCKMATE – Company-Smart Mock Interview Coach
- **Assignment:** Agentic AI – Assignment 1
- **Domain:** Artificial Intelligence and Natural Language Processing

---

**Built with AI to help students prepare smarter for interviews. 🚀**
