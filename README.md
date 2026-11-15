# 🎤 Company-Smart Mock Interview Coach (Assignment 1: Agentic AI)

Pick a company + role. Agents run a mock interview using past questions stored in ChromaDB, score every answer,
coach you with historical feedback patterns, and export a PDF report with a 5-day plan.

## Setup (Windows)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then paste your GOOGLE_API_KEY (free: aistudio.google.com/apikey)
streamlit run app.py
```
macOS/Linux: `source .venv/bin/activate` and `cp .env.example .env`.
First run downloads a small embedding model for ChromaDB (needs internet, ~80 MB, one time only).
You can also skip `.env` and paste the key in the sidebar.

Other LLMs: set `LLM_PROVIDER=groq` (install `langchain-groq`) or `ollama` (install `langchain-ollama`, run a local model).
If `gemini-2.5-flash` is retired, change `GOOGLE_MODEL` in `.env` to a current Gemini Flash model.

## IMPORTANT: data
`data/question_bank.csv` (50 questions) and `data/feedback_patterns.csv` (20 patterns) are **SAMPLE data I wrote**,
not real company records. Replace them with your institution's archive (same columns) or upload a CSV in the sidebar.
Delete the `chroma_db/` folder or press re-index after editing; the app also re-indexes automatically when the CSVs change.

## Agents
| Agent | File | Job |
|---|---|---|
| Interviewer | `Agents.next_question` | planner picks a search query, ChromaDB returns company/role questions, never repeats |
| Evaluator | `Agents.evaluate` | CoT + rubric scoring (correctness, depth, communication), calibrated with retrieved feedback |
| Coach | `Agents.coach` | retrieves feedback patterns, drafts advice, then a reflection step critiques and rewrites it |
| Report | `Agents.report` | stats in Python + LLM summary and 5-day plan, PDF + Markdown export |

## 7 acceptance criteria → where
1. LLM integration: `llm.py`  2. GUI: `app.py`  3. Prompt engineering: `prompts.py` (few-shot, ReAct-style planner, CoT, reflection; shown in the **Under the hood** tab)
4. LangChain + LCEL: `make_chain()` in `agents.py` = `prompt | llm | parser`  5. Agentic RAG: `plan_retrieval()` decides whether/what to retrieve
6. Intent routing: `detect_intent()` → PRACTICE / EVALUATE / COACH / REPORT / OTHER  7. Vector DB: `knowledge.py` (ChromaDB)

Conversation memory = `Session` in `state.py` (company, role, answers, scores, chat history passed into prompts).

## Demo script (5 min)
1. Sidebar: TCS → Software Engineer → Start new interview.
2. Answer Q1 properly, answer Q2 badly (type "idk") and compare scores.
3. Type `how should I prepare for the HR round?` → routed to COACH, see sources [P1].
4. Open **Under the hood**: show intent log, retrieval plan, prompts.
5. Click **Final report** → download PDF.

## Viva quick answers
- **Why RAG?** Answers are grounded in the institution's own questions/feedback, not generic LLM knowledge.
- **Where is it "agentic"?** The planner LLM decides whether to retrieve and what to search; the router decides which agent runs.
- **How is the answer scored?** LLM returns 0-10 for 3 dimensions; overall = 0.4 correctness + 0.3 depth + 0.3 communication (computed in code).
- **Prompt injection in answers?** Evaluator prompt treats the answer as data and ignores instructions inside it.
- **Limitations:** LLM scoring can be inconsistent; sample data; needs internet for the API and first embedding download.

## Tests
`pytest -q` runs offline tests with a fake LLM (no API key needed).
