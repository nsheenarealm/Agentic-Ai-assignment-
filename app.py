"""Company-Smart Mock Interview Coach: Streamlit UI.   Run:  streamlit run app.py"""
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from knowledge import DATA_DIR, KnowledgeBase          # noqa: E402
from llm import LLMConfigError, get_llm                # noqa: E402
from orchestrator import Orchestrator                   # noqa: E402
from prompts import PROMPTS, show                       # noqa: E402
from report_pdf import report_markdown, report_pdf     # noqa: E402
from state import Session                               # noqa: E402

st.set_page_config(page_title="Mock Interview Coach", page_icon="🎤", layout="wide")


@st.cache_resource(show_spinner="Indexing question bank in ChromaDB (first run downloads a small embedding model)...")
def load_kb():
    return KnowledgeBase()


kb = load_kb()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🎤 Interview Coach")
    st.caption("Company-smart mock interviews grounded in past questions and feedback.")

    provider = st.selectbox("LLM provider", ["gemini", "groq", "ollama"],
                            index=["gemini", "groq", "ollama"].index(os.getenv("LLM_PROVIDER", "gemini")))
    key_env = {"gemini": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"}.get(provider)
    api_key = None
    if key_env and not os.getenv(key_env):
        api_key = st.text_input(f"{key_env}", type="password", help="Or put it in your .env file")

    st.divider()
    companies = kb.companies()
    company = st.selectbox("Target company", companies, index=companies.index("TCS") if "TCS" in companies else 0)
    role = st.selectbox("Role", kb.roles())
    difficulty = st.selectbox("Difficulty", ["Mixed", "Easy", "Medium", "Hard"])
    total = st.slider("Number of questions", 3, 10, 5)
    start = st.button("▶ Start new interview", type="primary", use_container_width=True)

    st.divider()
    with st.expander("📥 Load your own question bank"):
        st.caption("CSV columns: id, company, role, round_type, difficulty, question, expected_points (separate points with ;), source_year")
        up = st.file_uploader("question_bank CSV", type="csv", key="qb")
        if up and st.button("Add & re-index"):
            (DATA_DIR / "question_bank_uploaded.csv").write_bytes(up.getvalue())
            kb.ensure_index(force=True)
            st.success("Re-indexed. Pick the company from the dropdown above.")
            st.rerun()
    st.caption(f"Knowledge base: {len(kb.questions)} questions · {len(kb.patterns)} feedback patterns")
    if all(r.get("source_year") == "sample" for r in kb.questions[:50]):
        st.warning("Using SAMPLE data. Replace data/*.csv with your institution's archive.", icon="⚠️")

# ------------------------------------------------------------------ LLM + orchestrator
try:
    llm = get_llm(provider=provider, api_key=api_key)
except LLMConfigError as e:
    st.info(f"👈 {e}")
    st.stop()
except Exception as e:
    st.error(f"Could not start the LLM: {e}")
    st.stop()

orch = Orchestrator(llm, kb)

if "session" not in st.session_state:
    st.session_state.session = Session()
session: Session = st.session_state.session

if start:
    st.session_state.session = session = Session(company=company, role=role, difficulty=difficulty, total_questions=total)
    with st.spinner("Preparing your first question..."):
        orch.handle(session, "Start the interview", forced_intent="PRACTICE")
    st.rerun()

# ------------------------------------------------------------------ tabs
tab_chat, tab_report, tab_hood = st.tabs(["💬 Interview", "📄 Report", "🔍 Under the hood"])

with tab_chat:
    if not session.messages:
        st.markdown("### Welcome 👋\nChoose a company and role in the sidebar, then press **Start new interview**.")
    st.caption(f"{session.company} · {session.role} · {session.difficulty} · "
               f"{len(session.turns)}/{session.total_questions} answered")
    for m in session.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    c1, c2, c3 = st.columns(3)
    action = None
    if c1.button("⏭ Skip / Next question", use_container_width=True, disabled=not session.messages):
        action = ("Next question please", "PRACTICE")
    if c2.button("🎯 Coach me", use_container_width=True, disabled=not session.messages):
        action = ("Coach me on how to improve for this company", "COACH")
    if c3.button("📄 Final report", use_container_width=True, disabled=not session.turns):
        action = ("Give me my final report", "REPORT")
    typed = st.chat_input("Type your answer, or ask for coaching / next question / report...",
                          disabled=not session.messages)
    if typed:
        action = (typed, None)
    if action:
        with st.spinner("Agents are working..."):
            orch.handle(session, action[0], forced_intent=action[1])
        st.rerun()

with tab_report:
    if not session.report:
        st.info("Finish a few questions, then click **Final report** (or type *report*).")
    else:
        r = session.report
        s = r["stats"]
        m = st.columns(4)
        m[0].metric("Overall", f"{s['overall']}/10")
        m[1].metric("Correctness", s["correctness"])
        m[2].metric("Depth", s["depth"])
        m[3].metric("Communication", s["communication"])
        st.markdown(report_markdown(r))
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download PDF", report_pdf(r), file_name="mock_interview_report.pdf", mime="application/pdf")
        d2.download_button("⬇ Download Markdown", report_markdown(r), file_name="mock_interview_report.md")

with tab_hood:
    st.subheader("Acceptance criteria map")
    st.markdown("""
| # | Criterion | Where |
|---|---|---|
| 1 | LLM integration | `llm.py` → Gemini / Groq / Ollama |
| 2 | Interactive GUI | `app.py` (Streamlit) |
| 3 | Prompt engineering | `prompts.py` → few-shot, ReAct-style planner, CoT, reflection |
| 4 | LangChain + LCEL | `agents.py` → `prompt \\| llm \\| parser` chains |
| 5 | Agentic RAG | `Agents.plan_retrieval()` decides when/what to retrieve |
| 6 | Intent routing | `Agents.detect_intent()` → PRACTICE / EVALUATE / COACH / REPORT |
| 7 | Vector DB | `knowledge.py` → ChromaDB similarity search |
""")
    st.subheader("Live trace (this session)")
    if not session.trace:
        st.caption("Nothing yet.")
    for t in reversed(session.trace[-30:]):
        st.json(t, expanded=False)
    st.subheader("Prompts (system + user)")
    for k in PROMPTS:
        p = show(k)
        with st.expander(f"{k} · {p['technique']}"):
            st.markdown("**System prompt**")
            st.code(p["system"], language="text")
            st.markdown("**User prompt template**")
            st.code(p["user"], language="text")
