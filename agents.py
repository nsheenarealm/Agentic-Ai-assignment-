"""The four agents (Interviewer, Evaluator, Coach, Report) + router, all built with LangChain LCEL."""
import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from prompts import PROMPTS

ROUND_CYCLE = ["Technical", "Technical", "HR", "Coding", "Managerial", "Technical", "Aptitude", "HR"]


class AgentError(Exception):
    pass


def _msg_to_text(msg) -> str:
    """Chat model output -> plain text (handles Gemini's list-of-blocks content)."""
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else p.get("text", "") for p in content
                       if isinstance(p, str) or (isinstance(p, dict) and p.get("type", "text") == "text"))
    return str(content)


def make_chain(llm, key: str):
    """LCEL: prompt template | model | text extraction | string parser."""
    prompt = ChatPromptTemplate.from_messages([("system", PROMPTS[key]["system"]), ("human", PROMPTS[key]["user"])])
    return prompt | llm | RunnableLambda(_msg_to_text) | StrOutputParser()


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise AgentError("Model did not return JSON.")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise AgentError(f"Invalid JSON from model: {e}")


def _clamp(v, lo=0, hi=10) -> int:
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return 0


def _bullets(items, n=6):
    return [str(i) for i in (items or [])][:n]


def format_patterns(patterns) -> str:
    if not patterns:
        return "(none retrieved)"
    return "\n".join(f"[P{i}] ({p['company']}, {p['theme']}) {p['pattern']}" for i, p in enumerate(patterns, 1))


class Agents:
    def __init__(self, llm, kb):
        self.llm, self.kb = llm, kb
        self.chains = {k: make_chain(llm, k) for k in PROMPTS}

    # ------------------------------------------------------------------ router
    def detect_intent(self, session, message: str):
        """Returns (intent, method). Long answers shortcut to EVALUATE, else LLM few-shot router with keyword fallback."""
        words = len(message.split())
        if session.pending and words >= 25:
            return "EVALUATE", "rule: long text while a question is pending"
        try:
            out = self.chains["router"].invoke({
                "company": session.company, "role": session.role,
                "pending": session.pending["question"] if session.pending else "none",
                "chat_history": session.chat_history(), "message": message})
            m = re.search(r"\b(PRACTICE|EVALUATE|COACH|REPORT|OTHER)\b", out.upper())
            if m:
                return m.group(1), "llm: few-shot router"
        except Exception:
            pass
        return self._keyword_intent(session, message), "fallback: keyword rules"

    @staticmethod
    def _keyword_intent(session, message: str) -> str:
        t = message.lower()
        if re.search(r"\b(report|summary|end (the )?interview|finish|wrap up|final score)\b", t):
            return "REPORT"
        if re.search(r"\b(coach|coaching|tips?|advice|feedback|improve|prepare|how (do|should|can) i|mistakes)\b", t) \
                and len(t.split()) <= 25:
            return "COACH"
        if re.search(r"\b(start|begin|next|another|skip|ask me|practice|mock|repeat)\b", t) and len(t.split()) <= 8:
            return "PRACTICE"
        return "EVALUATE" if session.pending else "OTHER"

    # ------------------------------------------------------------------ planner (agentic RAG decision)
    def plan_retrieval(self, session, agent: str, task: str, message: str, round_hint: str = "n/a") -> dict:
        default = {"retrieve": True, "source": "questions" if agent == "Interviewer" else "feedback",
                   "query": f"{session.role} {round_hint if round_hint != 'n/a' else message}"[:80],
                   "reason": "default plan (planner output unavailable)"}
        try:
            out = self.chains["planner"].invoke({
                "agent": agent, "task": task, "company": session.company, "role": session.role,
                "message": message, "performance": session.performance_summary(),
                "asked": session.topics_asked(), "round_hint": round_hint})
            plan = parse_json(out)
            plan["retrieve"] = bool(plan.get("retrieve", True))
            plan["source"] = plan.get("source") if plan.get("source") in ("questions", "feedback") else default["source"]
            plan["query"] = str(plan.get("query") or default["query"])[:100]
            plan["reason"] = str(plan.get("reason", ""))
        except Exception:
            plan = default
        if agent == "Interviewer":
            plan["retrieve"], plan["source"] = True, "questions"   # an interviewer always needs a question
        session.log("retrieval_plan", agent=agent, **plan)
        return plan

    # ------------------------------------------------------------------ Interviewer
    def next_question(self, session):
        """Pick the next question from ChromaDB. Returns question dict or None if exhausted."""
        idx = len(session.turns)
        hint = ROUND_CYCLE[idx % len(ROUND_CYCLE)]
        plan = self.plan_retrieval(session, "Interviewer", "Choose a search query for the next interview question",
                                   "next question", round_hint=hint)
        for diff in (session.difficulty, "Mixed"):
            cands = self.kb.search_questions(session.company, session.role, plan["query"], n=8,
                                             difficulty=diff, exclude_ids=session.asked_ids)
            if cands:
                break
        if not cands:
            return None
        cands.sort(key=lambda c: (c["round_type"] != hint,))   # stable: prefer the suggested round, keep similarity order
        q = cands[0]
        session.log("retrieved_questions", query=plan["query"],
                    results=[f"{c['company']} | {c['round_type']} | {c['question'][:60]}" for c in cands[:5]], picked=q["id"])
        session.pending = q
        session.asked_ids.append(q["id"])
        return q

    # ------------------------------------------------------------------ Evaluator
    def evaluate(self, session, answer: str) -> dict:
        q = session.pending
        patterns = self.kb.search_feedback(session.company, session.role, q["question"], n=2)
        session.log("retrieved_feedback", agent="Evaluator", query=q["question"][:60],
                    results=[f"{p['company']} | {p['theme']}" for p in patterns])
        vars_ = {"company": session.company, "role": session.role, "round_type": q["round_type"],
                 "difficulty": q["difficulty"], "question": q["question"],
                 "expected_points": "; ".join(q["expected_points"]), "patterns": format_patterns(patterns),
                 "answer": answer}
        last_err = None
        for _ in range(2):
            try:
                data = parse_json(self.chains["evaluator"].invoke(vars_))
                break
            except AgentError as e:
                last_err = e
        else:
            raise AgentError(f"Could not read the evaluation ({last_err}). Please send your answer again.")
        c, d, m = _clamp(data.get("correctness")), _clamp(data.get("depth")), _clamp(data.get("communication"))
        return {"reasoning": str(data.get("reasoning", "")), "correctness": c, "depth": d, "communication": m,
                "overall": round(0.4 * c + 0.3 * d + 0.3 * m, 1),
                "covered_points": _bullets(data.get("covered_points")), "missed_points": _bullets(data.get("missed_points")),
                "strengths": _bullets(data.get("strengths"), 3), "improvements": _bullets(data.get("improvements"), 3),
                "follow_up": str(data.get("follow_up", "")), "patterns": patterns}

    # ------------------------------------------------------------------ Coach (RAG + reflection)
    def coach(self, session, message: str):
        plan = self.plan_retrieval(session, "Coach", "Give personalised coaching grounded in past feedback", message)
        patterns = []
        if plan["retrieve"]:
            patterns = self.kb.search_feedback(session.company, session.role, plan["query"], n=4)
            session.log("retrieved_feedback", agent="Coach", query=plan["query"],
                        results=[f"{p['company']} | {p['theme']}" for p in patterns])
        pats = format_patterns(patterns)
        draft = self.chains["coach_draft"].invoke({
            "company": session.company, "role": session.role, "message": message,
            "performance": session.performance_summary(), "patterns": pats, "chat_history": session.chat_history()})
        final = draft
        try:
            refl = self.chains["coach_reflect"].invoke({"message": message, "patterns": pats, "draft": draft})
            if "FINAL:" in refl:
                final = refl.split("FINAL:", 1)[1].strip() or draft
                session.log("reflection", critique=refl.split("FINAL:", 1)[0].replace("CRITIQUE:", "").strip()[:600])
        except Exception:
            pass
        return final, patterns

    # ------------------------------------------------------------------ Report
    @staticmethod
    def compute_stats(session) -> dict:
        done = [t.evaluation for t in session.turns if t.evaluation]
        avg = lambda k: round(sum(e[k] for e in done) / len(done), 1) if done else 0.0
        stats = {"questions_asked": len(session.turns), "answered": len(done),
                 "skipped": sum(t.skipped for t in session.turns),
                 "correctness": avg("correctness"), "depth": avg("depth"),
                 "communication": avg("communication"), "overall": avg("overall")}
        dims = {k: stats[k] for k in ("correctness", "depth", "communication")}
        stats["weakest"] = min(dims, key=dims.get) if done else "n/a"
        return stats

    def report(self, session) -> dict:
        stats = self.compute_stats(session)
        patterns = self.kb.search_feedback(session.company, session.role,
                                           f"{stats['weakest']} {session.role} interview", n=3)
        session.log("retrieved_feedback", agent="Report", query=f"{stats['weakest']} {session.role}",
                    results=[f"{p['company']} | {p['theme']}" for p in patterns])
        try:
            data = parse_json(self.chains["report"].invoke({
                "company": session.company, "role": session.role, "stats": json.dumps(stats),
                "performance": session.performance_summary(), "patterns": format_patterns(patterns)}))
        except Exception:
            data = {}
        plan = data.get("plan") or [{"day": f"Day {i}", "focus": f, "tasks": ["Practise 3 questions on this topic",
                "Record yourself answering and re-listen"]} for i, f in
                enumerate(["Weakest dimension: " + stats["weakest"], "Core fundamentals", "Project story (STAR)",
                           "Mock interview round", "Revision + HR answers"], 1)]
        return {"stats": stats, "company": session.company, "role": session.role,
                "summary": str(data.get("summary", "Report generated from your session scores.")),
                "strengths": _bullets(data.get("strengths")), "gaps": _bullets(data.get("gaps")),
                "plan": plan, "patterns": patterns,
                "turns": [{"question": t.question["question"], "round": t.question["round_type"],
                           "skipped": t.skipped, "scores": None if not t.evaluation else
                           {k: t.evaluation[k] for k in ("correctness", "depth", "communication", "overall")}}
                          for t in session.turns]}
