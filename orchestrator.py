"""Intent routing + turn-by-turn interview flow."""
from agents import Agents, AgentError
from state import Session, Turn

HELP = ("I can run a **mock interview** (say *start* / *next*), **grade your answer** (just type it), "
        "give **coaching** (say *coach me* or ask about a company), or build your **final report**.")


def _bar(n):  # tiny visual score bar
    return "█" * int(n) + "░" * (10 - int(n))


def format_evaluation(e: dict) -> str:
    out = [f"**Score: {e['overall']}/10**  \n"
           f"Correctness `{_bar(e['correctness'])}` {e['correctness']}/10  \n"
           f"Depth `{_bar(e['depth'])}` {e['depth']}/10  \n"
           f"Communication `{_bar(e['communication'])}` {e['communication']}/10"]
    if e["strengths"]:
        out.append("**✅ Strengths**\n" + "\n".join(f"- {s}" for s in e["strengths"]))
    if e["missed_points"]:
        out.append("**⚠️ Missed points**\n" + "\n".join(f"- {s}" for s in e["missed_points"]))
    if e["improvements"]:
        out.append("**🛠 How to improve**\n" + "\n".join(f"- {s}" for s in e["improvements"]))
    if e["patterns"]:
        out.append("**📚 Grounded in past feedback**\n" + "\n".join(
            f"- [{p['company']}] {p['theme']}: {p['pattern']}" for p in e["patterns"]))
    if e["follow_up"]:
        out.append(f"💬 *Interviewer follow-up (optional): {e['follow_up']}*")
    return "\n\n".join(out)


def format_question(session: Session, q: dict) -> str:
    n = len(session.turns) + 1
    return (f"**Question {n}/{session.total_questions}** · {q['round_type']} · {q['difficulty']} · "
            f"from the {q['company']} bank\n\n> {q['question']}")


class Orchestrator:
    def __init__(self, llm, kb):
        self.agents = Agents(llm, kb)

    def handle(self, session: Session, text: str, forced_intent: str = None) -> dict:
        session.messages.append({"role": "user", "content": text})
        try:
            if forced_intent:
                intent, method = forced_intent, "button"
            else:
                intent, method = self.agents.detect_intent(session, text)
            session.log("intent", message=text[:80], intent=intent, method=method)
            reply = getattr(self, f"_{intent.lower()}")(session, text)
        except Exception as e:   # never crash the UI
            intent, reply = "ERROR", f"⚠️ Something went wrong: `{type(e).__name__}: {e}`"
        session.messages.append({"role": "assistant", "content": reply})
        return {"intent": intent, "reply": reply}

    # ---- handlers -------------------------------------------------------
    def _practice(self, session, text):
        t = text.lower()
        if session.pending and ("repeat" in t or "again" in t):
            return "Here it is again:\n\n" + format_question(session, session.pending)
        skipped = False
        if session.pending:      # "next"/"skip" while a question is open => skip it
            session.turns.append(Turn(question=session.pending, skipped=True))
            session.pending = None
            skipped = True
        return self._ask_next(session, prefix="Skipped. \n\n" if skipped else "")

    def _ask_next(self, session, prefix=""):
        if len(session.turns) >= session.total_questions:
            return prefix + "🎉 That was the last question! Say **report** to get your session report."
        q = self.agents.next_question(session)
        if q is None:
            return prefix + "I've run out of unseen questions for this setup. Say **report** to wrap up."
        intro = ""
        if not session.turns:
            intro = (f"Welcome! Mock interview for **{session.company} · {session.role}** "
                     f"({session.total_questions} questions). Answer like you would in the real room.\n\n")
        return prefix + intro + format_question(session, q)

    def _evaluate(self, session, text):
        if not session.pending:
            return "There's no open question. Say **start** to begin."
        q = session.pending
        ev = self.agents.evaluate(session, text)          # may raise AgentError -> question stays open
        session.turns.append(Turn(question=q, answer=text, evaluation=ev))
        session.pending = None
        session.log("evaluation", question=q["question"][:60], overall=ev["overall"], reasoning=ev["reasoning"][:400])
        reply = format_evaluation(ev)
        if len(session.turns) < session.total_questions:
            return reply + "\n\n---\n\n" + self._ask_next(session)
        return reply + "\n\n---\n\n🎉 That was the last question! Say **report** for your session report."

    def _coach(self, session, text):
        advice, patterns = self.agents.coach(session, text)
        src = ""
        if patterns:
            src = "\n\n**Sources used:**\n" + "\n".join(
                f"- [P{i}] {p['company']} · {p['theme']} ({p['source_year']} data)" for i, p in enumerate(patterns, 1))
        return advice + src

    def _report(self, session, text):
        if not session.turns:
            return "Answer at least one question first, then I can build your report."
        if session.pending:   # unanswered question is dropped from the report
            session.pending = None
        session.report = self.agents.report(session)
        r = session.report
        s = r["stats"]
        return (f"📄 **Session report ready** (see the *Report* tab to download the PDF).\n\n"
                f"Overall **{s['overall']}/10** · weakest area: **{s['weakest']}**\n\n{r['summary']}")

    def _other(self, session, text):
        return HELP
