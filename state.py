"""Session state = conversation memory (company, role, previous answers, trace)."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    question: dict
    answer: str = ""
    evaluation: Optional[dict] = None   # None when skipped
    skipped: bool = False


@dataclass
class Session:
    company: str = "General"
    role: str = "Software Engineer"
    difficulty: str = "Mixed"
    total_questions: int = 5
    pending: Optional[dict] = None                      # question waiting for an answer
    asked_ids: list = field(default_factory=list)
    turns: list = field(default_factory=list)
    messages: list = field(default_factory=list)        # [{"role": "user"|"assistant", "content": str}]
    trace: list = field(default_factory=list)           # routing / retrieval / agent log
    report: Optional[dict] = None

    # ---- memory helpers used in prompts ----
    def chat_history(self, n: int = 6) -> str:
        recent = self.messages[-n:]
        if not recent:
            return "(no previous messages)"
        return "\n".join(f"{m['role'].upper()}: {m['content'][:400]}" for m in recent)

    def performance_summary(self) -> str:
        done = [t for t in self.turns if t.evaluation]
        if not done and not self.turns:
            return "No answers yet."
        lines = []
        for i, t in enumerate(self.turns, 1):
            if t.skipped:
                lines.append(f"Q{i} [{t.question['round_type']}] {t.question['question'][:80]} -> SKIPPED")
            else:
                e = t.evaluation
                lines.append(
                    f"Q{i} [{t.question['round_type']}] {t.question['question'][:80]} -> overall {e['overall']}/10 "
                    f"(correctness {e['correctness']}, depth {e['depth']}, communication {e['communication']}); "
                    f"missed: {', '.join(e.get('missed_points', [])[:3]) or 'none'}"
                )
        return "\n".join(lines)

    def topics_asked(self) -> str:
        return "; ".join(t.question["question"][:60] for t in self.turns) or "none yet"

    def log(self, kind: str, **data):
        self.trace.append({"kind": kind, **data})
