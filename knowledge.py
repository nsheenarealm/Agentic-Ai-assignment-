"""ChromaDB knowledge base: past interview questions + historical feedback patterns."""
import csv
import hashlib
import os
from pathlib import Path

import chromadb

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
DB_DIR = Path(os.getenv("CHROMA_DIR", BASE / "chroma_db"))


def _read_csvs(pattern: str):
    rows = []
    for f in sorted(DATA_DIR.glob(pattern)):
        with open(f, newline="", encoding="utf-8-sig") as fh:
            for i, r in enumerate(csv.DictReader(fh), 1):
                r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
                r["id"] = f"{f.stem}:{r.get('id') or i}"
                rows.append(r)
    return rows


def _files_hash() -> str:
    h = hashlib.md5()
    for f in sorted(DATA_DIR.glob("*.csv")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


class KnowledgeBase:
    def __init__(self):
        DB_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(DB_DIR))
        self.questions, self.patterns = [], []
        self.ensure_index()

    # ---------- indexing ----------
    def ensure_index(self, force: bool = False):
        self.questions = _read_csvs("question_bank*.csv")
        self.patterns = _read_csvs("feedback_patterns*.csv")
        hash_file = DB_DIR / "index.hash"
        new_hash = _files_hash()
        old_hash = hash_file.read_text() if hash_file.exists() else ""
        names = {c if isinstance(c, str) else c.name for c in self.client.list_collections()}
        if force or new_hash != old_hash or not {"interview_questions", "feedback_patterns"} <= names:
            self._rebuild()
            hash_file.write_text(new_hash)
        self.q_col = self.client.get_or_create_collection("interview_questions", metadata={"hnsw:space": "cosine"})
        self.f_col = self.client.get_or_create_collection("feedback_patterns", metadata={"hnsw:space": "cosine"})

    def _rebuild(self):
        for name in ("interview_questions", "feedback_patterns"):
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
        q_col = self.client.get_or_create_collection("interview_questions", metadata={"hnsw:space": "cosine"})
        f_col = self.client.get_or_create_collection("feedback_patterns", metadata={"hnsw:space": "cosine"})
        self._add(q_col, self.questions,
                  lambda r: f"{r['company']} {r['role']} {r['round_type']} interview question: {r['question']}",
                  ["company", "role", "round_type", "difficulty", "question", "expected_points", "source_year"])
        self._add(f_col, self.patterns,
                  lambda r: f"{r['company']} {r['role']} {r['theme']} feedback: {r['pattern']}",
                  ["company", "role", "theme", "pattern", "source_year"])

    @staticmethod
    def _add(col, rows, doc_fn, meta_keys, batch=100):
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            col.add(ids=[r["id"] for r in chunk],
                    documents=[doc_fn(r) for r in chunk],
                    metadatas=[{k: r.get(k, "") for k in meta_keys} for r in chunk])

    # ---------- lookups for the UI ----------
    def companies(self):
        return sorted({r["company"] for r in self.questions})

    def roles(self):
        base = {"Software Engineer", "Data Analyst"}
        return sorted(base | {r["role"] for r in self.questions if r["role"] and r["role"] != "Any"})

    # ---------- similarity search ----------
    @staticmethod
    def _where(companies):
        return {"company": companies[0]} if len(companies) == 1 else {"company": {"$in": companies}}

    def _query(self, col, text, companies, n):
        where = self._where(companies)
        ids = col.get(where=where, include=[])["ids"]
        if not ids:
            return []
        res = col.query(query_texts=[text], n_results=min(n, len(ids)), where=where)
        out = []
        for id_, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"id": id_, "distance": float(dist), **meta})
        return out

    @staticmethod
    def _role_bonus(item_role, role):
        item_role, role = (item_role or "").lower(), (role or "").lower()
        return 0.15 if (item_role == role or item_role == "any") else 0.0

    def search_questions(self, company, role, query, n=6, difficulty=None, exclude_ids=()):
        """Company-specific questions first; 'General' bank fills the gap."""
        cands = self._query(self.q_col, query, [company], 20) if company != "General" else []
        cands += self._query(self.q_col, query, ["General"], 20)
        seen, out = set(), []
        for c in cands:
            if c["id"] in seen or c["id"] in exclude_ids:
                continue
            seen.add(c["id"])
            if difficulty and difficulty != "Mixed" and c["difficulty"] != difficulty:
                continue
            if c["role"].lower() not in ("any", role.lower()):
                continue
            c["rank"] = c["distance"] - self._role_bonus(c["role"], role) - (0.2 if c["company"] == company else 0)
            c["expected_points"] = [p.strip() for p in c["expected_points"].split(";") if p.strip()]
            out.append(c)
        out.sort(key=lambda x: x["rank"])
        return out[:n]

    def search_feedback(self, company, role, query, n=3):
        companies = [company] if company == "General" else [company, "General"]
        cands = self._query(self.f_col, query, companies, 12)
        for c in cands:
            c["rank"] = c["distance"] - self._role_bonus(c["role"], role) - (0.2 if c["company"] == company else 0)
        cands.sort(key=lambda x: x["rank"])
        return cands[:n]
