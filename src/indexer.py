import os
import re
import math
from collections import Counter
from typing import List, Dict, Any, Tuple

class PaperIndexer:
    def __init__(self, data_dir: str = "data/extracted"):
        self.data_dir = data_dir
        self.papers_meta = {
            "attention": {
                "title": "Attention Is All You Need",
                "arxiv_id": "1706.03762v7",
                "file": "attention.txt"
            },
            "moshi": {
                "title": "Moshi: a speech-text foundation model for real-time dialogue",
                "arxiv_id": "2410.00037v1",
                "file": "moshi.txt"
            },
            "dynamic_chunking": {
                "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
                "arxiv_id": "2507.07955v2",
                "file": "dynamic_chunking.txt"
            }
        }
        self.settings = {
            "A": {"piece_size": 200, "overlap": 30, "top_k": 3, "rerank": False, "name": "Small (200w, top-3, no-rerank)"},
            "B": {"piece_size": 500, "overlap": 75, "top_k": 4, "rerank": True, "name": "Medium (500w, top-4, rerank)"},
            "C": {"piece_size": 1000, "overlap": 150, "top_k": 6, "rerank": True, "name": "Large (1000w, top-6, rerank)"}
        }
        self.chunks_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.bm25_cache: Dict[str, Dict[str, Any]] = {}
        self._load_and_chunk_all()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', text.lower())

    def _load_and_chunk_all(self):
        for setting_key, cfg in self.settings.items():
            all_chunks = []
            for paper_id, meta in self.papers_meta.items():
                file_path = os.path.join(self.data_dir, meta["file"])
                if not os.path.exists(file_path):
                    continue
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse pages
                page_blocks = content.split("--- [Page ")
                for p_idx, block in enumerate(page_blocks):
                    if not block.strip():
                        continue
                    # page number
                    try:
                        page_num_str = block.split("] ---")[0].strip()
                        page_num = int(page_num_str)
                        page_text = block.split("] ---", 1)[1]
                    except Exception:
                        page_num = p_idx
                        page_text = block

                    # Words in page
                    words = page_text.split()
                    if not words:
                        continue

                    step = max(1, cfg["piece_size"] - cfg["overlap"])
                    for i in range(0, len(words), step):
                        chunk_words = words[i:i + cfg["piece_size"]]
                        if len(chunk_words) < 30 and i > 0:
                            continue
                        chunk_str = " ".join(chunk_words)
                        chunk_tokens = self._tokenize(chunk_str)
                        all_chunks.append({
                            "chunk_id": f"{paper_id}_p{page_num}_{i}",
                            "paper_id": paper_id,
                            "title": meta["title"],
                            "arxiv_id": meta["arxiv_id"],
                            "page": page_num,
                            "text": chunk_str,
                            "tokens": chunk_tokens,
                            "token_count": len(chunk_tokens)
                        })

            self.chunks_cache[setting_key] = all_chunks
            # Build BM25 index
            self.bm25_cache[setting_key] = self._build_bm25(all_chunks)

    def _build_bm25(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> Dict[str, Any]:
        N = len(chunks)
        doc_lens = [c["token_count"] for c in chunks]
        avg_doc_len = sum(doc_lens) / max(1, N)
        df = Counter()
        for c in chunks:
            unique_tokens = set(c["tokens"])
            for t in unique_tokens:
                df[t] += 1

        idf = {}
        for t, freq in df.items():
            idf[t] = math.log(1.0 + (N - freq + 0.5) / (freq + 0.5))

        return {
            "N": N,
            "avg_doc_len": avg_doc_len,
            "doc_lens": doc_lens,
            "idf": idf,
            "k1": k1,
            "b": b
        }

    def _bm25_score(self, query_tokens: List[str], chunk_idx: int, setting_key: str) -> float:
        bm = self.bm25_cache[setting_key]
        chunk = self.chunks_cache[setting_key][chunk_idx]
        doc_len = bm["doc_lens"][chunk_idx]
        score = 0.0
        c_counts = Counter(chunk["tokens"])

        for t in query_tokens:
            if t not in bm["idf"]:
                continue
            tf = c_counts.get(t, 0)
            idf = bm["idf"][t]
            num = tf * (bm["k1"] + 1.0)
            denom = tf + bm["k1"] * (1.0 - bm["b"] + bm["b"] * (doc_len / bm["avg_doc_len"]))
            score += idf * (num / denom)

        return score

    def retrieve(self, query: str, setting: str = "B", target_paper: str = None) -> List[Dict[str, Any]]:
        setting_key = setting.upper()
        if setting_key not in self.settings:
            setting_key = "B"
        cfg = self.settings[setting_key]
        chunks = self.chunks_cache[setting_key]

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Candidate pool
        scores = []
        for idx, chunk in enumerate(chunks):
            if target_paper and chunk["paper_id"] != target_paper:
                continue
            s = self._bm25_score(query_tokens, idx, setting_key)
            scores.append((idx, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        pool_size = min(len(scores), cfg["top_k"] * 3 if cfg["rerank"] else cfg["top_k"])
        candidates = scores[:pool_size]

        if not cfg["rerank"]:
            top_results = []
            for idx, s in candidates[:cfg["top_k"]]:
                item = dict(chunks[idx])
                item["retrieval_score"] = round(s, 3)
                top_results.append(item)
            return top_results

        # Re-ranking logic:
        # Boost by exact phrase overlap, question term density, and title keywords
        query_text_lower = query.lower()
        reranked = []
        for idx, bm_score in candidates:
            chunk = chunks[idx]
            text_lower = chunk["text"].lower()

            # Exact phrase matches
            phrase_bonus = 0.0
            for i in range(len(query_tokens) - 1):
                bi = f"{query_tokens[i]} {query_tokens[i+1]}"
                if bi in text_lower:
                    phrase_bonus += 2.0

            # Term coverage: what proportion of query tokens appear?
            found_terms = sum(1 for t in set(query_tokens) if t in text_lower)
            coverage = found_terms / max(1, len(set(query_tokens)))

            # Paper relevance boost if explicitly queried
            paper_boost = 0.0
            if "attention" in query_text_lower or "transformer" in query_text_lower:
                if chunk["paper_id"] == "attention":
                    paper_boost += 1.5
            if "moshi" in query_text_lower or "mimi" in query_text_lower or "speech" in query_text_lower:
                if chunk["paper_id"] == "moshi":
                    paper_boost += 1.5
            if "dynamic chunking" in query_text_lower or "h-net" in query_text_lower or "hierarchical" in query_text_lower:
                if chunk["paper_id"] == "dynamic_chunking":
                    paper_boost += 1.5

            composite_score = bm_score + phrase_bonus + (coverage * 5.0) + paper_boost
            reranked.append((idx, composite_score))

        reranked.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, comp_score in reranked[:cfg["top_k"]]:
            item = dict(chunks[idx])
            item["retrieval_score"] = round(comp_score, 3)
            results.append(item)

        return results

if __name__ == "__main__":
    indexer = PaperIndexer()
    for s in ["A", "B", "C"]:
        print(f"Setting {s}: {len(indexer.chunks_cache[s])} chunks indexed")
    res = indexer.retrieve("What is the audio frame rate and latency of Moshi?", setting="B")
    print(f"\nTest retrieval returned {len(res)} chunks. Top chunk from: {res[0]['title']} (Page {res[0]['page']})")
    print(res[0]['text'][:200] + "...")

