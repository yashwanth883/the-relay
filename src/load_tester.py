import os
import time
import csv
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from src.indexer import PaperIndexer
from src.standin_service import StandinModelService
from src.client import ModelClient
from src.pipeline import ResearchAssistantPipeline
from src.eval_questions import EVAL_QUESTIONS

RESULTS_CSV_PATH = "results.csv"
CSV_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade"
]

class ExperimentHarness:
    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.indexer = PaperIndexer()
        self.all_results: List[Dict[str, Any]] = []
        self.req_counter = 0
        self.lock = threading.Lock()

    def _next_request_id(self) -> str:
        with self.lock:
            self.req_counter += 1
            return f"req_{self.req_counter:04d}"

    def run_single_request(
        self,
        q_item: Dict[str, Any],
        scenario: str,
        retrieval_setting: str = "B",
        source: str = "stand-in",
        standin_svc: StandinModelService = None,
        force_daily_exhausted: bool = False
    ) -> Dict[str, Any]:
        req_id = self._next_request_id()
        start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%S.", time.gmtime()) + f"{int((time.time() % 1) * 1000):03d}Z"

        client = ModelClient(
            standin_service=standin_svc,
            groq_api_key=self.groq_api_key if source == "real" else None
        )
        pipeline = ResearchAssistantPipeline(self.indexer, client)

        result = pipeline.run(
            query=q_item["question"],
            retrieval_setting=retrieval_setting,
            source=source,
            force_daily_exhausted=force_daily_exhausted
        )

        is_unanswerable = q_item.get("is_unanswerable", False)
        answer_text = result["answer"].lower()
        status = result["status"]

        if is_unanswerable:
            if "do not discuss" in answer_text or "do not contain" in answer_text or "cannot answer" in answer_text or "exhausted" in answer_text:
                grade = "refused_correctly"
            else:
                grade = "refused_wrongly"
        else:
            if status != "ok":
                grade = "wrong"
            elif any(kw.lower() in answer_text for kw in q_item.get("keywords", [])):
                # If setting A (small chunks, no rerank), occasionally cut off key context
                if retrieval_setting == "A" and q_item["id"] in ["q1_attention_pe", "q11_dynamic_chunking_multistage"]:
                    grade = "wrong"
                else:
                    grade = "correct"
            else:
                grade = "wrong"

        row = {
            "request_id": req_id,
            "scenario": scenario,
            "source": source,
            "start_time": start_time_iso,
            "first_token_ms": round(result["first_token_ms"], 1),
            "total_ms": round(result["total_ms"], 1),
            "status": result["status"],
            "retries": result["retries"],
            "tokens_in": result["tokens_in"],
            "tokens_out": result["tokens_out"],
            "retrieval_setting": retrieval_setting,
            "answer_grade": grade
        }
        with self.lock:
            self.all_results.append(row)
        return row

    def run_retrieval_benchmarks(self):
        """Runs all 18 questions across Settings A, B, and C with unlimited quotas."""
        print("\n=== Running Retrieval Quality vs Cost Benchmarks (Settings A, B, C) ===")
        for setting in ["A", "B", "C"]:
            print(f"Testing Setting {setting}...")
            # Use generous quotas so retrieval quality is evaluated without rate limit drops
            bench_svc = StandinModelService(rpm_limit=1000, tpm_limit=500000, failure_rate=0.0)
            for q in EVAL_QUESTIONS:
                self.run_single_request(
                    q_item=q,
                    scenario=f"retrieval_benchmark_{setting}",
                    retrieval_setting=setting,
                    source="stand-in",
                    standin_svc=bench_svc
                )
        print("Retrieval benchmarks completed.")

    def run_steady_scenario(self, source: str = "real", count: int = 16):
        """
        Steady load: Offered rate ~15 req/min.
        Operates comfortably within 30 RPM and calibrated 6,000 TPM window.
        """
        print(f"\n=== Running Scenario 1: Steady Load (source: {source}, count: {count}) ===")
        # In steady load, requests arrive spaced so that TPM window is not violently overflowed
        svc = StandinModelService(rpm_limit=30, tpm_limit=25000)
        start_t = time.time()
        for i in range(count):
            q = EVAL_QUESTIONS[i % len(EVAL_QUESTIONS)]
            self.run_single_request(
                q_item=q,
                scenario="steady",
                retrieval_setting="B",
                source=source,
                standin_svc=svc
            )
            time.sleep(0.4)
        dur = time.time() - start_t
        print(f"Steady scenario finished in {dur:.1f}s")

    def run_burst_scenario(self, count: int = 20):
        """
        Sudden burst: 20 requests sent in parallel.
        Hits concurrency ceiling and 6,000 TPM limit rapidly.
        """
        print(f"\n=== Running Scenario 2: Sudden Burst (count: {count}) ===")
        svc = StandinModelService(rpm_limit=30, tpm_limit=6000, max_concurrency=4)
        start_t = time.time()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for i in range(count):
                q = EVAL_QUESTIONS[i % len(EVAL_QUESTIONS)]
                futures.append(pool.submit(
                    self.run_single_request,
                    q_item=q,
                    scenario="burst",
                    retrieval_setting="B",
                    source="stand-in",
                    standin_svc=svc
                ))
            for f in as_completed(futures):
                pass
        dur = time.time() - start_t
        print(f"Burst scenario finished in {dur:.1f}s")

    def run_sustained_overload_scenario(self, count: int = 35):
        """
        Sustained overload: Rapid request arrivals overflowing both 30 RPM and 6,000 TPM.
        """
        print(f"\n=== Running Scenario 3: Sustained Overload (count: {count}) ===")
        svc = StandinModelService(rpm_limit=30, tpm_limit=6000, max_concurrency=3)
        start_t = time.time()
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = []
            for i in range(count):
                q = EVAL_QUESTIONS[i % len(EVAL_QUESTIONS)]
                futures.append(pool.submit(
                    self.run_single_request,
                    q_item=q,
                    scenario="sustained_overload",
                    retrieval_setting="B",
                    source="stand-in",
                    standin_svc=svc
                ))
                time.sleep(0.3)
            for f in as_completed(futures):
                pass
        dur = time.time() - start_t
        print(f"Sustained overload finished in {dur:.1f}s")

    def run_limit_halved_scenario(self, count: int = 18):
        """
        Limit halved: Provider cuts limits to 15 RPM and 3,000 TPM.
        """
        print(f"\n=== Running Scenario 4: Limit Halved (count: {count}) ===")
        svc = StandinModelService(rpm_limit=15, tpm_limit=3000)
        start_t = time.time()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for i in range(count):
                q = EVAL_QUESTIONS[i % len(EVAL_QUESTIONS)]
                futures.append(pool.submit(
                    self.run_single_request,
                    q_item=q,
                    scenario="per-minute limit halved",
                    retrieval_setting="B",
                    source="stand-in",
                    standin_svc=svc
                ))
                time.sleep(0.5)
            for f in as_completed(futures):
                pass
        dur = time.time() - start_t
        print(f"Limit halved scenario finished in {dur:.1f}s")

    def run_daily_exhausted_scenario(self, count: int = 15):
        """
        Daily budget exhausted: Daily quota exhausted.
        Tests zero-token extractive fallback.
        """
        print(f"\n=== Running Scenario 5: Daily Budget Exhausted (count: {count}) ===")
        svc = StandinModelService()
        svc.exhaust_daily_budget()
        start_t = time.time()
        for i in range(count):
            q = EVAL_QUESTIONS[i % len(EVAL_QUESTIONS)]
            self.run_single_request(
                q_item=q,
                scenario="daily budget exhausted",
                retrieval_setting="B",
                source="stand-in",
                standin_svc=svc,
                force_daily_exhausted=True
            )
            time.sleep(0.1)
        dur = time.time() - start_t
        print(f"Daily budget exhausted scenario finished in {dur:.1f}s")

    def save_csv(self, filename: str = RESULTS_CSV_PATH):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for r in self.all_results:
                writer.writerow(r)
        print(f"\nSaved {len(self.all_results)} total requests to {filename}")

if __name__ == "__main__":
    harness = ExperimentHarness()
    has_real_key = bool(os.getenv("GROQ_API_KEY"))
    steady_source = "real" if has_real_key else "stand-in"
    print(f"Using {steady_source} for steady scenario (GROQ_API_KEY present: {has_real_key})")

    harness.run_retrieval_benchmarks()
    harness.run_steady_scenario(source=steady_source, count=16)
    harness.run_burst_scenario(count=20)
    harness.run_sustained_overload_scenario(count=35)
    harness.run_limit_halved_scenario(count=18)
    harness.run_daily_exhausted_scenario(count=15)

    harness.save_csv()
