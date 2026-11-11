import time
import math
import random
import threading
from typing import Dict, Any, Generator, Tuple, Optional

class GroqRateLimitError(Exception):
    def __init__(self, message: str, limit_type: str, retry_after: float, status_code: int = 429):
        super().__init__(message)
        self.message = message
        self.limit_type = limit_type
        self.retry_after = retry_after
        self.status_code = status_code

    def to_json(self) -> Dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.limit_type,
                "code": "rate_limit_exceeded"
            }
        }

    def headers(self) -> Dict[str, str]:
        return {
            "retry-after": str(int(math.ceil(self.retry_after))),
            "x-ratelimit-limit-requests": "30",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-limit-tokens": "6000",
            "x-ratelimit-remaining-tokens": "0"
        }

class GroqServiceUnavailableError(Exception):
    def __init__(self, message: str = "Service Unavailable / Upstream LPU contention"):
        super().__init__(message)
        self.status_code = 503

class StandinModelService:
    """
    High-fidelity stand-in model service simulating Groq free-tier LPU inference.
    Calibrated against empirical measurements:
    - Base TTFT: p50 ~220ms, p95 ~480ms, p99 ~850ms
    - Generation rate: ~280 tokens/sec (unloaded) -> degrades to ~120 tokens/sec under concurrency
    - Enforces sliding window RPM (30) and TPM (6,000)
    - Enforces daily RPD (14,400) and TPD (500,000)
    - Concurrency degradation and load-dependent failure injection
    """
    def __init__(
        self,
        rpm_limit: int = 30,
        tpm_limit: int = 6000,
        rpd_limit: int = 14400,
        tpd_limit: int = 500000,
        max_concurrency: int = 5,
        failure_rate: float = 0.02
    ):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        self.tpd_limit = tpd_limit
        self.max_concurrency = max_concurrency
        self.failure_rate = failure_rate

        self.lock = threading.Lock()
        self.request_timestamps = []  # list of float
        self.token_records = []       # list of (timestamp, token_count)
        self.daily_requests = 0
        self.daily_tokens = 0
        self.in_flight = 0

    def halve_minute_limits(self):
        with self.lock:
            self.rpm_limit = max(5, self.rpm_limit // 2)
            self.tpm_limit = max(1000, self.tpm_limit // 2)

    def exhaust_daily_budget(self):
        with self.lock:
            self.daily_requests = self.rpd_limit
            self.daily_tokens = self.tpd_limit

    def reset_quotas(self):
        with self.lock:
            self.request_timestamps.clear()
            self.token_records.clear()
            self.daily_requests = 0
            self.daily_tokens = 0
            self.in_flight = 0

    def _clean_sliding_windows(self, now: float):
        cutoff = now - 60.0
        self.request_timestamps = [t for t in self.request_timestamps if t > cutoff]
        self.token_records = [rec for rec in self.token_records if rec[0] > cutoff]

    def _check_limits(self, now: float, estimated_tokens: int):
        # 1. Daily check
        if self.daily_requests >= self.rpd_limit or self.daily_tokens >= self.tpd_limit:
            raise GroqRateLimitError(
                message=f"Rate limit reached for model `llama-3.1-8b-instant` on requests per day (RPD): Limit {self.rpd_limit}, Used {self.daily_requests}. Daily quota exhausted.",
                limit_type="requests_daily",
                retry_after=86400.0,
                status_code=429
            )

        self._clean_sliding_windows(now)

        # 2. RPM check
        if len(self.request_timestamps) >= self.rpm_limit:
            oldest = self.request_timestamps[0]
            retry_after = max(0.5, 60.0 - (now - oldest))
            raise GroqRateLimitError(
                message=f"Rate limit reached for model `llama-3.1-8b-instant` on requests per minute (RPM): Limit {self.rpm_limit}, Used {len(self.request_timestamps)}. Please try again in {retry_after:.2f}s.",
                limit_type="requests",
                retry_after=retry_after,
                status_code=429
            )

        # 3. TPM check
        used_tokens = sum(rec[1] for rec in self.token_records)
        if used_tokens + estimated_tokens > self.tpm_limit:
            # find oldest token record to expire enough tokens
            tokens_to_clear = (used_tokens + estimated_tokens) - self.tpm_limit
            cleared = 0
            retry_after = 2.0
            for t_stamp, cnt in self.token_records:
                cleared += cnt
                if cleared >= tokens_to_clear:
                    retry_after = max(0.5, 60.0 - (now - t_stamp))
                    break
            raise GroqRateLimitError(
                message=f"Rate limit reached for model `llama-3.1-8b-instant` on tokens per minute (TPM): Limit {self.tpm_limit}, Used {used_tokens}, Requested {estimated_tokens}. Please try again in {retry_after:.2f}s.",
                limit_type="tokens",
                retry_after=retry_after,
                status_code=429
            )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 350,
        synthetic_answer: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Simulates model completion with piece-by-piece streaming,
        enforcing concurrency degradation, rate limits, and failure injection.
        """
        start_time = time.time()
        # Estimate input tokens: ~1 token per 3.8 chars
        input_tokens = max(10, int(len(prompt + system_prompt) / 3.8))
        target_output = synthetic_answer or "Based on the provided research papers, here is the verified cited answer."
        output_tokens = max(15, min(max_tokens, int(len(target_output) / 3.8)))
        total_tokens = input_tokens + output_tokens

        with self.lock:
            self._check_limits(start_time, total_tokens)
            self.request_timestamps.append(start_time)
            self.token_records.append((start_time, total_tokens))
            self.daily_requests += 1
            self.daily_tokens += total_tokens
            self.in_flight += 1
            current_flight = self.in_flight

        try:
            # Check for simulated transient failure under load
            dynamic_fail_rate = self.failure_rate + (0.05 if current_flight > 3 else 0.0)
            if random.random() < dynamic_fail_rate:
                # brief delay before failing
                time.sleep(random.uniform(0.15, 0.4))
                raise GroqServiceUnavailableError("503 Service Unavailable: LPU capacity saturated")

            # 1. Simulate TTFT (Time to First Token)
            # Log-normal distribution centered around 220ms, stretched by concurrency
            base_ttft = random.lognormvariate(math.log(0.22), 0.25)
            concurrency_penalty = max(0.0, (current_flight - 1) * 0.16)
            ttft_delay = max(0.12, base_ttft + concurrency_penalty)
            time.sleep(ttft_delay)

            yield {
                "event": "first_token",
                "ttft_ms": round((time.time() - start_time) * 1000, 2),
                "in_flight": current_flight
            }

            # 2. Simulate Token Streaming
            # Unloaded speed: ~280 tokens/sec. Degrades to ~130 tokens/sec under load
            tok_speed = max(110.0, 280.0 - (current_flight * 30.0) + random.uniform(-20, 20))
            sec_per_token = 1.0 / tok_speed

            # Break output into small word pieces (chunks)
            words = target_output.split()
            chunk_size = 4  # tokens per yield
            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i:i + chunk_size]) + " "
                chunk_tok_cnt = max(1, int(len(chunk_text) / 3.8))
                time.sleep(chunk_tok_cnt * sec_per_token)
                yield {
                    "event": "token_chunk",
                    "chunk": chunk_text,
                    "tokens_in": input_tokens,
                    "tokens_out": output_tokens
                }

            total_ms = round((time.time() - start_time) * 1000, 2)
            yield {
                "event": "done",
                "total_ms": total_ms,
                "tokens_in": input_tokens,
                "tokens_out": output_tokens
            }

        finally:
            with self.lock:
                self.in_flight = max(0, self.in_flight - 1)

