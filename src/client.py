import os
import time
import json
import random
from typing import Dict, Any, Optional, Generator, Tuple
import urllib.request
import urllib.error

from src.standin_service import StandinModelService, GroqRateLimitError, GroqServiceUnavailableError

class ModelClient:
    def __init__(
        self,
        standin_service: Optional[StandinModelService] = None,
        groq_api_key: Optional[str] = None,
        model_name: str = "llama-3.1-8b-instant",
        max_retries: int = 2,
        request_timeout: float = 12.0,
        max_retry_wait: float = 6.0
    ):
        self.standin = standin_service or StandinModelService()
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.model_name = model_name
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.max_retry_wait = max_retry_wait

    def execute_request(
        self,
        prompt: str,
        system_prompt: str = "",
        source: str = "stand-in",
        synthetic_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes a streaming request against real Groq API or Stand-in service.
        Records detailed timing breakdown: waiting_ms, working_ms, retrying_ms,
        first_token_ms, total_ms, status, retries, tokens_in, tokens_out.
        """
        req_start_time = time.time()
        retries_done = 0
        total_waiting_ms = 0.0
        total_retrying_ms = 0.0
        total_working_ms = 0.0
        first_token_ms = 0.0
        status = "ok"
        tokens_in = 0
        tokens_out = 0
        accumulated_text = []

        for attempt in range(self.max_retries + 1):
            elapsed_so_far = time.time() - req_start_time
            if elapsed_so_far > self.request_timeout:
                status = "timeout"
                break

            work_start = time.time()
            try:
                if source == "real" and self.groq_api_key:
                    stream_res = self._call_real_groq(prompt, system_prompt)
                else:
                    # stand-in service
                    stream_res = self.standin.generate_stream(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        synthetic_answer=synthetic_answer
                    )

                saw_first_token = False
                attempt_first_token_ms = 0.0

                for chunk in stream_res:
                    ev = chunk.get("event")
                    if ev == "first_token":
                        saw_first_token = True
                        attempt_first_token_ms = chunk["ttft_ms"]
                    elif ev == "token_chunk":
                        accumulated_text.append(chunk.get("chunk", ""))
                        tokens_in = chunk.get("tokens_in", tokens_in)
                        tokens_out = chunk.get("tokens_out", tokens_out)
                    elif ev == "done":
                        tokens_in = chunk.get("tokens_in", tokens_in)
                        tokens_out = chunk.get("tokens_out", tokens_out)

                work_end = time.time()
                attempt_working_ms = (work_end - work_start) * 1000.0
                total_working_ms += attempt_working_ms

                # First token time is relative to when this successful attempt started plus retries
                if saw_first_token:
                    first_token_ms = round(total_retrying_ms + attempt_first_token_ms, 2)
                else:
                    first_token_ms = round(total_working_ms * 0.4, 2)

                status = "ok"
                break

            except (GroqRateLimitError, urllib.error.HTTPError) as e:
                work_end = time.time()
                total_working_ms += (work_end - work_start) * 1000.0

                # Extract retry_after
                retry_after = 2.0
                limit_type = "rate_limit"
                if isinstance(e, GroqRateLimitError):
                    retry_after = e.retry_after
                    limit_type = e.limit_type
                elif isinstance(e, urllib.error.HTTPError):
                    if e.code == 429:
                        limit_type = "requests_or_tokens"
                        hdr = e.headers.get("retry-after")
                        if hdr:
                            try:
                                retry_after = float(hdr)
                            except ValueError:
                                retry_after = 3.0
                    else:
                        status = "error"
                        break

                # Decision: Should we retry?
                if limit_type == "requests_daily" or retry_after > self.max_retry_wait or attempt >= self.max_retries:
                    status = "rate_limited"
                    retries_done = attempt
                    break

                # Sleep and retry with jitter
                retries_done += 1
                sleep_time = retry_after + random.uniform(0.1, 0.4)
                retry_sleep_start = time.time()
                time.sleep(sleep_time)
                actual_sleep_ms = (time.time() - retry_sleep_start) * 1000.0
                total_retrying_ms += actual_sleep_ms

            except (GroqServiceUnavailableError, urllib.error.URLError, TimeoutError) as e:
                work_end = time.time()
                total_working_ms += (work_end - work_start) * 1000.0

                if attempt >= self.max_retries:
                    status = "error"
                    retries_done = attempt
                    break

                retries_done += 1
                sleep_time = (0.4 * (2 ** attempt)) + random.uniform(0.05, 0.2)
                retry_sleep_start = time.time()
                time.sleep(sleep_time)
                actual_sleep_ms = (time.time() - retry_sleep_start) * 1000.0
                total_retrying_ms += actual_sleep_ms

            except Exception as e:
                work_end = time.time()
                total_working_ms += (work_end - work_start) * 1000.0
                status = "error"
                break

        total_ms = round((time.time() - req_start_time) * 1000.0, 2)
        if status != "ok" and first_token_ms == 0.0:
            first_token_ms = total_ms  # No token produced

        return {
            "status": status,
            "response_text": "".join(accumulated_text).strip(),
            "first_token_ms": first_token_ms,
            "total_ms": total_ms,
            "waiting_ms": round(total_waiting_ms, 2),
            "working_ms": round(total_working_ms, 2),
            "retrying_ms": round(total_retrying_ms, 2),
            "retries": retries_done,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out
        }

    def _call_real_groq(self, prompt: str, system_prompt: str) -> Generator[Dict[str, Any], None, None]:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a research assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "stream": True
        }

        start_time = time.time()
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
            saw_first = False
            tok_in = int(len(prompt + system_prompt) / 3.8)
            tok_out = 0

            for line in resp:
                line_str = line.decode("utf-8").strip()
                if not line_str or not line_str.startswith("data:"):
                    continue
                data_part = line_str[5:].strip()
                if data_part == "[DONE]":
                    break
                try:
                    obj = json.loads(data_part)
                    delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        if not saw_first:
                            saw_first = True
                            ttft = (time.time() - start_time) * 1000.0
                            yield {"event": "first_token", "ttft_ms": round(ttft, 2)}
                        tok_out += max(1, int(len(delta) / 3.8))
                        yield {"event": "token_chunk", "chunk": delta, "tokens_in": tok_in, "tokens_out": tok_out}
                except Exception:
                    pass

            yield {
                "event": "done",
                "total_ms": round((time.time() - start_time) * 1000.0, 2),
                "tokens_in": tok_in,
                "tokens_out": tok_out
            }

