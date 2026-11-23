#!/usr/bin/env python3
"""
squid5090 OpenAI-compatible HTTP API server.

Wraps the persistent native C++/CUDA worker (`squid5090 serve-stdio`) so the model
and captured CUDA Graphs stay resident in VRAM across requests, with live token
streaming (SSE), multi-turn chat, reasoning_content parsing, tool calls, and
both /v1/chat/completions and /v1/completions endpoints.

Dependencies: Python 3.10+ stdlib, tokenizers, jinja2 (zero FastAPI/uvicorn needed).
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from jinja2 import Environment
from tokenizers import Tokenizer


class SquidWorker:
    """Owns the persistent `squid5090 serve-stdio` GPU process."""

    def __init__(
        self,
        bin_path: str,
        model_dir: Path,
        mode: str,
        kv: str,
        ctx: int,
        mtp: int,
        embed_host: bool,
        no_graph: bool,
    ):
        self.model_dir = model_dir
        self.default_mtp = mtp
        self.max_ctx = ctx
        self.lock = threading.Lock()

        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tpl_str = (model_dir / "chat_template.jinja").read_text()
        tpl_str = re.sub(r"\{%-?\s*(?:end)?generation\s*-?%\}", "", tpl_str)
        env = Environment()
        env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
        self.chat_tpl = env.from_string(tpl_str)

        self.id_msg_model = self.tok.token_to_id("<|message_model|>")
        self.id_thinking = self.tok.token_to_id("<|content_thinking|>")
        self.id_text = self.tok.token_to_id("<|content_text|>")
        self.id_tool_json = self.tok.token_to_id("<|content_invoke_tool_json|>")
        self.id_end_msg = self.tok.token_to_id("<|end_message|>")
        self.id_eos = self.tok.token_to_id("<|content_model_end_sampling|>")

        cmd = [
            bin_path, "serve-stdio", str(model_dir),
            "--mode", mode,
            "--kv", kv,
            "--ctx", str(ctx),
            "--mtp", str(mtp),
        ]
        if embed_host:
            cmd.append("--embed-host")
        if no_graph:
            cmd.append("--no-graph")

        print(f"[server] Launching worker: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdout is not None and self.proc.stdin is not None
        self.ready_info = ""
        while True:
            line = self.proc.stdout.readline()
            if not line:
                rc = self.proc.poll()
                raise RuntimeError(f"squid5090 worker exited prematurely (rc={rc})")
            line_s = line.strip()
            print(f"[worker] {line_s}", flush=True)
            if line_s.startswith("READY"):
                self.ready_info = line_s
                break

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()

    def render_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: Any = "none",
    ) -> List[int]:
        # Canonicalize tool_calls arguments from JSON string -> dict if needed by template
        norm_msgs = []
        for m in messages:
            mc = dict(m)
            if mc.get("role") == "assistant" and mc.get("tool_calls"):
                tcs = []
                for tc in mc["tool_calls"]:
                    tcc = dict(tc)
                    fn = dict(tcc.get("function", {}))
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except Exception:
                            args = {}
                    fn["arguments"] = args
                    tcc["function"] = fn
                    tcs.append(tcc)
                mc["tool_calls"] = tcs
            norm_msgs.append(mc)

        rendered = self.chat_tpl.render(
            messages=norm_msgs,
            tools=tools or None,
            add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )
        return self.tok.encode(rendered, add_special_tokens=False).ids

    def encode_raw(self, prompt: str) -> List[int]:
        return self.tok.encode(prompt, add_special_tokens=False).ids

    def stream_tokens(
        self, prompt_ids: List[int], max_tokens: int, mtp: Optional[int] = None
    ) -> Iterator[Tuple[str, Any]]:
        """Yields ('tok', int) for each emitted token, then ('done', stats_dict) or ('err', msg)."""
        use_mtp = self.default_mtp if mtp is None else mtp
        tok_csv = ",".join(str(x) for x in prompt_ids)
        with self.lock:
            assert self.proc.stdin is not None and self.proc.stdout is not None
            self.proc.stdin.write(f"GEN {max_tokens} {use_mtp} {tok_csv}\n")
            self.proc.stdin.flush()
            finished = False
            try:
                while True:
                    line = self.proc.stdout.readline()
                    if not line:
                        finished = True
                        yield ("err", "worker_died")
                        return
                    s = line.strip()
                    if s.startswith("T "):
                        yield ("tok", int(s[2:]))
                    elif s.startswith("DONE "):
                        finished = True
                        parts = s.split()
                        p_toks = int(parts[1])
                        g_toks = int(parts[2])
                        prefill_ms = float(parts[3])
                        decode_ms = float(parts[4])
                        acc_pct = float(parts[5])
                        dl = float(parts[6])
                        finish = parts[7] if len(parts) > 7 else "stop"
                        dec_toks = max(0, g_toks - 1)
                        tps = (dec_toks * 1000.0 / decode_ms) if decode_ms > 0 else 0.0
                        yield (
                            "done",
                            {
                                "prompt_tokens": p_toks,
                                "completion_tokens": g_toks,
                                "prefill_ms": prefill_ms,
                                "decode_ms": decode_ms,
                                "tokens_per_second": round(tps, 2),
                                "spec_acc_pct": acc_pct,
                                "spec_dl": dl,
                                "finish_reason": finish,
                            },
                        )
                        return
                    elif s.startswith("ERR "):
                        finished = True
                        yield ("err", s[4:])
                        return
            finally:
                # If client disconnected mid-stream, drain remaining worker output so next request stays in sync
                if not finished:
                    while True:
                        line = self.proc.stdout.readline()
                        if not line:
                            break
                        s = line.strip()
                        if s.startswith("DONE ") or s.startswith("ERR "):
                            break


class InklingChatParser:
    """Incrementally parses Inkling-2-Mini token IDs into reasoning_content, content, and tool_calls."""

    def __init__(self, worker: SquidWorker, raw_thinking: bool = False):
        self.w = worker
        self.raw_thinking = raw_thinking
        self.section = "none"  # none | header | thinking | text | tool
        self.sec_ids: List[int] = []
        self.sec_decoded = ""
        self.reasoning_full = ""
        self.content_full = ""
        self.tool_calls: List[Dict[str, Any]] = []
        self.raw_ids: List[int] = []
        self.raw_decoded = ""

    def feed(self, tok_id: int) -> Tuple[str, str]:
        """Returns (delta_reasoning, delta_content) for this token."""
        if tok_id == self.w.id_eos:
            return ("", "")

        if self.raw_thinking:
            self.raw_ids.append(tok_id)
            full = self.w.tok.decode(self.raw_ids, skip_special_tokens=False)
            delta = full[len(self.raw_decoded):]
            self.raw_decoded = full
            self.content_full = full
            return ("", delta)

        if tok_id == self.w.id_msg_model:
            self.section = "header"
            self.sec_ids = []
            self.sec_decoded = ""
            return ("", "")
        if tok_id == self.w.id_thinking:
            self.section = "thinking"
            self.sec_ids = []
            self.sec_decoded = ""
            return ("", "")
        if tok_id == self.w.id_text:
            self.section = "text"
            self.sec_ids = []
            self.sec_decoded = ""
            return ("", "")
        if self.w.id_tool_json is not None and tok_id == self.w.id_tool_json:
            self.section = "tool"
            self.sec_ids = []
            self.sec_decoded = ""
            return ("", "")
        if tok_id == self.w.id_end_msg:
            if self.section == "tool":
                raw_json = self.w.tok.decode(self.sec_ids, skip_special_tokens=True).strip()
                try:
                    obj = json.loads(raw_json)
                    fn_name = obj.get("name", "")
                    fn_args = obj.get("args", obj.get("arguments", {}))
                    self.tool_calls.append(
                        {
                            "id": f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": json.dumps(fn_args, ensure_ascii=False),
                            },
                        }
                    )
                except Exception:
                    pass
            self.section = "none"
            self.sec_ids = []
            self.sec_decoded = ""
            return ("", "")

        if self.section in ("thinking", "text", "none"):
            # Fallback to text if model emits directly without header
            target = "thinking" if self.section == "thinking" else "text"
            self.sec_ids.append(tok_id)
            full = self.w.tok.decode(self.sec_ids, skip_special_tokens=False)
            # Avoid emitting trailing U+FFFD replacement char mid-UTF8 sequence
            if full.endswith("\ufffd"):
                return ("", "")
            delta = full[len(self.sec_decoded):]
            self.sec_decoded = full
            if target == "thinking":
                self.reasoning_full += delta
                return (delta, "")
            self.content_full += delta
            return ("", delta)
        elif self.section == "tool":
            self.sec_ids.append(tok_id)
        return ("", "")


def normalize_stops(stop_val: Any) -> List[str]:
    if not stop_val:
        return []
    if isinstance(stop_val, str):
        return [stop_val] if stop_val else []
    if isinstance(stop_val, list):
        return [s for s in stop_val if isinstance(s, str) and s]
    return []


def check_stop(text: str, stops: List[str]) -> Optional[int]:
    best = None
    for s in stops:
        idx = text.find(s)
        if idx != -1 and (best is None or idx < best):
            best = idx
    return best


def make_handler(worker: SquidWorker, model_name: str, api_key: str, default_effort: str, raw_thinking: bool):
    class OpenAIHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stdout.write(f"[http] {self.address_string()} - {fmt % args}\n")
            sys.stdout.flush()

        def _send_cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _check_auth(self) -> bool:
            if not api_key:
                return True
            hdr = self.headers.get("Authorization", "")
            if hdr == f"Bearer {api_key}":
                return True
            self._send_json(401, {"error": {"message": "Invalid API key", "type": "authentication_error"}})
            return False

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._send_cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "model": model_name,
                        "max_context": worker.max_ctx,
                        "worker": worker.ready_info,
                    },
                )
                return
            if path in ("/v1/models", "/models"):
                if not self._check_auth():
                    return
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": model_name,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "squid5090",
                                "max_model_len": worker.max_ctx,
                            }
                        ],
                    },
                )
                return
            self._send_json(404, {"error": {"message": f"Unknown path: {path}", "type": "invalid_request_error"}})

        def do_POST(self) -> None:
            if not self._check_auth():
                return
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                req = json.loads(raw_body.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request_error"}})
                return

            if path in ("/v1/chat/completions", "/chat/completions"):
                self._handle_chat(req)
            elif path in ("/v1/completions", "/completions"):
                self._handle_completions(req)
            else:
                self._send_json(404, {"error": {"message": f"Unknown path: {path}", "type": "invalid_request_error"}})

        def _write_sse(self, obj: Any) -> None:
            data = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
            chunk = f"data: {data}\n\n".encode("utf-8")
            self.wfile.write(chunk)
            self.wfile.flush()

        def _handle_chat(self, req: Dict[str, Any]) -> None:
            messages = req.get("messages")
            if not isinstance(messages, list) or not messages:
                self._send_json(400, {"error": {"message": "'messages' must be a non-empty list", "type": "invalid_request_error"}})
                return

            max_tokens = req.get("max_completion_tokens") or req.get("max_tokens") or 1024
            max_tokens = max(1, int(max_tokens))
            stream = bool(req.get("stream", False))
            include_usage = bool((req.get("stream_options") or {}).get("include_usage", False))
            stops = normalize_stops(req.get("stop"))
            tools = req.get("tools")
            effort = req.get("reasoning_effort", default_effort)
            req_mtp = req.get("mtp", None)
            req_model = req.get("model") or model_name

            try:
                prompt_ids = worker.render_chat(messages, tools=tools, reasoning_effort=effort)
            except Exception as e:
                self._send_json(400, {"error": {"message": f"Chat template error: {e}", "type": "invalid_request_error"}})
                return

            if len(prompt_ids) >= worker.max_ctx:
                self._send_json(
                    400,
                    {
                        "error": {
                            "message": f"Prompt length ({len(prompt_ids)}) exceeds max_ctx ({worker.max_ctx})",
                            "type": "invalid_request_error",
                        }
                    },
                )
                return

            req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            parser = InklingChatParser(worker, raw_thinking=raw_thinking)

            if not stream:
                stats: Dict[str, Any] = {}
                hit_stop = False
                for kind, val in worker.stream_tokens(prompt_ids, max_tokens, req_mtp):
                    if kind == "err":
                        self._send_json(500, {"error": {"message": str(val), "type": "server_error"}})
                        return
                    elif kind == "tok":
                        if not hit_stop:
                            parser.feed(val)
                            if stops and check_stop(parser.content_full, stops) is not None:
                                idx = check_stop(parser.content_full, stops)
                                assert idx is not None
                                parser.content_full = parser.content_full[:idx]
                                hit_stop = True
                    elif kind == "done":
                        stats = val

                finish = "tool_calls" if parser.tool_calls else ("stop" if hit_stop else stats.get("finish_reason", "stop"))
                msg: Dict[str, Any] = {"role": "assistant", "content": parser.content_full if parser.content_full or not parser.tool_calls else None}
                if parser.reasoning_full:
                    msg["reasoning_content"] = parser.reasoning_full
                if parser.tool_calls:
                    msg["tool_calls"] = parser.tool_calls

                self._send_json(
                    200,
                    {
                        "id": req_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": req_model,
                        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                        "usage": {
                            "prompt_tokens": stats.get("prompt_tokens", len(prompt_ids)),
                            "completion_tokens": stats.get("completion_tokens", 0),
                            "total_tokens": stats.get("prompt_tokens", len(prompt_ids)) + stats.get("completion_tokens", 0),
                        },
                        "timings": {
                            "prefill_ms": stats.get("prefill_ms", 0.0),
                            "decode_ms": stats.get("decode_ms", 0.0),
                            "tokens_per_second": stats.get("tokens_per_second", 0.0),
                            "spec_acc_pct": stats.get("spec_acc_pct", 0.0),
                            "spec_dl": stats.get("spec_dl", 1.0),
                        },
                    },
                )
                return

            # Streaming SSE response
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            stream_iter = worker.stream_tokens(prompt_ids, max_tokens, req_mtp)
            try:
                # Initial role chunk
                self._write_sse(
                    {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req_model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                )
                stats = {}
                hit_stop = False
                emitted_content_len = 0
                max_stop_len = max((len(s) for s in stops), default=0)

                for kind, val in stream_iter:
                    if kind == "err":
                        break
                    elif kind == "tok":
                        if hit_stop:
                            continue
                        d_reason, d_content = parser.feed(val)
                        if d_reason:
                            self._write_sse(
                                {
                                    "id": req_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": req_model,
                                    "choices": [{"index": 0, "delta": {"reasoning_content": d_reason}, "finish_reason": None}],
                                }
                            )
                        if d_content:
                            if stops:
                                idx = check_stop(parser.content_full, stops)
                                if idx is not None:
                                    safe = parser.content_full[emitted_content_len:idx]
                                    if safe:
                                        self._write_sse(
                                            {
                                                "id": req_id,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": req_model,
                                                "choices": [{"index": 0, "delta": {"content": safe}, "finish_reason": None}],
                                            }
                                        )
                                    emitted_content_len = idx
                                    hit_stop = True
                                else:
                                    # Hold back up to max_stop_len-1 chars to avoid leaking a partial stop string
                                    safe_end = max(emitted_content_len, len(parser.content_full) - (max_stop_len - 1))
                                    safe = parser.content_full[emitted_content_len:safe_end]
                                    if safe:
                                        self._write_sse(
                                            {
                                                "id": req_id,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": req_model,
                                                "choices": [{"index": 0, "delta": {"content": safe}, "finish_reason": None}],
                                            }
                                        )
                                        emitted_content_len = safe_end
                            else:
                                self._write_sse(
                                    {
                                        "id": req_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": req_model,
                                        "choices": [{"index": 0, "delta": {"content": d_content}, "finish_reason": None}],
                                    }
                                )
                                emitted_content_len = len(parser.content_full)
                    elif kind == "done":
                        stats = val

                # Flush any held-back non-stop suffix
                if stops and not hit_stop and emitted_content_len < len(parser.content_full):
                    rem = parser.content_full[emitted_content_len:]
                    if rem:
                        self._write_sse(
                            {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": req_model,
                                "choices": [{"index": 0, "delta": {"content": rem}, "finish_reason": None}],
                            }
                        )

                # Emit tool calls if any
                if parser.tool_calls:
                    tc_deltas = [{"index": i, **tc} for i, tc in enumerate(parser.tool_calls)]
                    self._write_sse(
                        {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": req_model,
                            "choices": [{"index": 0, "delta": {"tool_calls": tc_deltas}, "finish_reason": None}],
                        }
                    )

                finish = "tool_calls" if parser.tool_calls else ("stop" if hit_stop else stats.get("finish_reason", "stop"))
                final_chunk: Dict[str, Any] = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                }
                if include_usage:
                    p_toks = stats.get("prompt_tokens", len(prompt_ids))
                    c_toks = stats.get("completion_tokens", 0)
                    final_chunk["usage"] = {
                        "prompt_tokens": p_toks,
                        "completion_tokens": c_toks,
                        "total_tokens": p_toks + c_toks,
                    }
                self._write_sse(final_chunk)
                self._write_sse("[DONE]")
            except BrokenPipeError:
                pass
            finally:
                stream_iter.close()

        def _handle_completions(self, req: Dict[str, Any]) -> None:
            prompt = req.get("prompt", "")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            if not isinstance(prompt, str) or not prompt:
                self._send_json(400, {"error": {"message": "'prompt' must be a non-empty string", "type": "invalid_request_error"}})
                return

            max_tokens = max(1, int(req.get("max_tokens", 256)))
            stream = bool(req.get("stream", False))
            echo = bool(req.get("echo", False))
            stops = normalize_stops(req.get("stop"))
            req_mtp = req.get("mtp", None)
            req_model = req.get("model") or model_name

            prompt_ids = worker.encode_raw(prompt)
            if len(prompt_ids) >= worker.max_ctx:
                self._send_json(
                    400,
                    {"error": {"message": f"Prompt length ({len(prompt_ids)}) exceeds max_ctx ({worker.max_ctx})", "type": "invalid_request_error"}},
                )
                return

            req_id = f"cmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            gen_ids: List[int] = []
            decoded = ""

            if not stream:
                stats: Dict[str, Any] = {}
                hit_stop = False
                for kind, val in worker.stream_tokens(prompt_ids, max_tokens, req_mtp):
                    if kind == "err":
                        self._send_json(500, {"error": {"message": str(val), "type": "server_error"}})
                        return
                    elif kind == "tok":
                        if not hit_stop and val != worker.id_eos:
                            gen_ids.append(val)
                            decoded = worker.tok.decode(gen_ids, skip_special_tokens=True)
                            if stops:
                                idx = check_stop(decoded, stops)
                                if idx is not None:
                                    decoded = decoded[:idx]
                                    hit_stop = True
                    elif kind == "done":
                        stats = val
                finish = "stop" if hit_stop else stats.get("finish_reason", "stop")
                out_text = (prompt + decoded) if echo else decoded
                p_toks = stats.get("prompt_tokens", len(prompt_ids))
                c_toks = stats.get("completion_tokens", len(gen_ids))
                self._send_json(
                    200,
                    {
                        "id": req_id,
                        "object": "text_completion",
                        "created": created,
                        "model": req_model,
                        "choices": [{"index": 0, "text": out_text, "logprobs": None, "finish_reason": finish}],
                        "usage": {"prompt_tokens": p_toks, "completion_tokens": c_toks, "total_tokens": p_toks + c_toks},
                        "timings": {
                            "prefill_ms": stats.get("prefill_ms", 0.0),
                            "decode_ms": stats.get("decode_ms", 0.0),
                            "tokens_per_second": stats.get("tokens_per_second", 0.0),
                        },
                    },
                )
                return

            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            stream_iter = worker.stream_tokens(prompt_ids, max_tokens, req_mtp)
            try:
                stats = {}
                hit_stop = False
                for kind, val in stream_iter:
                    if kind == "err":
                        break
                    elif kind == "tok":
                        if hit_stop or val == worker.id_eos:
                            continue
                        gen_ids.append(val)
                        full = worker.tok.decode(gen_ids, skip_special_tokens=True)
                        if full.endswith("\ufffd"):
                            continue
                        if stops:
                            idx = check_stop(full, stops)
                            if idx is not None:
                                delta = full[len(decoded):idx]
                                if delta:
                                    self._write_sse(
                                        {
                                            "id": req_id,
                                            "object": "text_completion",
                                            "created": created,
                                            "model": req_model,
                                            "choices": [{"index": 0, "text": delta, "logprobs": None, "finish_reason": None}],
                                        }
                                    )
                                hit_stop = True
                                continue
                        delta = full[len(decoded):]
                        decoded = full
                        if delta:
                            self._write_sse(
                                {
                                    "id": req_id,
                                    "object": "text_completion",
                                    "created": created,
                                    "model": req_model,
                                    "choices": [{"index": 0, "text": delta, "logprobs": None, "finish_reason": None}],
                                }
                            )
                    elif kind == "done":
                        stats = val
                finish = "stop" if hit_stop else stats.get("finish_reason", "stop")
                self._write_sse(
                    {
                        "id": req_id,
                        "object": "text_completion",
                        "created": created,
                        "model": req_model,
                        "choices": [{"index": 0, "text": "", "logprobs": None, "finish_reason": finish}],
                    }
                )
                self._write_sse("[DONE]")
            except BrokenPipeError:
                pass
            finally:
                stream_iter.close()

    return OpenAIHandler


def main() -> None:
    ap = argparse.ArgumentParser(description="squid5090 OpenAI-compatible API server")
    ap.add_argument("model_dir", help="Path to Inkling-2-Mini model directory")
    ap.add_argument("--bin", default=str(Path(__file__).resolve().parent.parent / "build" / "squid5090"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-name", default="inkling-2-mini")
    ap.add_argument("--mode", default="nvfp4", choices=["exact", "fp8", "nvfp4"])
    ap.add_argument("--kv", default="auto", choices=["auto", "bf16", "fp8", "q4"])
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--mtp", type=int, default=3)
    ap.add_argument("--effort", default="none", help="Default reasoning_effort (none|minimal|low|medium|high)")
    ap.add_argument("--api-key", default=os.environ.get("SQUID_API_KEY", ""))
    ap.add_argument("--raw-thinking", action="store_true", help="Emit raw special tags in content instead of splitting reasoning_content")
    ap.add_argument("--embed-host", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()

    worker = SquidWorker(
        bin_path=args.bin,
        model_dir=Path(args.model_dir),
        mode=args.mode,
        kv=args.kv,
        ctx=args.ctx,
        mtp=args.mtp,
        embed_host=args.embed_host,
        no_graph=args.no_graph,
    )

    handler_cls = make_handler(worker, args.model_name, args.api_key, args.effort, args.raw_thinking)
    httpd = ThreadingHTTPServer((args.host, args.port), handler_cls)

    def shutdown_handler(_signum: int, _frame: Any) -> None:
        print("\n[server] Shutting down...", flush=True)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print(f"[server] OpenAI API ready on http://{args.host}:{args.port}/v1 (model={args.model_name})", flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        worker.close()


if __name__ == "__main__":
    main()
