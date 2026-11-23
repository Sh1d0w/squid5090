#!/usr/bin/env python3
import argparse, json, os, re, subprocess, sys, tempfile
from pathlib import Path
from tokenizers import Tokenizer
from jinja2 import Environment

def encode_prompt(model_dir: Path, prompt: str, reasoning_effort: str = "none"):
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tpl_str = (model_dir / "chat_template.jinja").read_text()
    tpl_str = re.sub(r"\{%-?\s*(?:end)?generation\s*-?%\}", "", tpl_str)
    env = Environment()
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    tpl = env.from_string(tpl_str)
    rendered = tpl.render(
        messages=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )
    ids = tok.encode(rendered, add_special_tokens=False).ids
    return tok, ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("prompt")
    ap.add_argument("--bin", default="/opt/squid5090/build/squid5090")
    ap.add_argument("--cmd", default="generate", choices=["generate", "bench", "verify", "profile"])
    ap.add_argument("--mode", default="exact", choices=["exact", "fp8", "nvfp4"])
    ap.add_argument("--kv", default="auto", choices=["auto", "bf16", "fp8", "q4"])
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--mtp", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--effort", default="none")
    ap.add_argument("--embed-host", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()

    tok, ids = encode_prompt(Path(args.model_dir), args.prompt, args.effort)
    tmp_tok_path = None
    if len(ids) > 4096:
        fd, tmp_tok_path = tempfile.mkstemp(prefix="squid_toks_", suffix=".csv")
        with os.fdopen(fd, "w") as f:
            f.write(",".join(str(x) for x in ids))
        tok_args = ["--tokens-file", tmp_tok_path]
    else:
        tok_args = ["--tokens", ",".join(str(x) for x in ids)]

    cmd = [
        args.bin, args.cmd, args.model_dir,
        *tok_args,
        "--mode", args.mode,
        "--kv", args.kv,
        "--ctx", str(args.ctx),
        "--mtp", str(args.mtp),
        "--max-tokens", str(args.max_tokens),
        "--runs", str(args.runs),
    ]
    if args.embed_host:
        cmd.append("--embed-host")
    if args.no_graph:
        cmd.append("--no-graph")

    try:
        proc = subprocess.run(cmd, text=True, capture_output=True)
    finally:
        if tmp_tok_path and os.path.exists(tmp_tok_path):
            os.remove(tmp_tok_path)
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

    for line in proc.stdout.splitlines():
        if line.startswith("TOKENS:"):
            out_ids = [int(x) for x in line.split(":", 1)[1].strip().split(",") if x.strip()]
            text = tok.decode(out_ids, skip_special_tokens=False)
            print("\n--- DECODED OUTPUT ---")
            print(text)
            print("----------------------")

if __name__ == "__main__":
    main()
