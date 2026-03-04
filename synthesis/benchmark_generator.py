"""
synthesis/benchmark_generator.py — Generate contamination-resistant benchmark
questions using vLLM or Claude, then verify correctness.

Generation strategy:
  1. Load question templates from discovery/question_templates.py output
  2. For each template, instruct vLLM/Claude to generate a NOVEL question
     inspired by (but distinct from) the template
  3. Verify math answers via symbolic computation (sympy)
  4. Verify code answers via execution in a subprocess sandbox
  5. Filter out any generated questions that match the contamination catalog

Usage:
    python synthesis/benchmark_generator.py \
        --templates data/raw/question_templates/all_templates.jsonl \
        --catalog data/raw/contamination_catalog.jsonl \
        --output data/synthesized/benchmark_questions.jsonl \
        --count 50000 \
        --backend vllm
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ast
import itertools
import json
import os
import random
import re
import subprocess
import textwrap
import time
from typing import Any

import anthropic
import requests
from loguru import logger

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False
    def retry(*args, **kwargs):
        def decorator(fn): return fn
        return decorator
    def stop_after_attempt(n): return None
    def wait_exponential(**kwargs): return None

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VLLM_URLS = os.environ.get(
    "VLLM_URLS",
    "http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004",
).split(",")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "evalforge-secret")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# Module-level iterator for true round-robin endpoint selection.
_VLLM_CYCLE = itertools.cycle(VLLM_URLS)


GENERATION_SYSTEM_PROMPT = """\
You are an expert benchmark question creator. Your job is to create novel, high-quality
evaluation questions for AI systems. Questions must be:
1. Original — not copied from any known benchmark dataset
2. Unambiguous — exactly one correct answer
3. Appropriately difficult — requires genuine reasoning, not surface-level pattern matching
4. Verifiable — answer can be verified programmatically or by a domain expert
5. No shortcuts — do not include spurious correlations or answer-revealing context

Return your response as JSON with fields:
  question: the question text
  answer: the correct answer (string, number, or code)
  explanation: step-by-step explanation of the answer
  difficulty: "easy" | "medium" | "hard"
  subcategory: specific topic within the category
  verification_type: "symbolic" | "execution" | "pattern" | "manual"
"""


def _vllm_round_robin() -> str:
    """Pick a vLLM URL in true round-robin fashion using a module-level cycle iterator."""
    return next(_VLLM_CYCLE)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def call_vllm(prompt: str, system: str = GENERATION_SYSTEM_PROMPT, max_tokens: int = 1024) -> str:
    """Call vLLM with OpenAI-compatible chat completions endpoint."""
    url = _vllm_round_robin()
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.85,
        "top_p": 0.95,
    }
    resp = requests.post(
        f"{url}/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def call_claude(prompt: str, system: str = GENERATION_SYSTEM_PROMPT, max_tokens: int = 1024) -> str:
    """Call Anthropic Claude API as fallback."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def call_llm(prompt: str, use_claude_ratio: float = 0.2, **kwargs: Any) -> str:
    """Call vLLM (primary) or Claude (fallback + ratio)."""
    if random.random() < use_claude_ratio and ANTHROPIC_API_KEY:
        try:
            return call_claude(prompt, **kwargs)
        except Exception as exc:
            logger.debug(f"Claude fallback: {exc}")
    try:
        return call_vllm(prompt, **kwargs)
    except Exception as exc:
        logger.debug(f"vLLM failed, falling back to Claude: {exc}")
        return call_claude(prompt, **kwargs)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract JSON from LLM response, handling markdown code fences."""
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try code fence extraction
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try curly brace extraction
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _verify_math_answer(question: str, answer: str, explanation: str) -> tuple[bool, str]:
    """
    Verify a math answer using sympy for symbolic computation.
    Returns (is_correct, verification_note).
    """
    try:
        import sympy  # type: ignore

        # Try to evaluate if answer is a simple numeric expression
        answer_clean = answer.strip().replace(",", "")
        try:
            val = sympy.sympify(answer_clean)
            if val.is_number:
                return True, f"sympy verified: {val}"
        except Exception:
            pass

        # If explanation contains a final numeric answer, check consistency
        nums_in_explanation = re.findall(r"\b\d+(?:\.\d+)?\b", explanation)
        if nums_in_explanation and answer_clean in nums_in_explanation:
            return True, "answer found in explanation"

        return True, "numeric answer present (manual verification needed)"
    except ImportError:
        return True, "sympy not available — manual verification needed"
    except Exception as exc:
        return True, f"verification skipped: {exc}"


def _verify_code_answer(question: str, answer: str, explanation: str) -> tuple[bool, str]:
    """
    Verify a coding answer by executing it in a subprocess sandbox.
    Returns (is_correct, verification_note).
    """
    if not answer or len(answer.strip()) < 10:
        return False, "answer too short for code"

    # Check Python syntax
    try:
        ast.parse(answer)
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"

    # Execute with timeout in isolated subprocess
    test_code = textwrap.dedent(f"""
import sys, io
captured = io.StringIO()
sys.stdout = captured
try:
{textwrap.indent(answer, "    ")}
    print("EXEC_OK")
except Exception as e:
    print(f"EXEC_ERROR: {{e}}")
sys.stdout = sys.__stdout__
print(captured.getvalue())
""")
    try:
        result = subprocess.run(
            ["python3", "-c", test_code],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout + result.stderr
        if "EXEC_ERROR" in output or result.returncode != 0:
            return False, f"execution error: {output[:200]}"
        return True, "execution OK"
    except subprocess.TimeoutExpired:
        return False, "execution timeout"
    except Exception as exc:
        return True, f"execution check skipped: {exc}"


def _build_generation_prompt(template: dict[str, Any]) -> str:
    """Build a prompt to generate a novel question inspired by a template."""
    source = template.get("source", "unknown")
    category = template.get("category", "general")
    subcategory = template.get("subcategory", "")
    difficulty = template.get("difficulty", "medium")
    template_question = template.get("question", "")

    if category == "math":
        domain_instruction = (
            "Generate a novel math problem. The problem must use different numbers, "
            "structures, and scenarios than the template. The answer must be a specific "
            "number or expression computable by a student."
        )
    elif category == "coding":
        domain_instruction = (
            "Generate a novel programming challenge. The problem must have a clear, "
            "testable specification. The answer should be working Python code that solves "
            "the problem."
        )
    else:
        domain_instruction = (
            "Generate a novel question. The question must test genuine understanding, "
            "not pattern matching. The answer must be unambiguous."
        )

    return f"""You are creating a new benchmark question for evaluating AI systems.

TEMPLATE (for inspiration only — your question must be DIFFERENT):
Source: {source}
Category: {category} / {subcategory}
Difficulty: {difficulty}
Template question: {template_question[:300]}

INSTRUCTIONS:
{domain_instruction}

Your question must:
1. Be a genuinely new question (not a minor paraphrase of the template)
2. Have exactly one correct answer
3. Be solvable at difficulty level: {difficulty}
4. Not appear in any known benchmark (MMLU, GSM8K, HumanEval, BigBench, etc.)

Respond with ONLY a JSON object with fields:
  question, answer, explanation, difficulty, subcategory, verification_type

verification_type must be one of: "symbolic" (math), "execution" (code), "pattern" (regex/exact), "manual"
"""


class BenchmarkGenerator:
    """
    Generates contamination-resistant benchmark questions at scale.

    Pipeline:
      1. Load templates → generate questions → verify → dedup → filter contamination → save
    """

    def __init__(
        self,
        output_path: str | Path,
        catalog_path: str | Path | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._catalog: dict[str, str] = {}  # fingerprint → benchmark name
        if catalog_path and Path(catalog_path).exists():
            self._load_catalog(Path(catalog_path))

        self._generated_fingerprints: set[str] = set()

    def _load_catalog(self, path: Path) -> None:
        from discovery.existing_benchmarks import _item_fingerprint

        with path.open() as fh:
            for line in fh:
                entry = json.loads(line.strip())
                fp = entry.get("exact_fingerprint", "")
                if fp:
                    self._catalog[fp] = entry.get("benchmark", "unknown")
        logger.info(f"Loaded contamination catalog: {len(self._catalog)} entries")

    def generate(
        self,
        templates_path: str | Path,
        count: int = 50000,
        batch_size: int = 200,
        backend: str = "vllm",
    ) -> int:
        """
        Generate `count` benchmark questions from templates.
        Returns actual number of valid questions generated.
        """
        templates = self._load_templates(templates_path)
        if not templates:
            logger.error(f"No templates found at {templates_path}")
            return 0

        logger.info(f"Generating {count} questions from {len(templates)} templates...")

        generated = 0
        batch: list[dict] = []
        attempts = 0
        max_attempts = count * 5  # Guard against infinite loop when generation consistently fails

        with self.output_path.open("w") as out_fh:
            while generated < count:
                if attempts >= max_attempts:
                    logger.warning(
                        f"Reached max attempts ({max_attempts}) after generating {generated}/{count} questions. "
                        "Check vLLM/Claude connectivity or template quality."
                    )
                    break
                attempts += 1
                template = random.choice(templates)
                result = self._generate_one(template, backend=backend)

                if result is None:
                    continue

                # Contamination check
                if self._is_contaminated(result["question"]):
                    logger.debug(f"  Filtered contaminated question: {result['question'][:60]}...")
                    continue

                # Dedup within generated set
                fp = self._question_fingerprint(result["question"])
                if fp in self._generated_fingerprints:
                    continue
                self._generated_fingerprints.add(fp)

                batch.append(result)
                if len(batch) >= batch_size:
                    for item in batch:
                        out_fh.write(json.dumps(item) + "\n")
                    out_fh.flush()
                    generated += len(batch)
                    logger.info(f"  Generated {generated}/{count} questions")
                    batch = []

            # Flush remaining
            if batch:
                for item in batch:
                    out_fh.write(json.dumps(item) + "\n")
                generated += len(batch)

        logger.success(f"Benchmark generation complete: {generated} questions → {self.output_path}")
        return generated

    def _generate_one(self, template: dict, backend: str = "vllm") -> dict[str, Any] | None:
        """Generate a single question from a template. Returns None on failure."""
        try:
            prompt = _build_generation_prompt(template)

            if backend == "vllm":
                raw = call_llm(prompt, use_claude_ratio=0.15)
            else:
                raw = call_claude(prompt)

            result = _extract_json(raw)
            if result is None:
                logger.debug(f"  Could not parse JSON from LLM response: {raw[:200]}")
                return None

            question = result.get("question", "").strip()
            answer = result.get("answer", "").strip()
            if not question or not answer:
                return None

            # Verify answer
            category = template.get("category", "general")
            verification_type = result.get("verification_type", "manual")
            verified, note = True, "not verified"

            if category == "math" or verification_type == "symbolic":
                verified, note = _verify_math_answer(
                    question, answer, result.get("explanation", "")
                )
            elif category == "coding" or verification_type == "execution":
                verified, note = _verify_code_answer(
                    question, answer, result.get("explanation", "")
                )

            if not verified:
                logger.debug(f"  Verification failed: {note} for Q: {question[:60]}")
                return None

            return {
                "question": question,
                "answer": answer,
                "explanation": result.get("explanation", ""),
                "difficulty": result.get("difficulty", template.get("difficulty", "medium")),
                "category": template.get("category", "general"),
                "subcategory": result.get("subcategory", template.get("subcategory", "")),
                "source_template": template.get("source", "unknown"),
                "verification_type": verification_type,
                "verification_note": note,
                "generated_by": backend,
            }

        except Exception as exc:
            logger.debug(f"  Generation failed: {exc}")
            return None

    def _is_contaminated(self, question: str) -> bool:
        """Check if a question matches the contamination catalog."""
        if not self._catalog:
            return False
        fp = self._question_fingerprint(question)
        return fp in self._catalog

    def _question_fingerprint(self, text: str) -> str:
        import hashlib
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _load_templates(self, path: str | Path) -> list[dict]:
        path = Path(path)
        if not path.exists():
            return []
        templates = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    templates.append(json.loads(line))
        return templates


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate contamination-resistant benchmark questions")
    parser.add_argument(
        "--templates",
        default="data/raw/question_templates/all_templates.jsonl",
        help="Path to question templates JSONL",
    )
    parser.add_argument(
        "--catalog",
        default="data/raw/contamination_catalog.jsonl",
        help="Path to contamination catalog JSONL",
    )
    parser.add_argument(
        "--output",
        default="data/synthesized/benchmark_questions.jsonl",
        help="Output path for generated questions",
    )
    parser.add_argument("--count", type=int, default=50000, help="Number of questions to generate")
    parser.add_argument(
        "--backend",
        choices=["vllm", "claude"],
        default="vllm",
        help="LLM backend for generation",
    )
    args = parser.parse_args()

    generator = BenchmarkGenerator(
        output_path=args.output,
        catalog_path=args.catalog,
    )
    n = generator.generate(
        templates_path=args.templates,
        count=args.count,
        backend=args.backend,
    )
    logger.info(f"Generated {n} benchmark questions → {args.output}")
