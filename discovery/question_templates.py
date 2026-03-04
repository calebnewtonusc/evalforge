"""
discovery/question_templates.py — Pull question templates from math and coding
contest sites for contamination-resistant benchmark generation.

Sources:
  - Art of Problem Solving (AoPS) — math competition problems
  - Project Euler — mathematical programming problems
  - Rosetta Code — programming task templates
  - HackerRank — algorithmic challenge templates

Usage:
    python discovery/question_templates.py \
        --output data/raw/question_templates \
        --sources aops euler rosetta hackerrank
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from loguru import logger

try:
    from tenacity import retry, stop_after_attempt, wait_exponential

    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False

    def retry(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    def stop_after_attempt(n):
        return None

    def wait_exponential(**kwargs):
        return None


ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VLLM_URLS = os.environ.get(
    "VLLM_URLS",
    "http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004",
).split(",")

REQUEST_TIMEOUT = 20
REQUEST_DELAY = 0.8  # seconds between requests to each site

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "EvalForge-Research/1.0 (+https://github.com/evalforge)"
    )
}

# Project Euler API endpoint
EULER_API = "https://projecteuler.net/minimal=problems"
# Rosetta Code API endpoint
ROSETTA_API = "https://rosettacode.org/mw/api.php"


def _clean_text(text: str) -> str:
    """Remove excess whitespace and normalize."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _fetch(
    url: str, params: dict | None = None, timeout: int = REQUEST_TIMEOUT
) -> requests.Response:
    """HTTP GET with retry."""
    resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp


class AoPSCrawler:
    """
    Crawl Art of Problem Solving (AoPS) for math competition problems.

    Sources: AMC 8, AMC 10, AMC 12, AIME, MATHCOUNTS, HMMT, USA(J)MO
    Uses the AoPS alcumus API endpoint and public problem pages.
    """

    ALCUMUS_URL = "https://artofproblemsolving.com/wiki/index.php"
    PROBLEM_SETS = [
        ("AMC_8_2023_Problems", "AMC 8 2023"),
        ("AMC_10A_2023_Problems", "AMC 10A 2023"),
        ("AMC_10B_2023_Problems", "AMC 10B 2023"),
        ("AMC_12A_2023_Problems", "AMC 12A 2023"),
        ("AMC_12B_2023_Problems", "AMC 12B 2023"),
        ("2023_AIME_I_Problems", "AIME I 2023"),
        ("2023_AIME_II_Problems", "AIME II 2023"),
        ("AMC_8_2022_Problems", "AMC 8 2022"),
        ("AMC_10A_2022_Problems", "AMC 10A 2022"),
        ("AMC_10B_2022_Problems", "AMC 10B 2022"),
        ("AMC_12A_2022_Problems", "AMC 12A 2022"),
        ("AMC_12B_2022_Problems", "AMC 12B 2022"),
        ("2022_AIME_I_Problems", "AIME I 2022"),
        ("2022_AIME_II_Problems", "AIME II 2022"),
        ("AMC_8_2020_Problems", "AMC 8 2020"),
        ("AMC_10A_2020_Problems", "AMC 10A 2020"),
        ("AMC_10B_2020_Problems", "AMC 10B 2020"),
    ]

    def crawl(self, output_dir: Path, max_problems: int = 2000) -> list[dict]:
        """Fetch math competition problems from AoPS wiki."""
        problems: list[dict] = []

        for page_name, contest_name in self.PROBLEM_SETS:
            if len(problems) >= max_problems:
                break
            try:
                batch = self._fetch_problem_set(page_name, contest_name)
                problems.extend(batch)
                logger.debug(f"  AoPS {contest_name}: {len(batch)} problems")
                time.sleep(REQUEST_DELAY)
            except Exception as exc:
                logger.warning(f"  AoPS {contest_name} failed: {exc}")

        return problems[:max_problems]

    def _fetch_problem_set(self, page_name: str, contest_name: str) -> list[dict]:
        """Fetch a single problem set page and extract numbered problems."""
        resp = _fetch(
            self.ALCUMUS_URL,
            params={"title": page_name, "action": "raw"},
        )
        text = resp.text

        # Extract Problem N patterns from wiki markup
        problems: list[dict] = []
        problem_pattern = re.compile(
            r"==\s*Problem\s+(\d+)\s*==(.*?)(?===\s*Problem\s+\d+\s*==|==\s*Solution|$)",
            re.DOTALL | re.IGNORECASE,
        )

        for match in problem_pattern.finditer(text):
            problem_num = match.group(1)
            problem_text = match.group(2).strip()
            # Remove wiki markup artifacts
            problem_text = re.sub(r"\[\[.*?\]\]", "", problem_text)
            problem_text = re.sub(r"\{\{.*?\}\}", "", problem_text)
            problem_text = re.sub(r"<[^>]+>", "", problem_text)
            problem_text = _clean_text(problem_text)

            if len(problem_text) < 20:
                continue

            # Determine difficulty
            if "AIME" in contest_name:
                difficulty = "hard"
            elif "AMC 12" in contest_name:
                difficulty = "medium-hard"
            elif "AMC 10" in contest_name:
                difficulty = "medium"
            else:
                difficulty = "easy-medium"

            problems.append(
                {
                    "source": "aops",
                    "contest": contest_name,
                    "problem_number": int(problem_num),
                    "question": problem_text,
                    "category": "math",
                    "subcategory": self._classify_math(problem_text),
                    "difficulty": difficulty,
                    "answer": None,  # Answers require separate page
                    "url": f"https://artofproblemsolving.com/wiki/index.php/{page_name}",
                }
            )

        return problems

    def _classify_math(self, text: str) -> str:
        """Classify math problem subcategory from problem text."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["probability", "expected value", "random"]):
            return "probability"
        if any(
            w in text_lower
            for w in ["prime", "divisible", "modulo", "integer", "digit"]
        ):
            return "number_theory"
        if any(
            w in text_lower
            for w in ["triangle", "circle", "area", "perimeter", "angle"]
        ):
            return "geometry"
        if any(w in text_lower for w in ["sequence", "sum", "series", "term"]):
            return "sequences"
        if any(
            w in text_lower for w in ["polynomial", "equation", "roots", "function"]
        ):
            return "algebra"
        if any(
            w in text_lower for w in ["combination", "permutation", "arrange", "choose"]
        ):
            return "combinatorics"
        return "general"


class ProjectEulerCrawler:
    """
    Crawl Project Euler problems.
    Problems 1-800+ are publicly available.
    """

    def crawl(self, output_dir: Path, max_problems: int = 400) -> list[dict]:
        """Fetch Project Euler problem list and descriptions."""
        problems: list[dict] = []
        try:
            # Project Euler provides a minimal list in text format
            resp = _fetch("https://projecteuler.net/minimal=problems")
            lines = resp.text.strip().split("\n")

            for line in lines[:max_problems]:
                parts = line.split("##")
                if len(parts) >= 3:
                    try:
                        prob_id = int(parts[0].strip())
                        title = parts[1].strip()
                        # Content not directly available in minimal format
                        problems.append(
                            {
                                "source": "project_euler",
                                "problem_id": prob_id,
                                "title": title,
                                "question": f"Project Euler Problem {prob_id}: {title}",
                                "category": "math",
                                "subcategory": "computational_math",
                                "difficulty": "hard" if prob_id > 100 else "medium",
                                "answer": None,
                                "url": f"https://projecteuler.net/problem={prob_id}",
                            }
                        )
                    except (ValueError, IndexError):
                        continue

            logger.debug(f"  Project Euler: {len(problems)} problems from minimal API")
            return problems

        except Exception as exc:
            logger.warning(f"  Project Euler minimal API failed: {exc}")

        # Fallback: crawl individual problem pages
        return self._crawl_individual(max_problems)

    def _crawl_individual(self, max_problems: int) -> list[dict]:
        """Crawl individual Project Euler problem pages."""
        problems: list[dict] = []
        for prob_id in range(1, min(max_problems + 1, 200)):
            try:
                resp = _fetch(f"https://projecteuler.net/problem={prob_id}")
                soup = BeautifulSoup(resp.text, "html.parser")
                problem_content = soup.find("div", {"class": "problem_content"})
                if not problem_content:
                    continue

                text = _clean_text(problem_content.get_text())
                if len(text) < 20:
                    continue

                problems.append(
                    {
                        "source": "project_euler",
                        "problem_id": prob_id,
                        "title": f"Problem {prob_id}",
                        "question": text,
                        "category": "math",
                        "subcategory": "computational_math",
                        "difficulty": "hard" if prob_id > 100 else "medium",
                        "answer": None,
                        "url": f"https://projecteuler.net/problem={prob_id}",
                    }
                )

                time.sleep(REQUEST_DELAY * 2)  # Polite
            except Exception as exc:
                logger.debug(f"  Euler problem {prob_id}: {exc}")
                break

        return problems


class RosettaCodeCrawler:
    """
    Crawl Rosetta Code for programming task templates.
    Uses the MediaWiki API to list tasks by category.
    """

    CATEGORIES = [
        "Sorting_algorithms",
        "String_operations",
        "Mathematical_problems",
        "Recursion",
        "Dynamic_programming",
        "Trees",
        "Graphs",
        "Data_structures",
        "Pattern_matching",
        "Arithmetic",
        "Number_theory",
        "Search_algorithms",
        "Array_manipulation",
    ]

    def crawl(self, output_dir: Path, max_tasks: int = 500) -> list[dict]:
        """Fetch programming task descriptions from Rosetta Code."""
        tasks: list[dict] = []

        for category in self.CATEGORIES:
            if len(tasks) >= max_tasks:
                break
            try:
                batch = self._fetch_category(category, limit=50)
                tasks.extend(batch)
                logger.debug(f"  Rosetta Code {category}: {len(batch)} tasks")
                time.sleep(REQUEST_DELAY)
            except Exception as exc:
                logger.warning(f"  Rosetta Code {category}: {exc}")

        return tasks[:max_tasks]

    def _fetch_category(self, category: str, limit: int = 50) -> list[dict]:
        """Fetch task list for a category from Rosetta Code MediaWiki API."""
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": limit,
            "format": "json",
        }
        resp = _fetch(ROSETTA_API, params=params)
        data = resp.json()

        tasks: list[dict] = []
        members = data.get("query", {}).get("categorymembers", [])

        for member in members[:limit]:
            title = member.get("title", "")
            if (
                not title
                or title.startswith("Category:")
                or title.startswith("Template:")
            ):
                continue

            tasks.append(
                {
                    "source": "rosetta_code",
                    "task_name": title,
                    "question": f"Write a program to: {title}",
                    "description": f"Implement the following programming task: {title}. "
                    f"This is a standard programming challenge from the Rosetta Code collection.",
                    "category": "coding",
                    "subcategory": category.lower().replace("_", " "),
                    "difficulty": "medium",
                    "answer": None,
                    "url": f"https://rosettacode.org/wiki/{title.replace(' ', '_')}",
                }
            )

        return tasks


class HackerRankCrawler:
    """
    Fetch publicly available HackerRank challenge templates.
    Uses the public HackerRank API for challenge metadata.
    """

    TRACKS = [
        {"slug": "algorithms", "category": "algorithms"},
        {"slug": "data-structures", "category": "data_structures"},
        {"slug": "mathematics", "category": "math"},
        {"slug": "artificial-intelligence", "category": "ml"},
        {"slug": "python", "category": "coding"},
    ]

    HR_API_BASE = "https://www.hackerrank.com/rest/contests/master/tracks"

    def crawl(self, output_dir: Path, max_challenges: int = 500) -> list[dict]:
        """Fetch challenge metadata from HackerRank."""
        challenges: list[dict] = []

        for track in self.TRACKS:
            if len(challenges) >= max_challenges:
                break
            try:
                batch = self._fetch_track(track["slug"], track["category"])
                challenges.extend(batch)
                logger.debug(f"  HackerRank {track['slug']}: {len(batch)} challenges")
                time.sleep(REQUEST_DELAY)
            except Exception as exc:
                logger.warning(f"  HackerRank {track['slug']}: {exc}")

        return challenges[:max_challenges]

    def _fetch_track(self, slug: str, category: str, limit: int = 100) -> list[dict]:
        """Fetch challenges for a HackerRank track."""
        url = f"{self.HR_API_BASE}/{slug}/challenges"
        try:
            resp = _fetch(url, params={"limit": limit, "offset": 0})
            data = resp.json()
        except Exception:
            # HackerRank may block; try alternate public endpoint
            url2 = "https://www.hackerrank.com/rest/contests/master/challenges"
            resp = _fetch(url2, params={"track": slug, "limit": limit})
            data = resp.json()

        challenges: list[dict] = []
        models_list = data.get("models", data.get("data", []))

        for item in models_list:
            name = item.get("name", "") or item.get("slug", "")
            preview = item.get("preview", "") or item.get("description_short", "")
            difficulty = item.get("difficulty_name", "medium").lower()

            if not name:
                continue

            challenges.append(
                {
                    "source": "hackerrank",
                    "challenge_slug": item.get("slug", ""),
                    "question": f"{name}: {preview}" if preview else name,
                    "category": category,
                    "subcategory": slug,
                    "difficulty": difficulty,
                    "max_score": item.get("max_score", 0),
                    "answer": None,
                    "url": f"https://www.hackerrank.com/challenges/{item.get('slug', '')}/problem",
                }
            )

        return challenges


class QuestionTemplateCollector:
    """
    Orchestrates crawling all template sources and saves unified output.

    Output:
      data/raw/question_templates/aops.jsonl
      data/raw/question_templates/euler.jsonl
      data/raw/question_templates/rosetta.jsonl
      data/raw/question_templates/hackerrank.jsonl
      data/raw/question_templates/all_templates.jsonl  — combined
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        sources: list[str] | None = None,
        max_per_source: int = 500,
    ) -> dict[str, int]:
        if sources is None:
            sources = ["aops", "euler", "rosetta", "hackerrank"]

        all_templates: list[dict] = []
        counts: dict[str, int] = {}

        if "aops" in sources:
            logger.info("Crawling AoPS...")
            crawled = AoPSCrawler().crawl(self.output_dir, max_problems=max_per_source)
            self._save("aops.jsonl", crawled)
            all_templates.extend(crawled)
            counts["aops"] = len(crawled)

        if "euler" in sources:
            logger.info("Crawling Project Euler...")
            crawled = ProjectEulerCrawler().crawl(
                self.output_dir, max_problems=max_per_source
            )
            self._save("euler.jsonl", crawled)
            all_templates.extend(crawled)
            counts["euler"] = len(crawled)

        if "rosetta" in sources:
            logger.info("Crawling Rosetta Code...")
            crawled = RosettaCodeCrawler().crawl(
                self.output_dir, max_tasks=max_per_source
            )
            self._save("rosetta.jsonl", crawled)
            all_templates.extend(crawled)
            counts["rosetta"] = len(crawled)

        if "hackerrank" in sources:
            logger.info("Crawling HackerRank...")
            crawled = HackerRankCrawler().crawl(
                self.output_dir, max_challenges=max_per_source
            )
            self._save("hackerrank.jsonl", crawled)
            all_templates.extend(crawled)
            counts["hackerrank"] = len(crawled)

        self._save("all_templates.jsonl", all_templates)
        counts["total"] = len(all_templates)
        logger.info(f"Total templates collected: {len(all_templates)}")
        return counts

    def _save(self, filename: str, records: list[dict]) -> None:
        path = self.output_dir / filename
        with path.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        logger.debug(f"  Saved {len(records)} → {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect question templates from contest/coding sites"
    )
    parser.add_argument("--output", default="data/raw/question_templates")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["aops", "euler", "rosetta", "hackerrank"],
        choices=["aops", "euler", "rosetta", "hackerrank"],
    )
    parser.add_argument("--max-per-source", type=int, default=500)
    args = parser.parse_args()

    collector = QuestionTemplateCollector(output_dir=args.output)
    counts = collector.run(sources=args.sources, max_per_source=args.max_per_source)

    logger.info("=== Template Collection Summary ===")
    for source, count in counts.items():
        logger.info(f"  {source:<15} {count:>5} templates")
