"""
synthesis/prompts.py — Prompt templates for all EvalForge synthesis tasks.

All prompts produce structured JSON outputs that are validated before saving.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# SYSTEM PROMPTS
# ---------------------------------------------------------------------------

SYSTEM_AUDIT = """\
You are an expert evaluation scientist trained in psychometrics, benchmark design, and AI evaluation methodology.
Your task is to produce rigorous, structured audits of benchmark items.
Always ground your analysis in specific, observable properties of the items — not vague impressions.
Output valid JSON only. No markdown, no prose outside the JSON structure.
"""

SYSTEM_ITEM_GENERATOR = """\
You are an expert evaluation item designer. You create novel, adversarially robust benchmark items
that test genuine capability without exploitable shortcuts.
Requirements:
- Items must not be resolvable by lexical overlap, length heuristics, negation artifacts, or position bias
- Items must target a specific, nameable cognitive or language capability
- Answer choices (if multiple choice) must be plausibly wrong — not trivially rejectable
Output valid JSON only.
"""

SYSTEM_SHORTCUT_DETECTOR = """\
You are a benchmark red-teamer. Your task is to identify specific, exploitable shortcuts in evaluation items —
patterns that allow a model to achieve correct answers without the target capability.
Be precise: name the specific pattern, quantify its severity, and identify the affected fraction of items.
Output valid JSON only.
"""

SYSTEM_GOODHART = """\
You are an expert in measurement theory and Goodhart's Law. Given a case study of a metric being gamed,
identify which of the canonical Goodhart patterns it exemplifies, and explain how the same pattern
manifests in AI benchmark gaming.
Output valid JSON only.
"""

SYSTEM_IRT = """\
You are a psychometrician specializing in Item Response Theory. Given a set of items and response data,
produce IRT parameter estimates and item quality diagnostics.
Use the 2PL or 3PL model as appropriate. Flag items with low discrimination (a < 0.3).
Output valid JSON only.
"""

# ---------------------------------------------------------------------------
# AUDIT PAIR PROMPT
# ---------------------------------------------------------------------------

AUDIT_PAIR_PROMPT = """\
Analyze the following benchmark item for evaluation quality issues.

BENCHMARK: {benchmark_name}
CATEGORY: {category}
ITEM:
{item_json}

MODEL PERFORMANCE:
{model_scores_json}

Produce a structured audit report with this exact schema:
{{
  "contamination": {{
    "score": <float 0.0-1.0>,
    "evidence": "<specific evidence string or null>",
    "suspected_source": "<likely pretraining source or null>"
  }},
  "shortcuts": [
    {{
      "pattern": "<pattern name from taxonomy>",
      "severity": <float 0.0-1.0>,
      "description": "<specific description of how this shortcut manifests>",
      "affected_fraction": <float 0.0-1.0>
    }}
  ],
  "irt": {{
    "discrimination_a": <float>,
    "difficulty_b": <float>,
    "guessing_c": <float 0.0-0.33>,
    "quality_flag": "<OK | LOW_DISCRIMINATION | CEILING | FLOOR>"
  }},
  "downstream_validity": {{
    "target_construct": "<what capability this item is supposed to measure>",
    "construct_validity": "<STRONG | MODERATE | WEAK>",
    "notes": "<why>"
  }},
  "recommendation": "<KEEP | REPLACE | REVISE | FLAG_FOR_REVIEW>",
  "replacement_priority": <integer 1-5, 1=most urgent>
}}
"""

# ---------------------------------------------------------------------------
# ITEM GENERATION PROMPT
# ---------------------------------------------------------------------------

ITEM_GENERATION_PROMPT = """\
Generate {n_items} novel benchmark items for the following specification:

TARGET CONSTRUCT: {construct}
DIFFICULTY RANGE: {difficulty_min} to {difficulty_max} (IRT theta scale, -3 to +3)
AVOID SHORTCUTS: {shortcuts_to_avoid}
BENCHMARK STYLE: {benchmark_style}
EXISTING ITEMS (for diversity reference):
{existing_items_sample}

Requirements:
- Each item must test {construct} and ONLY {construct}
- Answer choices must each be plausible (not trivially rejectable)
- No lexical overlap between question stem and correct answer choice
- Balanced answer length across choices (no length bias)
- Novel surface form — not paraphrases of existing items

Output schema:
{{
  "items": [
    {{
      "id": "<uuid>",
      "question": "<question text>",
      "choices": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "<A|B|C|D>",
      "construct": "{construct}",
      "estimated_difficulty_b": <float -3 to 3>,
      "shortcut_check": {{
        "length_bias_free": <bool>,
        "lexical_overlap_free": <bool>,
        "position_bias_free": <bool>
      }},
      "rationale": "<why the correct answer is correct>"
    }}
  ]
}}
"""

# ---------------------------------------------------------------------------
# SHORTCUT DETECTION PROMPT
# ---------------------------------------------------------------------------

SHORTCUT_DETECTION_PROMPT = """\
You are given a set of benchmark items. Identify all exploitable shortcuts present.

ITEMS:
{items_json}

For each shortcut found, specify:
{{
  "shortcuts_found": [
    {{
      "pattern": "<pattern name>",
      "category": "<SURFACE | DISTRIBUTIONAL | STRUCTURAL | SEMANTIC>",
      "severity": <float 0.0-1.0>,
      "affected_item_ids": [<list of item IDs>],
      "detection_method": "<how to detect this programmatically>",
      "example": "<concrete example from the provided items>"
    }}
  ],
  "overall_quality_score": <float 0.0-1.0>,
  "recommendation": "<summary recommendation>"
}}

Known shortcut patterns to check:
- length_bias: correct answers systematically longer/shorter than distractors
- lexical_overlap: answer found by matching words from question to answer choice
- negation_artifact: presence of "not/never/no" as sufficient signal
- position_bias: correct answer at same position too often (A/B/C/D distribution)
- distractor_implausibility: distractors trivially wrong without domain knowledge
- semantic_collapse: multiple distractors with near-identical meaning
- annotation_agreement: items where annotators had >95% agreement (too easy)
- template_fill: item is fill-in-the-blank resolvable without reasoning
"""

# ---------------------------------------------------------------------------
# GOODHART PATTERN PROMPT
# ---------------------------------------------------------------------------

GOODHART_PATTERN_PROMPT = """\
Analyze this case study through the lens of Goodhart's Law and benchmark gaming.

CASE STUDY:
{case_study}

DOMAIN: {domain}

Produce this analysis:
{{
  "goodhart_patterns": [
    {{
      "pattern_id": "<pattern ID from taxonomy>",
      "pattern_name": "<pattern name>",
      "manifestation": "<how this pattern appears in the case study>",
      "ai_benchmark_analog": "<how the same pattern appears in AI evaluation>",
      "severity": <float 0.0-1.0>,
      "mitigation": "<how to design benchmarks that resist this pattern>"
    }}
  ],
  "primary_pattern": "<most dominant pattern ID>",
  "proxy_construct_gap": "<description of the gap between proxy metric and true construct>",
  "historical_timeline": "<how the gaming evolved over time>"
}}
"""

# ---------------------------------------------------------------------------
# IRT CALIBRATION PROMPT
# ---------------------------------------------------------------------------

IRT_CALIBRATION_PROMPT = """\
Given this response matrix (rows = models, columns = items, values = 0/1 correct),
produce IRT parameter estimates using the 2PL model.

RESPONSE MATRIX:
{response_matrix_json}

ITEM METADATA:
{item_metadata_json}

Produce:
{{
  "item_parameters": [
    {{
      "item_id": "<id>",
      "discrimination_a": <float, target range 0.5-3.0>,
      "difficulty_b": <float, target range -3 to +3>,
      "quality_flags": ["<LOW_DISCRIMINATION>" | "<CEILING>" | "<FLOOR>" | "<OK>"],
      "information_function_peak": <float — theta value at maximum information>
    }}
  ],
  "model_ability_estimates": {{
    "<model_name>": <float theta estimate>
  }},
  "test_information": {{
    "peak_theta": <float>,
    "reliability_estimate": <float 0.0-1.0>,
    "effective_n_items": <int — items contributing substantial information>
  }},
  "recommendations": {{
    "items_to_replace": ["<item_id>"],
    "target_difficulty_gap": "<description of underrepresented difficulty region>",
    "n_items_needed_for_target_reliability": <int>
  }}
}}
"""
