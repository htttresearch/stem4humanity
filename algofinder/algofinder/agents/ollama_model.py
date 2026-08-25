"""Concrete local-model adapter for evaluator-driven solver research."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from algofinder.agents.campaign import Campaign
from algofinder.agents.candidate_templates import (
    TSP_EDGE_FORMULA_TEMPLATE,
    TemplateError,
    validate_values,
)
from algofinder.agents.learning.normalization import NormalizationError, normalize_formula
from algofinder.agents.research import (
    AnalysisDraft,
    CandidateProposal,
    DEFAULT_METHOD_PROMPT,
    ResearchContext,
    ResearchError,
    SearchDirective,
    TEMPLATE_METHOD_PROMPT,
)
from algofinder.agents.tsp_hir import HIREdit, HIRError


def _repair_formula_aliases(expression: str) -> str:
    """Repair narrow, semantics-preserving vocabulary slips from small models."""
    return re.sub(r"\bangle\b", "angular_offset", expression.strip())


_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mechanism": {"type": "string"},
        "affected_strata": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "predicted_tradeoffs": {"type": "string"},
        "falsification_test": {"type": "string"},
        "patch": {"type": "string"},
        "solver_entrypoints": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "expected_result": {"type": "string"},
        "descriptors": {
            "type": "object",
            "properties": {
                "algorithm_family": {"type": "string"},
                "runtime_class": {"type": "string"},
            },
            "required": ["algorithm_family", "runtime_class"],
            "additionalProperties": True,
        },
        "supporting_sources": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": [
        "mechanism", "affected_strata", "predicted_tradeoffs", "falsification_test",
        "patch", "solver_entrypoints", "expected_result", "descriptors",
        "supporting_sources", "confidence",
    ],
    "additionalProperties": False,
}

_TSP_FORMULA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "priority_expressions": {
            "type": "array",
            "items": {"type": "string", "maxLength": 240},
            "minItems": 6,
            "maxItems": 8,
            "uniqueItems": True,
        },
    },
    "required": ["priority_expressions"],
    "additionalProperties": False,
}

_TSP_HIR_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operator": {
            "type": "string",
            "enum": ["replace_component", "set_parameter", "replace_expression"],
        },
        "target": {"type": "string", "maxLength": 80},
        "value": {"type": "object"},
        "hypothesis": {"type": "string", "minLength": 1, "maxLength": 600},
    },
    "required": ["operator", "target", "value", "hypothesis"],
    "additionalProperties": False,
}

_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "observation": {"type": "string"},
        "interpretation": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "next_test": {"type": "string"},
        "trace_regions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["observation", "interpretation", "confidence", "next_test", "trace_regions"],
    "additionalProperties": False,
}


def _template_research_evidence(
    mapping: dict[str, Any], directive: SearchDirective
) -> dict[str, Any]:
    """Compact ledger evidence for small-model template proposals.

    Opaque UUIDs and raw runs are deliberately omitted: they add prompt noise
    and used to make otherwise identical seeded campaigns diverge.  The model
    instead receives the selected parent's exact authority-derived template
    state, its public score, and a few scored archive exemplars.
    """
    candidates = [item for item in mapping.get("candidates", []) if isinstance(item, dict)]
    by_id = {str(item.get("id")): item for item in candidates if item.get("id")}
    evaluations: dict[str, list[dict[str, Any]]] = {}
    for item in mapping.get("evaluations", []):
        if isinstance(item, dict) and item.get("candidate_id"):
            evaluations.setdefault(str(item["candidate_id"]), []).append(item)

    def public_result(candidate_id: str) -> dict[str, Any] | None:
        rows = evaluations.get(candidate_id, [])
        if not rows:
            return None
        row = max(rows, key=lambda item: (_stage_number(item.get("stage")), str(item.get("gate", ""))))
        return {
            "stage": row.get("stage"),
            "gate": row.get("gate"),
            "metric_vector": row.get("metric_vector", {}),
            "paired_baselines": row.get("paired_baselines", {}),
        }

    def compact(candidate: dict[str, Any]) -> dict[str, Any]:
        descriptors = candidate.get("descriptors", {})
        if not isinstance(descriptors, dict):
            descriptors = {}
        hypothesis = candidate.get("hypothesis", {})
        return {
            "proposal_ordinal": descriptors.get("proposal_ordinal"),
            "generation_operator": candidate.get("generation_operator"),
            "hypothesis": hypothesis.get("mechanism") if isinstance(hypothesis, dict) else None,
            "template_state": descriptors.get("template_state"),
            "behavior": {
                key: descriptors[key]
                for key in (
                    "edge_score_features", "distance_monotonicity", "rank_monotonicity",
                    "uses_angular_offset", "hir_constructor", "hir_edge_generator",
                    "hir_move_schedule", "hir_diversification", "hir_niche",
                )
                if key in descriptors
            },
            "public_result": public_result(str(candidate.get("id"))),
        }

    parents = [compact(by_id[parent_id]) for parent_id in directive.parent_ids if parent_id in by_id]
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for candidate in candidates:
        result = public_result(str(candidate.get("id")))
        score = _paired_gap(result)
        descriptors = candidate.get("descriptors", {})
        ordinal = descriptors.get("proposal_ordinal", 10**9) if isinstance(descriptors, dict) else 10**9
        if score is not None:
            scored.append((score, int(ordinal) if isinstance(ordinal, int) else 10**9, candidate))
    minimize = str(mapping.get("campaign", {}).get("objective", {}).get("direction", "minimize")).lower() != "maximize"
    scored.sort(key=lambda item: ((item[0] if minimize else -item[0]), item[1]))
    exemplars = [compact(item[2]) for item in scored[:6]]
    return {
        "objective": mapping.get("campaign", {}).get("objective", {}),
        "assigned_operator": directive.operator,
        "selected_parents": parents,
        "best_public_exemplars": exemplars,
        "recent_analyses": [
            {
                key: item.get(key)
                for key in ("observation", "interpretation", "confidence", "next_test")
            }
            for item in mapping.get("analyses", [])[-4:]
            if isinstance(item, dict)
        ],
        "remaining_budget": mapping.get("budget", {}).get("remaining", {}),
        "learned_cases": mapping.get("retrieved_cases", []),
    }


def _stage_number(value: object) -> int:
    try:
        return int(str(value).rpartition("_")[2])
    except ValueError:
        return -1


def _paired_gap(result: dict[str, Any] | None) -> float | None:
    if result is None or result.get("gate") != "pass":
        return None
    metrics = result.get("metric_vector", {})
    if isinstance(metrics, dict):
        for name in ("paired_gap_percent", "mean_gap_percent"):
            value = metrics.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                return float(value)
    paired = result.get("paired_baselines", {})
    if isinstance(paired, dict):
        values = [
            float(item["paired_gap_percent"])
            for item in paired.values()
            if isinstance(item, dict)
            and isinstance(item.get("paired_gap_percent"), (int, float))
            and math.isfinite(float(item["paired_gap_percent"]))
        ]
        if values:
            return sum(values) / len(values)
    return None


@dataclass(frozen=True)
class OllamaModelConfig:
    model: str = "qwen2.5-coder:7b"
    endpoint: str = "http://127.0.0.1:11434"
    temperature: float = 0.2
    seed: int = 75348
    context_tokens: int = 16_384
    proposal_tokens: int = 6_000
    analysis_tokens: int = 900
    request_timeout_seconds: float = 600.0
    keep_alive: str = "15m"
    structured_response_attempts: int = 2

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.endpoint.startswith(("http://", "https://")):
            raise ResearchError("Ollama model and HTTP endpoint are required")
        if self.context_tokens < 2_048 or self.proposal_tokens < 256 or self.analysis_tokens < 128:
            raise ResearchError("Ollama token limits are too small for program search")
        if not 0.0 <= self.temperature <= 2.0 or self.request_timeout_seconds <= 0:
            raise ResearchError("invalid Ollama sampling or timeout configuration")
        if self.structured_response_attempts < 1:
            raise ResearchError("Ollama structured response attempts must be positive")


class OllamaResearchModel:
    """Use Ollama structured outputs as the concrete ``ResearchModel``.

    The model receives only the already-bounded :class:`ResearchContext`.  It
    never receives an AgentTools object, a filesystem handle, or evaluator code.
    Optional campaign accounting reserves a conservative token estimate before
    each generation so the local model cannot silently bypass campaign budgets.
    """

    def __init__(
        self,
        config: OllamaModelConfig | None = None,
        *,
        campaign: Campaign | None = None,
    ) -> None:
        self.config = config or OllamaModelConfig()
        self.campaign = campaign
        self.usage: list[dict[str, int]] = []
        self._request_count = 0
        # The recorder consumes the most recent exchange after a proposal call.
        # It is not written by this adapter itself, keeping retention policy in
        # the campaign-owned learning recorder.
        self.last_exchange: dict[str, Any] | None = None

    def propose(self, context: ResearchContext, directive: SearchDirective) -> CandidateProposal:
        mapping = context.to_mapping()
        if context.candidate_template == "tsp-edge-formula-v1":
            return self._propose_tsp_formula(mapping, directive)
        if context.candidate_template == "tsp-hir-v1":
            return self._propose_tsp_hir(mapping, directive)
        allowed_paths = mapping.get("campaign", {}).get("allowed_paths", [])
        repair = mapping.get("prior_validation_failure")
        repair_instruction = ""
        if isinstance(repair, dict) and repair.get("previous_patch"):
            repair_instruction = (
                "\n\nThe immediately preceding proposal was rejected before evaluation. "
                "Return a fresh, complete replacement diff that fixes these exact errors; "
                "do not repeat its invalid imports, entrypoint, or incomplete class.\n"
                f"Errors: {json.dumps(repair.get('errors', []), ensure_ascii=False)}\n"
                f"Previous entrypoints: {json.dumps(repair.get('previous_entrypoints', []), ensure_ascii=False)}\n"
                "Previous patch:\n"
                + str(repair["previous_patch"])
            )
        prompt = (
            "Design exactly one next solver candidate.\n\n"
            f"Assigned operator: {directive.operator}\n"
            f"Selection rationale: {directive.rationale}\n"
            f"Research focus: {directive.focus}\n"
            f"Writable paths: {json.dumps(allowed_paths)}\n\n"
            "Return a complete Git unified diff in `patch`. For a new file use "
            "`--- /dev/null` and `+++ b/<path>`. Create exactly one solver module "
            "under the writable path, with a unique solver id and an entrypoint "
            "`algofinder.solvers....:ClassName`. The module may import only APIs "
            "shown in source_files (plus Python/NumPy); the `*` in a writable-path "
            "policy is a matcher, never a literal filename character. For this TSP "
            "campaign add one concrete file such as "
            "`algofinder/algofinder/solvers/tsp/campaign_candidate_local_001.py` "
            "and declare its full module name, including the new filename, in "
            "`solver_entrypoints`; never use the package name `algofinder.solvers.tsp` "
            "as an entrypoint. Define a real direct `Solver` subclass with its own literal "
            "nonempty `id` and a complete indented `solve` method. For a new file, the patch must literally start with "
            "`--- /dev/null`, then `+++ b/<exact-path>`, then a valid hunk header such "
            "as `@@ -0,0 +1,3 @@`; every new source line begins with `+`. Do not use "
            "`*** Begin Patch` syntax or omit hunk headers. Do not edit or "
            "import another solver implementation. The candidate module should be "
            "small enough to fit in the response. The module may import only APIs "
            "shown in source_files (plus Python/NumPy); do not delegate to another "
            "registered solver. When parent_patches is non-empty, your diff must apply "
            "incrementally on top of that parent state: edit its existing candidate "
            "module and keep its entrypoint instead of repeating its new-file diff. "
            "Preserve every parent change not explicitly tested by this mutation. Do not "
            "claim a score; predict a directional, falsifiable outcome."
            + repair_instruction
            + "\n\n"
            "Research evidence (JSON):\n" + json.dumps(mapping, ensure_ascii=False, sort_keys=True)
        )
        data = self._chat(
            system=DEFAULT_METHOD_PROMPT,
            user=prompt,
            schema=_PROPOSAL_SCHEMA,
            max_tokens=self.config.proposal_tokens,
            purpose="model:proposal",
        )
        try:
            return CandidateProposal(
                mechanism=str(data["mechanism"]),
                affected_strata=tuple(str(item) for item in data["affected_strata"]),
                predicted_tradeoffs=str(data["predicted_tradeoffs"]),
                falsification_test=str(data["falsification_test"]),
                patch=str(data["patch"]),
                solver_entrypoints=tuple(str(item) for item in data["solver_entrypoints"]),
                expected_result=str(data["expected_result"]),
                descriptors=dict(data["descriptors"]),
                supporting_sources=tuple(str(item) for item in data["supporting_sources"]),
                confidence=_bounded_confidence(data["confidence"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResearchError(f"invalid structured Ollama proposal: {exc}") from exc

    def _propose_tsp_formula(
        self,
        mapping: dict[str, Any],
        directive: SearchDirective,
    ) -> CandidateProposal:
        repair = mapping.get("prior_validation_failure")
        previous = (
            repair.get("previous_template_values", {}).get("priority_expression")
            if isinstance(repair, dict) and isinstance(repair.get("previous_template_values"), dict)
            else None
        )
        repair_note = (
            "\nPrevious rejected expression: " + json.dumps(previous)
            + ". Return a corrected expression, not source code."
            if isinstance(previous, str) else ""
        )
        existing_expressions = sorted({
            str(state["priority_expression"])
            for candidate in mapping.get("candidates", ())
            if isinstance(candidate, dict)
            for descriptors in (candidate.get("descriptors"),)
            if isinstance(descriptors, dict)
            for state in (descriptors.get("template_state"),)
            if isinstance(state, dict) and isinstance(state.get("priority_expression"), str)
        })
        existing_behaviors = {
            str(descriptors["formula_behavior_digest"])
            for candidate in mapping.get("candidates", ())
            if isinstance(candidate, dict)
            for descriptors in (candidate.get("descriptors"),)
            if isinstance(descriptors, dict)
            and isinstance(descriptors.get("formula_behavior_digest"), str)
        }
        # ``distance`` is the experiment's control, not a novel candidate.
        # The semantic probes also collapse positive distance/rank mixtures
        # that preserve the exact same edge ordering.
        existing_behaviors.add(normalize_formula("distance").probe_behavior_digest)
        for expression in existing_expressions:
            try:
                existing_behaviors.add(normalize_formula(expression).probe_behavior_digest)
            except NormalizationError:
                pass
        novelty_note = (
            "\nAlready evaluated normalized expressions (do not repeat any of them): "
            + json.dumps(existing_expressions[:24], ensure_ascii=False)
            if existing_expressions else ""
        )
        evidence = _template_research_evidence(mapping, directive)
        learned_recommendations = [
            item.get("recommendation")
            for item in evidence.get("learned_cases", ())
            if isinstance(item, dict) and isinstance(item.get("recommendation"), dict)
        ]
        warm_start_note = ""
        if learned_recommendations:
            top = learned_recommendations[0]
            expression = top.get("priority_expression")
            if directive.operator == "invent" and not existing_expressions and isinstance(expression, str):
                warm_start_note = (
                    "\nThis campaign has no evaluated candidate. Put the top learned program "
                    "recommendation first in the pool exactly: " + json.dumps(expression) + ". "
                    "Make every remaining pool member a behaviorally distinct causal variant. "
                    "Its predicted values are retrospective estimates, not evaluator measurements."
                )
            else:
                warm_start_note = (
                    "\nLearned program recommendations are advisory retrospective estimates. "
                    "Use them as priors, but return a behaviorally novel causal variant whenever "
                    "the recommended behavior is already listed as evaluated."
                )
        prompt = (
            "Return a ranked pool of six to eight Euclidean TSP candidate-edge ranking expressions. The trusted "
            "scaffold already owns the solver class, IDs, file path, tour construction, "
            "2-opt loop, and all imports. You must return only `priority_expressions`: "
            "a JSON array of short Python numeric expressions, best hypothesis first. "
            "Smaller values select edges first. Every pool member must be mutually "
            "behaviorally distinct; vary mechanisms and coefficients, not whitespace. "
            "The only allowed variables are `distance`, `normalized_rank`, "
            "`angular_offset`, `mean_distance`, and `n`; the only allowed calls are "
            "`abs`, `min`, and `max`. Use the exact name `angular_offset`, never "
            "the alias `angle`. Because `normalized_rank` increases with `distance`, "
            "a positive distance/rank mixture alone is control-equivalent; include "
            "an angular interaction or another term that can actually change ordering. "
            "Do not return a diff, class, import, function, "
            "assignment, explanation in the expression, or code fences. A valid simple "
            "example is `distance + 0.05 * mean_distance * normalized_rank`. "
            "Use the evaluator-authored evidence below. For mutate, tune, repair, or "
            "distill, make one explicit causal change to the selected parent's exact "
            "priority_expression; do not return it unchanged. For recombine, preserve "
            "one identifiable term from each selected parent. Do not infer or invent "
            "scores that are absent from the evidence.\n\n"
            f"Assigned operator: {directive.operator}\n"
            f"Selection rationale: {directive.rationale}\n"
            f"Research focus: {directive.focus}"
            + warm_start_note
            + repair_note
            + novelty_note
            + "\n\nEvaluator-authored research evidence (JSON):\n"
            + json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        )
        data = self._chat(
            system=TEMPLATE_METHOD_PROMPT,
            user=prompt,
            schema=_TSP_FORMULA_SCHEMA,
            max_tokens=self.config.proposal_tokens,
            purpose="model:formula-proposal",
        )
        try:
            raw_pool = data["priority_expressions"]
            if not isinstance(raw_pool, list):
                raise TypeError("priority_expressions must be a list")
            selected_expression: str | None = None
            seen_pool_behaviors: set[str] = set()
            for raw_expression in raw_pool:
                try:
                    repaired = _repair_formula_aliases(str(raw_expression))
                    checked = validate_values(
                        TSP_EDGE_FORMULA_TEMPLATE.template_id,
                        {"priority_expression": repaired},
                    )["priority_expression"]
                    behavior = normalize_formula(checked).probe_behavior_digest
                except (TypeError, ValueError, TemplateError, NormalizationError):
                    # A ranked pool is useful only if one malformed suggestion
                    # cannot discard its valid siblings.
                    continue
                if behavior in existing_behaviors or behavior in seen_pool_behaviors:
                    continue
                seen_pool_behaviors.add(behavior)
                selected_expression = checked
                break
            if selected_expression is None:
                raise ResearchError(
                    "formula pool contained no valid behaviorally novel expression"
                )
            return CandidateProposal(
                mechanism=f"candidate-edge priority: {selected_expression}",
                affected_strata=("tsp:euclidean", "tsp:clustered"),
                predicted_tradeoffs="alters candidate-edge ordering while retaining the fixed construction and local-search budget",
                falsification_test="paired public benchmark against the distance-ranked control",
                expected_result="an evaluator-verified feasible TSP tour",
                template_values={"priority_expression": selected_expression},
                descriptors={"algorithm_family": "candidate-edge-formula", "runtime_class": "fixed-2opt"},
                confidence=0.5,
            )
        except (KeyError, TypeError, ValueError, TemplateError, NormalizationError) as exc:
            raise ResearchError(f"invalid structured Ollama formula proposal: {exc}") from exc

    def _propose_tsp_hir(
        self,
        mapping: dict[str, Any],
        directive: SearchDirective,
    ) -> CandidateProposal:
        repair = mapping.get("prior_validation_failure")
        previous = (
            repair.get("previous_template_values", {}).get("edit_json")
            if isinstance(repair, dict) and isinstance(repair.get("previous_template_values"), dict)
            else None
        )
        repair_note = (
            "\nThe prior edit was rejected. Return one corrected typed edit, not Python: "
            + str(previous)[:1_500]
            if isinstance(previous, str) else ""
        )
        evidence = _template_research_evidence(mapping, directive)
        prompt = (
            "Propose exactly one typed edit to a trusted Euclidean-TSP heuristic genome. "
            "Return no source code. `replace_component` targets one of constructor, "
            "edge_generator, move_schedule, acceptance, diversification, adaptation; "
            "its value is a partial object merged into that component. `set_parameter` "
            "targets component.field and uses value {\"value\": number}. "
            "`replace_expression` targets edge_score and its value is a typed tree: "
            "feature {kind:feature,name:distance|normalized_rank|angular_offset|mean_distance|n}; "
            "constant {kind:constant,value:number}; binary {kind:binary,op:add|sub|mul|div|min|max,left:<expr>,right:<expr>}; "
            "or if {kind:if,condition:{op:lt|lte|gt|gte,left:<expr>,right:<expr>},then:<expr>,otherwise:<expr>}. "
            "Component kinds: constructor nearest_neighbor|regret_insertion|multi_fragment; "
            "edge_generator k_nearest|angular|reciprocal|union; move_schedule two_opt|or_opt|mixed|bounded_three_opt; "
            "acceptance first|best|threshold|annealed; diversification none|double_bridge|ruin_recreate|restart. "
            "When selecting bounded_three_opt, also set three_opt_trials to a positive value. "
            "When selecting ruin_recreate, set ruin_fraction deliberately. "
            "The edit is applied to the first selected parent's exact genome shown in "
            "the evidence. For mutate, tune, repair, or distill, change that genome "
            "causally and do not repeat its current value. For recombine, copy one "
            "useful component or parameter from the second parent into the first. "
            "Use public evaluator outcomes and stratum summaries when present; never "
            "invent a missing score.\n\n"
            f"Assigned operator: {directive.operator}\n"
            f"Selection rationale: {directive.rationale}\n"
            f"Research focus: {directive.focus}"
            + repair_note
            + "\n\nEvaluator-authored research evidence (JSON):\n"
            + json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        )
        data = self._chat(
            system=TEMPLATE_METHOD_PROMPT,
            user=prompt,
            schema=_TSP_HIR_EDIT_SCHEMA,
            max_tokens=self.config.proposal_tokens,
            purpose="model:hir-edit",
        )
        try:
            edit = HIREdit.from_mapping(data)
            return CandidateProposal(
                mechanism=edit.hypothesis,
                affected_strata=("tsp:euclidean", "tsp:clustered"),
                predicted_tradeoffs="typed HIR component change under the fixed resource envelope",
                falsification_test="paired public benchmark against the distance-ranked control",
                expected_result="an evaluator-verified feasible TSP tour",
                template_values={"edit_json": edit.to_json()},
                descriptors={
                    "algorithm_family": "typed-hir",
                    "runtime_class": "compiled-local-search",
                    "hir_operator": edit.operator,
                    "hir_target": edit.target,
                },
                confidence=0.5,
            )
        except (HIRError, KeyError, TypeError, ValueError) as exc:
            raise ResearchError(f"invalid structured Ollama HIR edit: {exc}") from exc

    def analyze(
        self,
        context: ResearchContext,
        directive: SearchDirective,
        proposal: CandidateProposal,
        feedback: tuple[dict[str, Any], ...],
    ) -> AnalysisDraft:
        compact_feedback = tuple(
            {key: value for key, value in item.items() if key != "raw_runs"}
            for item in feedback
        )
        prompt = (
            "Analyze only the evaluator feedback below. Separate observation from "
            "interpretation, do not manufacture missing measurements, and propose one "
            "falsifying next test.\n\n"
            + json.dumps(
                {
                    "operator": directive.operator,
                    "mechanism": proposal.mechanism,
                    "expected_result": proposal.expected_result,
                    "feedback": list(compact_feedback),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        data = self._chat(
            system=DEFAULT_METHOD_PROMPT,
            user=prompt,
            schema=_ANALYSIS_SCHEMA,
            max_tokens=self.config.analysis_tokens,
            purpose="model:analysis",
        )
        try:
            return AnalysisDraft(
                observation=str(data["observation"]),
                interpretation=str(data["interpretation"]),
                confidence=_bounded_confidence(data["confidence"]),
                next_test=str(data["next_test"]),
                trace_regions=tuple(str(item) for item in data["trace_regions"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResearchError(f"invalid structured Ollama analysis: {exc}") from exc

    def _chat(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int,
        purpose: str,
    ) -> dict[str, Any]:
        last_error: ResearchError | None = None
        for _ in range(self.config.structured_response_attempts):
            if self.campaign is not None:
                estimated_tokens = math.ceil((len(system) + len(user)) / 4) + max_tokens
                self.campaign.reserve_budget(purpose=purpose, amounts={"tokens": estimated_tokens})
            self._request_count += 1
            body = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # Qwen3.5 emits its reasoning in a separate message field by
                # default.  A research proposal is a strict machine-readable
                # contract, so reasoning mode must be off: the JSON schema is
                # then returned in `message.content`, the only field this
                # adapter accepts as a candidate proposal.
                "think": False,
                "format": schema,
                "options": {
                    "temperature": self.config.temperature,
                    "seed": self.config.seed + self._request_count,
                    "num_ctx": self.config.context_tokens,
                    "num_predict": max_tokens,
                },
                "keep_alive": self.config.keep_alive,
            }
            request = Request(
                self.config.endpoint.rstrip("/") + "/api/chat",
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.config.request_timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[-1_000:]
                self.last_exchange = {
                    "request_id": f"ollama-{self._request_count}",
                    "request": body,
                    "response": {"error": f"HTTP {exc.code}", "detail": detail},
                }
                raise ResearchError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
            except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                self.last_exchange = {
                    "request_id": f"ollama-{self._request_count}",
                    "request": body,
                    "response": {"error": str(exc)},
                }
                last_error = ResearchError(f"Ollama request failed before usage could be measured: {exc}")
                continue
            usage = {
                "prompt_tokens": int(payload.get("prompt_eval_count", 0)),
                "completion_tokens": int(payload.get("eval_count", 0)),
            }
            self.usage.append(usage)
            if self.campaign is not None:
                self.campaign.record_usage(
                    purpose=purpose,
                    amounts={"tokens": usage["prompt_tokens"] + usage["completion_tokens"]},
                )
            self.last_exchange = {
                "request_id": f"ollama-{self._request_count}",
                "request": body,
                "response": {
                    "content": payload.get("message", {}).get("content"),
                    "usage": usage,
                    "done_reason": payload.get("done_reason"),
                },
            }
            try:
                content = payload["message"]["content"]
                result = json.loads(content)
                if not isinstance(result, dict):
                    raise TypeError("Ollama structured response must be an object")
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                last_error = ResearchError(f"Ollama response was not valid structured JSON: {exc}")
                continue
            return result
        raise last_error or ResearchError("Ollama response did not contain structured JSON")


def unload_ollama_model(*, endpoint: str, model: str, timeout_seconds: float = 30.0) -> None:
    """Explicitly evict one model from the Ollama runtime.

    ``keep_alive`` is deliberately long while a campaign is running so model
    weights are not repeatedly paged into GPU memory.  Call this at an arm
    boundary before another model is allowed to start, particularly on GPUs
    that cannot safely retain both model weights.
    """
    if not endpoint.startswith(("http://", "https://")) or not model.strip():
        raise ResearchError("a valid Ollama endpoint and model name are required for unload")
    body = {"model": model, "stream": False, "keep_alive": 0}
    request = Request(
        endpoint.rstrip("/") + "/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[-1_000:]
        raise ResearchError(f"Ollama unload returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ResearchError(f"could not unload Ollama model {model!r}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("done") is not True:
        raise ResearchError(f"Ollama did not confirm unloading model {model!r}")


def ollama_model_identity(
    *, endpoint: str, model: str, timeout_seconds: float = 30.0
) -> dict[str, Any]:
    """Resolve the immutable Ollama blob digest used by an experiment."""
    request = Request(endpoint.rstrip("/") + "/api/tags", method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ResearchError(f"cannot resolve Ollama model identity: {exc}") from exc
    rows = payload.get("models", []) if isinstance(payload, dict) else []
    match = next(
        (
            item for item in rows
            if isinstance(item, dict)
            and (item.get("name") == model or item.get("model") == model)
        ),
        None,
    )
    if match is None:
        raise ResearchError(f"Ollama model is not installed: {model}")
    digest = match.get("digest")
    if not isinstance(digest, str) or not digest:
        raise ResearchError(f"Ollama did not report a blob digest for {model}")
    return {
        "requested_name": model,
        "resolved_name": match.get("name", match.get("model", model)),
        "blob_digest": digest,
        "details": match.get("details", {}),
    }


def _bounded_confidence(value: Any) -> float:
    """Keep fallible model self-assessments from aborting a real campaign.

    Confidence is descriptive metadata, not an evaluator signal.  Local models
    occasionally emit values on a 0--100 scale despite the response schema, so
    reject non-finite data but normalize numeric over/underflow to the contract
    range before constructing an immutable record.
    """
    confidence = float(value)
    if not math.isfinite(confidence):
        raise ValueError("confidence must be finite")
    return min(1.0, max(0.0, confidence))


__all__ = [
    "OllamaModelConfig", "OllamaResearchModel", "ollama_model_identity",
    "unload_ollama_model",
]
