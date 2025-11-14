"""Utility for extracting question templates and estimating sample difficulty.

This module groups natural-language VQA questions into reusable templates,
computes lightweight difficulty signals, and surfaces the hardest samples so
that augmentation pipelines can focus on under-represented patterns.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from dataset.base import BaseDataset


@dataclass
class DifficultyWeights:
    """Weight configuration for combining difficulty factors."""

    template_rarity: float = 0.4
    answer_rarity: float = 0.35
    question_complexity: float = 0.25

    def normalized(self) -> "DifficultyWeights":
        total = self.template_rarity + self.answer_rarity + self.question_complexity
        if total == 0:
            return DifficultyWeights(
                template_rarity=1 / 3,
                answer_rarity=1 / 3,
                question_complexity=1 / 3,
            )
        return DifficultyWeights(
            template_rarity=self.template_rarity / total,
            answer_rarity=self.answer_rarity / total,
            question_complexity=self.question_complexity / total,
        )

    def as_dict(self) -> Dict[str, float]:
        normalized = self.normalized()
        return {
            "template_rarity": normalized.template_rarity,
            "answer_rarity": normalized.answer_rarity,
            "question_complexity": normalized.question_complexity,
        }


@dataclass
class TemplateExtractionConfig:
    """Configuration for question template extraction and difficulty scoring."""

    dataset_name: str = "unknown"
    min_template_support: int = 3
    max_structural_tokens: int = 48
    structural_token_percentile: float = 0.2
    hardness_percentile: float = 0.8
    max_hard_samples: int = 256
    preserve_punctuation: bool = True
    include_special_maps: bool = True
    weights: DifficultyWeights = field(default_factory=DifficultyWeights)

    def __post_init__(self):
        if not 0 <= self.structural_token_percentile <= 1:
            raise ValueError("structural_token_percentile must be between 0 and 1")
        if not 0 <= self.hardness_percentile <= 1:
            raise ValueError("hardness_percentile must be between 0 and 1")
        if self.max_structural_tokens <= 0:
            raise ValueError("max_structural_tokens must be positive")
        if self.max_hard_samples <= 0:
            raise ValueError("max_hard_samples must be positive")


@dataclass
class TemplateStats:
    """Aggregated statistics for a single template."""

    template: str
    support: int
    avg_difficulty: float
    sample_indices: List[int]
    representative_question: str
    representative_answer: str


@dataclass
class HardSample:
    """Metadata for a high-difficulty data point."""

    index: int
    question: str
    answer: str
    img_path: str
    template: str
    difficulty: float
    reason_scores: Dict[str, float]
    suggested_augmentations: List[str]


@dataclass
class TemplateExtractionReport:
    """Result payload returned by the extractor."""

    dataset_name: str
    templates: List[TemplateStats]
    hard_samples: List[HardSample]
    structural_tokens: List[str]

    def get_template(self, template: str) -> Optional[TemplateStats]:
        for stats in self.templates:
            if stats.template == template:
                return stats
        return None

    def build_augmentation_payload(self) -> List[Dict[str, object]]:
        """Create a JSON-serializable payload for augmentation jobs."""
        payload = []
        for sample in self.hard_samples:
            payload.append(
                {
                    "index": sample.index,
                    "img_path": sample.img_path,
                    "question": sample.question,
                    "answer": sample.answer,
                    "template": sample.template,
                    "difficulty": round(sample.difficulty, 4),
                    "reason_scores": sample.reason_scores,
                    "suggested_augmentations": sample.suggested_augmentations,
                }
            )
        return payload


class QuestionTemplateExtractor:
    """Extract question templates and estimate their difficulty."""

    TOKEN_PATTERN = re.compile(r"\d+\.\d+|\d+|[^\W\d_]+|[^\s]", re.UNICODE)
    NUM_PATTERN = re.compile(r"\d+(?:\.\d+)?")
    PUNCTUATION = {"?", "!", ".", ",", ":", ";", "-", "(", ")", "\"", "'", "/"}

    def __init__(self, config: Optional[TemplateExtractionConfig] = None):
        self.config = config or TemplateExtractionConfig()
        self.weights = self.config.weights.normalized()
        self.special_token_map = self._build_special_token_map()

    def extract(self, dataset_or_batch: Sequence[str] | BaseDataset | Dict[str, Sequence[str]]) -> TemplateExtractionReport:
        questions, answers, img_paths = self._resolve_dataset(dataset_or_batch)
        normalized_questions = [q.strip() for q in questions]
        tokenized_questions = [self._tokenize(question) for question in normalized_questions]
        normalized_tokens = [self._normalize_tokens(tokens) for tokens in tokenized_questions]

        structural_tokens = self._select_structural_tokens(normalized_tokens)
        templates = [self._build_template(tokens, structural_tokens) for tokens in normalized_tokens]

        template_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, template in enumerate(templates):
            template_to_indices[template].append(idx)

        answer_counter = Counter(answers)
        lengths = [len(tokens) for tokens in normalized_tokens]
        unique_ratios = [self._safe_div(len(set(tokens)), max(1, len(tokens))) for tokens in normalized_tokens]

        difficulties, reason_scores = self._score_samples(
            templates=templates,
            template_supports={tpl: len(indices) for tpl, indices in template_to_indices.items()},
            answers=answers,
            answer_counter=answer_counter,
            lengths=lengths,
            unique_ratios=unique_ratios,
        )

        template_stats = self._build_template_stats(
            template_to_indices=template_to_indices,
            answers=answers,
            questions=questions,
            difficulties=difficulties,
        )

        hard_samples = self._select_hard_samples(
            questions=questions,
            answers=answers,
            img_paths=img_paths,
            templates=templates,
            difficulties=difficulties,
            reason_scores=reason_scores,
        )

        return TemplateExtractionReport(
            dataset_name=self.config.dataset_name,
            templates=template_stats,
            hard_samples=hard_samples,
            structural_tokens=sorted(structural_tokens),
        )

    # ------------------------------------------------------------------
    # Data resolution helpers
    # ------------------------------------------------------------------
    def _resolve_dataset(self, dataset_or_batch) -> tuple[List[str], List[str], List[str]]:
        if isinstance(dataset_or_batch, BaseDataset):
            data = dataset_or_batch.data
            return data["questions"], data["answers"], data["img_paths"]

        if isinstance(dataset_or_batch, dict):
            return (
                list(dataset_or_batch.get("questions", [])),
                list(dataset_or_batch.get("answers", [])),
                list(dataset_or_batch.get("img_paths", [])),
            )

        raise TypeError(
            "dataset_or_batch must be a BaseDataset instance or a dict with questions/answers/img_paths"
        )

    # ------------------------------------------------------------------
    # Token processing
    # ------------------------------------------------------------------
    def _tokenize(self, question: str) -> List[str]:
        lowered = question.lower()
        return self.TOKEN_PATTERN.findall(lowered)

    def _normalize_tokens(self, tokens: Sequence[str]) -> List[str]:
        normalized = []
        for token in tokens:
            if self.NUM_PATTERN.fullmatch(token):
                normalized.append("<num>")
                continue

            mapped = self.special_token_map.get(token)
            if mapped is not None:
                normalized.append(mapped)
                continue

            normalized.append(token)
        return normalized

    def _build_template(self, tokens: Sequence[str], structural_tokens: Sequence[str]) -> str:
        structural = set(structural_tokens)
        template_tokens: List[str] = []
        slot_idx = 1
        for token in tokens:
            if token in self.PUNCTUATION:
                template_tokens.append(token)
                continue

            if token in structural:
                template_tokens.append(token)
                continue

            placeholder = f"<slot_{slot_idx}>"
            template_tokens.append(placeholder)
            slot_idx += 1

        template = " ".join(template_tokens)
        template = re.sub(r"\s+([?!.,;:/])", r"\1", template)
        template = template.replace("( ", "(").replace(" )", ")")
        template = re.sub(r"\s+", " ", template).strip()
        return template

    def _select_structural_tokens(self, normalized_tokens: Sequence[Sequence[str]]) -> List[str]:
        counter = Counter()
        for tokens in normalized_tokens:
            for token in tokens:
                counter[token] += 1

        unique_count = len(counter)
        if unique_count == 0:
            return list(self.PUNCTUATION)

        desired_count = max(1, int(unique_count * self.config.structural_token_percentile))
        desired_count = min(desired_count, self.config.max_structural_tokens)
        structural_tokens = [token for token, _ in counter.most_common(desired_count)]

        structural_tokens.extend(self.special_token_map.values())
        structural_tokens.append("<num>")
        structural_tokens.extend(self.PUNCTUATION)

        # Deduplicate while preserving order
        seen = set()
        ordered_tokens = []
        for token in structural_tokens:
            if token not in seen:
                ordered_tokens.append(token)
                seen.add(token)
        return ordered_tokens

    # ------------------------------------------------------------------
    # Difficulty scoring
    # ------------------------------------------------------------------
    def _score_samples(
        self,
        templates: Sequence[str],
        template_supports: Dict[str, int],
        answers: Sequence[str],
        answer_counter: Counter,
        lengths: Sequence[int],
        unique_ratios: Sequence[float],
    ) -> tuple[List[float], List[Dict[str, float]]]:
        support_values = list(template_supports.values()) or [0]
        answer_values = list(answer_counter.values()) or [0]
        length_values = list(lengths) or [0]
        unique_values = list(unique_ratios) or [0]

        min_support, max_support = min(support_values), max(support_values)
        min_answer, max_answer = min(answer_values), max(answer_values)
        min_length, max_length = min(length_values), max(length_values)
        min_unique, max_unique = min(unique_values), max(unique_values)

        difficulties: List[float] = []
        reasons: List[Dict[str, float]] = []
        weights = self.weights.as_dict()

        for template, answer, length, unique_ratio in zip(templates, answers, lengths, unique_ratios):
            support = template_supports.get(template, 1)
            template_rarity = self._invert_scale(support, min_support, max_support)
            answer_frequency = answer_counter[answer]
            answer_rarity = self._invert_scale(answer_frequency, min_answer, max_answer)

            length_score = self._scale(length, min_length, max_length)
            unique_score = self._scale(unique_ratio, min_unique, max_unique)
            question_complexity = (0.6 * length_score) + (0.4 * unique_score)

            difficulty = (
                weights["template_rarity"] * template_rarity
                + weights["answer_rarity"] * answer_rarity
                + weights["question_complexity"] * question_complexity
            )
            difficulty = max(0.0, min(1.0, difficulty))

            difficulties.append(difficulty)
            reasons.append(
                {
                    "template_rarity": round(template_rarity, 4),
                    "answer_rarity": round(answer_rarity, 4),
                    "question_complexity": round(question_complexity, 4),
                }
            )

        return difficulties, reasons

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    def _build_template_stats(
        self,
        template_to_indices: Dict[str, List[int]],
        answers: Sequence[str],
        questions: Sequence[str],
        difficulties: Sequence[float],
    ) -> List[TemplateStats]:
        stats: List[TemplateStats] = []
        for template, indices in template_to_indices.items():
            support = len(indices)
            if support < self.config.min_template_support:
                continue

            avg_difficulty = sum(difficulties[i] for i in indices) / support
            first_idx = indices[0]
            stats.append(
                TemplateStats(
                    template=template,
                    support=support,
                    avg_difficulty=round(avg_difficulty, 4),
                    sample_indices=list(indices),
                    representative_question=questions[first_idx],
                    representative_answer=answers[first_idx],
                )
            )
        stats.sort(key=lambda item: item.avg_difficulty, reverse=True)
        return stats

    def _select_hard_samples(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        img_paths: Sequence[str],
        templates: Sequence[str],
        difficulties: Sequence[float],
        reason_scores: Sequence[Dict[str, float]],
    ) -> List[HardSample]:
        threshold = self._percentile_threshold(difficulties, self.config.hardness_percentile)
        ranked_indices = [
            idx for idx, score in enumerate(difficulties) if score >= threshold
        ]
        ranked_indices.sort(key=lambda idx: difficulties[idx], reverse=True)
        ranked_indices = ranked_indices[: self.config.max_hard_samples]

        hard_samples: List[HardSample] = []
        for idx in ranked_indices:
            augmentations = self._suggest_augmentations(reason_scores[idx])
            hard_samples.append(
                HardSample(
                    index=idx,
                    question=questions[idx],
                    answer=answers[idx],
                    img_path=img_paths[idx],
                    template=templates[idx],
                    difficulty=round(difficulties[idx], 4),
                    reason_scores=reason_scores[idx],
                    suggested_augmentations=augmentations,
                )
            )
        return hard_samples

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _scale(value: float, min_value: float, max_value: float) -> float:
        if max_value == min_value:
            return 0.0
        return (value - min_value) / (max_value - min_value)

    @staticmethod
    def _invert_scale(value: float, min_value: float, max_value: float) -> float:
        return 1.0 - QuestionTemplateExtractor._scale(value, min_value, max_value)

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _percentile_threshold(values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return sorted_values[0]
        percentile = min(max(percentile, 0.0), 1.0)
        position = percentile * (len(sorted_values) - 1)
        lower_index = int(math.floor(position))
        upper_index = min(len(sorted_values) - 1, lower_index + 1)
        interpolation = position - lower_index
        return sorted_values[lower_index] * (1 - interpolation) + sorted_values[upper_index] * interpolation

    def _suggest_augmentations(self, reason_scores: Dict[str, float]) -> List[str]:
        suggestions: List[str] = []
        template_rarity = reason_scores.get("template_rarity", 0.0)
        answer_rarity = reason_scores.get("answer_rarity", 0.0)
        question_complexity = reason_scores.get("question_complexity", 0.0)

        if template_rarity >= 0.6:
            suggestions.append("text")
        if answer_rarity >= 0.6:
            suggestions.append("image")
        if question_complexity >= 0.6 and "text" not in suggestions:
            suggestions.append("text")

        if not suggestions:
            suggestions.append("image")
        return suggestions

    def _build_special_token_map(self) -> Dict[str, str]:
        if not self.config.include_special_maps:
            return {}

        color_terms = {
            "red", "blue", "green", "yellow", "black", "white",
            "orange", "purple", "pink", "brown", "gray"
        }
        direction_terms = {"left", "right", "top", "bottom", "front", "back"}
        quantity_terms = {"many", "several", "few"}

        special_map: Dict[str, str] = {}
        for term in color_terms:
            special_map[term] = "<color>"
        for term in direction_terms:
            special_map[term] = "<direction>"
        for term in quantity_terms:
            special_map[term] = "<quantity>"

        return special_map
