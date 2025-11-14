"""Example: extract question templates and surface hard samples.

This script demonstrates how to inspect an existing dataset, generate
question templates, and export a payload that can be fed into downstream
augmentation jobs.
"""

import json
import sys
from pathlib import Path

sys.path.append("..")

from dataset import ViVQADataset
from transformers import AutoTokenizer, AutoImageProcessor
from template import (
    QuestionTemplateExtractor,
    TemplateExtractionConfig,
)


def main():
    # Load processors (same as training pipeline)
    text_model = "vinai/bartpho-syllable-base"
    vision_model = "google/vit-base-patch16-224"
    text_processor = AutoTokenizer.from_pretrained(text_model)
    vis_processor = AutoImageProcessor.from_pretrained(vision_model)

    dataset = ViVQADataset(
        ann_path=ViVQADataset.train_ann,
        img_dir="data/vivqa/images",
        text_processor=text_processor,
        vis_processor=vis_processor,
        max_length=128,
    )

    extractor = QuestionTemplateExtractor(
        TemplateExtractionConfig(
            dataset_name="vivqa",
            min_template_support=3,
            hardness_percentile=0.85,
            max_hard_samples=50,
        )
    )

    report = extractor.extract(dataset)
    print(f"Dataset: {report.dataset_name}")
    print(f"Templates (>= support {extractor.config.min_template_support}): {len(report.templates)}")

    print("\nTop templates by difficulty:")
    for stats in report.templates[:5]:
        print(f"  - {stats.template} | support={stats.support} | avg_difficulty={stats.avg_difficulty}")

    hard_payload = report.build_augmentation_payload()
    print(f"\nHard samples surfaced: {len(hard_payload)}")
    for sample in hard_payload[:5]:
        print(
            f"  - idx={sample['index']} | diff={sample['difficulty']} | template={sample['template']}"
        )

    output_path = Path("runs/template_analysis/vivqa_hard_samples.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(hard_payload, ensure_ascii=False, indent=2))
    print(f"\nSaved augmentation payload to {output_path}")


if __name__ == "__main__":
    main()
