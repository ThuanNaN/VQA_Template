"""
Pipeline hoàn chỉnh để trích xuất template từ VQA dataset.

Pipeline gồm 3 bước:
1. Tính độ khó (entropy, confidence) cho từng câu hỏi bằng model baseline
2. Chọn ra các câu hỏi khó (top N%)
3. Trích xuất template bằng LLM và gom nhóm theo loại câu hỏi

Chạy toàn bộ pipeline trong một file với đầy đủ tham số.
"""
import argparse
import torch
import logging
from pathlib import Path
import sys

from dataset.viocrvqa import ViOCRVQADataset

# Add template_extraction/scripts to path
template_scripts_path = Path(__file__).parent / "template_extraction" / "scripts"
sys.path.insert(0, str(template_scripts_path))

from dataset import ViVQADataset, OpenViVQADataset, ViVQAXDataset
from models import SimpleVQA, SimpleVQAConfig
from transformers import AutoTokenizer, AutoProcessor

# Import functions from template_extraction/scripts
from utils import load_jsonl, save_jsonl, save_json
from llm_client import TemplateLLMClient
from compute_difficulty import compute_difficulty_scores
from select_hard_samples import select_hard_samples
from extract_templates_llm import extract_templates_with_classification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline hoàn chỉnh: Inference → Hard Sample Selection → Template Extraction"
    )
    
    # ===== Model & Dataset =====
    parser.add_argument('--checkpoint_path', type=str, required=False,
                        help='Đường dẫn tới checkpoint model baseline')
    parser.add_argument('--dataset_name', type=str, default='ViVQA',
                        choices=['ViVQA', 'OpenViVQA', 'ViVQA-X', 'ViOCRVQA'],
                        help='Tên dataset')
    parser.add_argument('--vis_model_name', type=str, default='google/vit-base-patch16-224',
                        help='Vision model name')
    parser.add_argument('--text_model_name', type=str, default='vinai/bartpho-syllable-base',
                        help='Text model name')
    parser.add_argument('--seq_len', type=int, default=64,
                        help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size cho inference')
    
    # ===== Hard Sample Selection =====
    parser.add_argument('--top_percent', type=float, default=0.1,
                        help='Tỷ lệ phần trăm mẫu khó cần lấy (0.0-1.0)')
    parser.add_argument('--selection_method', type=str, default='entropy',
                        choices=['entropy', 'confidence'],
                        help='Phương pháp chọn mẫu khó')
    
    # ===== Template Extraction =====
    parser.add_argument('--api_key', type=str, required=False,
                        help='API key cho LLM service')
    parser.add_argument('--llm_model', type=str, default='ollama/gemma2:9b',
                        help='Tên model LLM')
    parser.add_argument('--question_types', type=str, nargs='+',
                        default=['màu_sắc', 'đối_tượng', 'hành_động', 
                                'vị_trí', 'lý_do', 'trạng_thái', 'thời_gian'],
                        help='Danh sách loại câu hỏi tiếng Việt')
    parser.add_argument('--min_count', type=int, default=2,
                        help='Số lần xuất hiện tối thiểu để giữ template')
    parser.add_argument('--batch_delay', type=float, default=1.0,
                        help='Delay giữa các lần gọi API (giây)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Số lượng mẫu tối đa để xử lý (None = tất cả)')
    parser.add_argument('--max_templates_per_type', type=int, default=12,
                        help='Số lượng templates TỐI ĐA cho mỗi loại câu hỏi (None = không giới hạn)')
    parser.add_argument('--min_templates_per_type', type=int, default=0,
                        help='Số lượng templates TỐI THIỂU cho mỗi loại câu hỏi (default: 0)')
    
    # ===== Output =====
    parser.add_argument('--output_dir', type=str, default='template_extraction/data',
                        help='Thư mục lưu kết quả (sẽ tự động thêm tên dataset)')
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create output directory with dataset name
    output_dir = Path(args.output_dir) / args.dataset_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # ===== STEP 1: Compute Difficulty =====
    logger.info("\n" + "="*60)
    logger.info("STEP 1: COMPUTING DIFFICULTY SCORES")
    logger.info("="*60)
    
    # Load processors
    text_processor = AutoTokenizer.from_pretrained(args.text_model_name)
    vis_processor = AutoProcessor.from_pretrained(args.vis_model_name)
    
    # Load dataset
    logger.info(f"Loading {args.dataset_name} dataset...")
    if args.dataset_name == 'ViVQA':
        dataset = ViVQADataset(
            ann_path="data/vivqa/train.csv",
            img_dir="data/vivqa/images",
            text_processor=text_processor,
            vis_processor=vis_processor,
            max_length=args.seq_len
        )
    elif args.dataset_name == 'OpenViVQA':
        dataset = OpenViVQADataset(
            ann_path="data/vivqa/train.csv",
            img_dir="data/vivqa/images",
            text_processor=text_processor,
            vis_processor=vis_processor,
            max_length=args.seq_len
        )
    elif args.dataset_name == 'ViVQA-X':
        dataset = ViVQAXDataset(
            ann_path="data/vivqa-x/ViVQA-X_train.json",
            img_dir="data/MSCOCO/train2014",
            text_processor=text_processor,
            vis_processor=vis_processor,
            max_length=args.seq_len
        )
    elif args.dataset_name == 'ViOCRVQA':
        dataset = ViOCRVQADataset(
            ann_path="data/viocrvqa/train.json",
            img_dir="data/viocrvqa/images",
            text_processor=text_processor,
            vis_processor=vis_processor,
            max_length=args.seq_len
        )
    
    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint_path}")
    config = SimpleVQAConfig(
        vis_model_name=args.vis_model_name,
        text_model_name=args.text_model_name,
        num_classes=len(dataset.label_encoder)
    )
    model = SimpleVQA(config)

    if args.checkpoint_path:
        # Check if checkpoint path is a directory or file
        checkpoint_path = Path(args.checkpoint_path)
        if checkpoint_path.is_dir():
            model_file = checkpoint_path / "pytorch_model.bin"
        else:
            model_file = checkpoint_path
        
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
    
    # Compute difficulty
    difficulty_scores = compute_difficulty_scores(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device
    )
    
    # ===== STEP 2: Select Hard Samples =====
    logger.info("\n" + "="*60)
    logger.info("STEP 2: SELECTING HARD SAMPLES")
    logger.info("="*60)
    
    hard_samples = select_hard_samples(
        samples=difficulty_scores,
        top_percent=args.top_percent,
        method=args.selection_method
    )
    
    # Đánh dấu is_hard cho tất cả samples
    hard_sample_ids = set(s['question_id'] for s in hard_samples)
    for sample in difficulty_scores:
        sample['is_hard'] = sample['question_id'] in hard_sample_ids
    
    # Lưu tất cả samples với cờ is_hard
    all_samples_path = output_dir / "all_samples_with_difficulty.jsonl"
    save_jsonl(difficulty_scores, str(all_samples_path))
    logger.info(f"Saved all samples with difficulty scores to: {all_samples_path}")
    logger.info(f"  - Total samples: {len(difficulty_scores)}")
    logger.info(f"  - Hard samples (is_hard=True): {len(hard_samples)}")
    
    # Limit samples if specified
    if args.max_samples is not None:
        hard_samples = hard_samples[:args.max_samples]
        logger.info(f"Limiting to {len(hard_samples)} samples for processing")
    
    # ===== STEP 3: Extract Templates =====
    logger.info("\n" + "="*60)
    logger.info("STEP 3: EXTRACTING TEMPLATES WITH LLM")
    logger.info("="*60)
    
    llm_client = TemplateLLMClient(
        api_key=args.api_key,
        model=args.llm_model
    )
    
    templates = extract_templates_with_classification(
        samples=hard_samples,
        llm_client=llm_client,
        question_types=args.question_types,
        min_count=args.min_count,
        batch_delay=args.batch_delay,
        max_templates_per_type=args.max_templates_per_type,
        min_templates_per_type=args.min_templates_per_type
    )
    
    # Save templates
    templates_path = output_dir / "templates.json"
    save_json(templates, str(templates_path))
    logger.info(f"Saved templates to: {templates_path}")
    
    # ===== Summary =====
    logger.info("\n" + "="*60)
    logger.info("📊 PIPELINE SUMMARY")
    logger.info("="*60)
    logger.info(f"Total samples processed: {len(difficulty_scores)}")
    logger.info(f"Hard samples selected: {len(hard_samples)}")
    logger.info(f"Template extraction results:")
    for qtype, template_list in templates.items():
        total_count = sum(t['count'] for t in template_list)
        logger.info(f"  - {qtype}: {len(template_list)} templates ({total_count} questions)")
    logger.info(f"\n✅ Pipeline hoàn tất!")
    logger.info(f"\n📁 Output files:")
    logger.info(f"  - All samples with difficulty: {all_samples_path}")
    logger.info(f"  - Templates: {templates_path}")


if __name__ == '__main__':
    main()
