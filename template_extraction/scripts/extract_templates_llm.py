"""
pt trích xuất template tổng quát từ các câu hỏi khó bằng LLM.
"""
import logging
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_template(template: str) -> str:
    """
    Chuẩn hóa template để gộp các template giống nhau.
    
    - Loại bỏ khoảng trắng thừa
    - Chuyển về lowercase
    - Thêm dấu ? nếu chưa có
    - Loại bỏ dấu câu thừa
    """
    # Loại bỏ khoảng trắng thừa
    template = ' '.join(template.split())
    
    # Chuyển về lowercase
    template = template.lower().strip()
    
    # Loại bỏ dấu ? cuối nếu có
    template = template.rstrip('?').rstrip()
    
    # Loại bỏ dấu chấm cuối nếu có
    template = template.rstrip('.').rstrip()
    
    return template


def extract_templates_with_classification(
    samples: List[Dict],
    llm_client,
    question_types: List[str],
    min_count: int = 2,
    batch_delay: float = 1.0,
    max_templates_per_type: int = None,
    min_templates_per_type: int = 0
) -> Dict[str, List[Dict]]:
    """
    Trích xuất template từ các câu hỏi và gom nhóm theo loại.
    
    Args:
        samples: Danh sách các hard samples
        llm_client: Client để gọi LLM
        question_types: Danh sách các loại câu hỏi
        min_count: Số lần xuất hiện tối thiểu để giữ template
        batch_delay: Delay giữa các lần gọi API (giây)
        max_templates_per_type: Số lượng templates tối đa cho mỗi loại (None = không giới hạn)
        min_templates_per_type: Số lượng templates tối thiểu cho mỗi loại (default: 0)
        
    Returns:
        Dict mapping từ question_type → list of templates
    """
    # Step 1: Classify và extract template cho từng câu hỏi
    logger.info(f"Bước 1: Phân loại và trích xuất template cho {len(samples)} câu hỏi...")
    
    question_data = []
    for sample in tqdm(samples, desc="Processing questions"):
        question_text = sample['question_text']
        
        # Classify question type if not already present
        if 'question_type' not in sample or not sample['question_type']:
            question_type = llm_client.classify_question_type(question_text, question_types)
            sample['question_type'] = question_type
            time.sleep(batch_delay)  # Rate limiting
        else:
            question_type = sample['question_type']
        
        # Extract template
        template = llm_client.extract_template(question_text)
        time.sleep(batch_delay)  # Rate limiting
        
        question_data.append({
            'question_text': question_text,
            'template': template,
            'question_type': question_type,
            'gt_answer': sample.get('gt_answer', ''),
            'entropy': sample.get('entropy', 0.0),
            'confidence': sample.get('confidence', 0.0)
        })
    
    logger.info(f"✅ Hoàn thành phân loại và trích xuất template")
    
    # Step 2: Group by question type and template
    logger.info("Bước 2: Gom nhóm template theo loại câu hỏi...")
    
    template_groups = defaultdict(lambda: defaultdict(list))
    
    for item in question_data:
        qtype = item['question_type']
        template = item['template']
        
        # Chuẩn hóa template để gộp các template giống nhau
        normalized_template = normalize_template(template)
        
        template_groups[qtype][normalized_template].append(item['question_text'])
    
    # Step 3: Count and filter templates
    logger.info(f"Bước 3: Lọc template có tần suất >= {min_count}...")
    
    final_templates = {}
    total_templates_before = 0
    total_templates_after = 0
    
    for qtype, templates_dict in template_groups.items():
        total_templates_before += len(templates_dict)
        
        filtered_templates = []
        for template, examples in templates_dict.items():
            count = len(examples)
            if count >= min_count:
                # Take up to 5 examples
                sample_examples = examples[:min(5, len(examples))]
                
                filtered_templates.append({
                    'template': template,
                    'count': count,
                    'examples': sample_examples
                })
        
        # Sort by count descending
        filtered_templates.sort(key=lambda x: x['count'], reverse=True)
        
        # Apply max_templates_per_type limit
        original_count = len(filtered_templates)
        if max_templates_per_type is not None and len(filtered_templates) > max_templates_per_type:
            filtered_templates = filtered_templates[:max_templates_per_type]
            logger.info(f"  - {qtype}: Limited from {original_count} to {max_templates_per_type} templates (max_per_type)")
        
        # Check min_templates_per_type requirement
        if len(filtered_templates) < min_templates_per_type:
            logger.warning(f"  ⚠️  {qtype}: Only {len(filtered_templates)} templates (< min_per_type={min_templates_per_type})")
        
        # Only add question type if it meets minimum requirement
        if len(filtered_templates) >= min_templates_per_type:
            final_templates[qtype] = filtered_templates
            total_templates_after += len(filtered_templates)
            logger.info(f"  - {qtype}: {len(templates_dict)} → {len(filtered_templates)} templates ✅")
        else:
            logger.info(f"  - {qtype}: {len(templates_dict)} → {len(filtered_templates)} templates ❌ (removed - below min)")
    
    logger.info(f"✅ Tổng số template: {total_templates_before} → {total_templates_after}")
    logger.info(f"✅ Số loại câu hỏi còn lại: {len(final_templates)}")
    
    return final_templates
