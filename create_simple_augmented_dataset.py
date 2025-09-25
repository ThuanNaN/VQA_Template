import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import time
import logging
from pathlib import Path
import dotenv

from dataset import ViVQADataset, OpenViVQADataset, ViVQAXDataset
from augmentation.augment_client import VlmAugmentClient
from augmentation.translator import Translator
from transformers import AutoTokenizer, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def parse_translated_qa_pairs(translated_data):
    """
    Parse translated question-answer pairs from the translation output.
    Handles the format: [{'question': 'vi: ...', 'answer': 'vi: ...'}, ...]
    
    Args:
        translated_data: List of dictionaries or string containing QA pairs
    
    Returns:
        List of dictionaries with cleaned question and answer pairs
    """
    parsed_pairs = []
    
    # If translated_data is a string, try to parse as JSON
    if isinstance(translated_data, str):
        try:
            translated_data = json.loads(translated_data)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse translated data as JSON: {translated_data[:100]}...")
            return []
    
    # Handle list of QA pairs
    if isinstance(translated_data, list):
        for item in translated_data:
            if isinstance(item, dict) and 'question' in item and 'answer' in item:
                question = item['question']
                answer = item['answer']
                
                # Remove language prefix if present
                if question.startswith('vi: '):
                    question = question[4:].strip()
                elif question.startswith('en: '):
                    question = question[4:].strip()
                
                if answer.startswith('vi: '):
                    answer = answer[4:].strip()
                elif answer.startswith('en: '):
                    answer = answer[4:].strip()
                
                # Clean up formatting issues
                question = question.strip()
                answer = answer.strip()
                
                # Handle parsing errors in answers (like "5. THỰC HIỆN...")
                if '. ' in answer and len(answer.split('. ', 1)) > 1:
                    potential_extra = answer.split('. ', 1)[1]
                    # If the part after the first period looks like unwanted text
                    if any(word in potential_extra.upper() for word in ['THỰC HIỆN', 'CHƯƠNG TRÌNH', 'DỰ ÁN']):
                        answer = answer.split('. ', 1)[0]
                
                parsed_pairs.append({
                    'question': question,
                    'answer': answer
                })
    
    return parsed_pairs

def parse_augmented_question(augmented_text, aug_idx=None):
    """
    Parse augmented questions from Gemini response.
    Gemini returns numbered list based on lambda value.
    
    Args:
        augmented_text: Text response from Gemini
        aug_idx: If specified, return only the question at this index. If None, return all questions.
    
    Returns:
        If aug_idx is None: List of all parsed questions
        If aug_idx is specified: Single question at that index
    """
    lines = augmented_text.strip().split('\n')
    questions = []
    
    # Extract numbered questions
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            # Remove numbering (1., 2., etc.)
            question = line.split('.', 1)[-1].strip()
            if question:
                questions.append(question)
    
    # If no numbered questions found, try to extract from plain text
    if not questions:
        for line in lines:
            if line.strip() and not line.startswith('λ') and not line.startswith('Sentence:'):
                questions.append(line.strip())
    
    # Return based on aug_idx parameter
    if aug_idx is None:
        return questions  # Return all questions
    else:
        # Return the requested question index, or first one if index out of range
        if questions:
            return questions[min(aug_idx, len(questions) - 1)]
        return "What is in the image?"  # Ultimate fallback

def generate_simple_augmented_dataset(
    dataset, 
    gemini_client, 
    translator, 
    output_path, 
    num_samples=None, 
    num_augmentations_per_sample=3, 
    lambda_value=0.7,
    request_delay=2.0,
    use_translation=False,
):
    """
    Generate augmented dataset with improved translation parsing.
    Process all samples or a specified number of samples.
    
    Args:
        dataset: Dataset to augment
        gemini_client: VLM augmentation client
        translator: Translation client (will be used automatically)
        output_path: Path to save augmented dataset
        num_samples: Number of samples to process (None = all samples)
        num_augmentations_per_sample: Number of augmentations per sample
        lambda_value: Lambda value for augmentation intensity
        request_delay: Delay between API requests
        use_translation: Whether to use translation augmentation (auto enabled)
    """
    augmented_data = []
    failed_augmentations = 0
    api_calls_count = 0
    translation_calls_count = 0
    
    # Auto enable translation
    use_translation = True
    
    # Determine how many samples to process
    total_samples = len(dataset)
    if num_samples is None or num_samples > total_samples:
        samples_to_process = total_samples
    else:
        samples_to_process = num_samples
    
    logger.info(f"Processing {samples_to_process} samples from {total_samples} total samples")
    logger.info(f"Generating {num_augmentations_per_sample} augmentations per sample")
    logger.info(f"Lambda value: {lambda_value}")
    logger.info(f"Request delay: {request_delay}s")
    logger.info(f"Translation enabled: Each LLM-generated question will be automatically translated")
    
    for idx in tqdm(range(samples_to_process), desc="Generating augmented samples"):
        try:
            # Get original data
            original_data = dataset.data
            img_path = original_data['img_paths'][idx]
            original_question = original_data['questions'][idx]
            original_answer = original_data['answers'][idx]
            
            # Load image (keep for potential future use)
            from PIL import Image
            pil_image = Image.open(img_path).convert('RGB')
            
            logger.debug(f"Processing sample {idx}: {original_question[:50]}...")
            
            # === VLM-based augmentation ===
            try:
                # Generate augmented questions using VLM client
                # Pass image path (string) to client, not PIL Image object
                augmented_questions_text = gemini_client.generate_questions(img_path)
                api_calls_count += 1
                
                # Try to parse as JSON first (like in augment_client example)
                try:
                    list_qa = json.loads(augmented_questions_text)
                    # If JSON, extract questions from the structured response
                    if isinstance(list_qa, dict):
                        all_questions = [item for item in list_qa.get('questions', [])]
                    else:
                        # Fallback to text parsing
                        all_questions = parse_augmented_question(augmented_questions_text, aug_idx=None)
                except json.JSONDecodeError:
                    # If not JSON, parse as text
                    all_questions = parse_augmented_question(augmented_questions_text, aug_idx=None)
                
                # Use up to num_augmentations_per_sample questions
                questions_to_use = all_questions[:num_augmentations_per_sample]
                
                # Add delay between requests
                if request_delay > 0:
                    time.sleep(request_delay)
                
            except Exception as e:
                logger.warning(f"VLM augmentation failed for sample {idx}: {e}")
                failed_augmentations += num_augmentations_per_sample
                if request_delay > 0:
                    time.sleep(request_delay)  # Still sleep on error
            
            # === Translation augmentation (Auto enabled) ===
            if translator and len(questions_to_use) > 0:
                try:
                    logger.debug(f"Translating {len(questions_to_use)} questions for sample {idx}")
                    
                    try:
                        translated_text = translator.translate_qa_pairs(questions_to_use, src_lang='en')
                            
                        translation_calls_count += 1
                            
                        # Parse translated output - could be string or list of QA pairs
                        parsed_qa_pairs = parse_translated_qa_pairs(translated_text)
                            
                        # Use parsed QA pairs
                        for qa_pair in parsed_qa_pairs:
                            translation_entry = {
                                "img_path": str(img_path),
                                "augmented_question": qa_pair['question'],
                                "answer": qa_pair['answer']  # Use translated answer
                            }
                            print(f"{translation_entry}")  # Debug print
                            augmented_data.append(translation_entry)
                        
                        # Small delay between individual translations
                        if request_delay > 0:
                            time.sleep(request_delay / 2)
                                
                    except Exception as e:
                        logger.warning(f"Translation failed for question in sample {idx}: {e}")
                        failed_augmentations += 1
                        continue
                    
                except Exception as e:
                    logger.warning(f"Translation batch failed for sample {idx}: {e}")
                    failed_augmentations += len(questions_to_use)
            
        except Exception as e:
            logger.error(f"Failed to process sample {idx}: {e}")
            total_augmentations_for_sample = num_augmentations_per_sample + (num_augmentations_per_sample if use_translation else 0)
            failed_augmentations += total_augmentations_for_sample
            continue
    
    # Save augmented dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving augmented dataset to {augmented_data}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2, default=convert_numpy)
    
    logger.info(f"Saved {len(augmented_data)} augmented samples to {output_path}")
    
    logger.info(f"Used {api_calls_count} VLM API calls")
    logger.info(f"Used {translation_calls_count} Translation API calls")
    logger.info(f"Failed augmentations: {failed_augmentations}")
    
    expected_total = samples_to_process * (num_augmentations_per_sample + (num_augmentations_per_sample if use_translation else 0))
    if expected_total > 0:
        logger.info(f"Success rate: {(len(augmented_data)/expected_total)*100:.1f}%")
    
    return augmented_data

def main():
    parser = argparse.ArgumentParser(description="Generate simple augmented dataset with auto translation")
    parser.add_argument('--dataset_name', type=str, choices=['ViVQA', 'OpenViVQA', 'ViVQA-X'],
                        default='ViVQA', help='Dataset name')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to process (default: 5)')
    parser.add_argument('--num_augmentations_per_sample', type=int, default=6,
                        help='Number of paraphrase augmentations per sample (default: 6)')
    parser.add_argument('--lambda_value', type=float, default=0.7,
                        help='Lambda value for augmentation intensity (0.0-1.0, default: 0.7)')
    parser.add_argument('--request_delay', type=float, default=2.0,
                        help='Delay in seconds between API requests (default: 2.0)')
    parser.add_argument('--output_dir', type=str, default='simple_augmented_datasets',
                        help='Output directory for augmented datasets')
    parser.add_argument('--translator_model', type=str, default='VietAI/envit5-translation',
                        help='Translation model name')
    parser.add_argument('--vis_model_name', type=str, default='google/vit-base-patch16-224',
                        help='Vision model name')
    parser.add_argument('--text_model_name', type=str, default='vinai/bartpho-syllable-base',
                        help='Text model name')
    
    args = parser.parse_args()
    
    # Setup
    logger.info(f"Simple dataset augmentation starting...")
    logger.info(f"Dataset: {args.dataset_name}")
    
    # Load processors for dataset
    vis_processor = AutoProcessor.from_pretrained(args.vis_model_name, use_fast=True)
    text_processor = AutoTokenizer.from_pretrained(args.text_model_name)
    
    # Load dataset
    if args.dataset_name == 'ViVQA':
        dataset = ViVQADataset(
            ann_path="data/vivqa/train.csv",
            img_dir="data/vivqa/images",
            text_processor=text_processor,
            vis_processor=vis_processor,
            max_length=64
        )
    elif args.dataset_name == 'OpenViVQA':
        dataset = OpenViVQADataset(
            ann_path="data/openvivqa/vlsp2023_train_data.json",
            img_dir="data/openvivqa/training-images",
            text_processor=text_processor,
            vis_processor=vis_processor,
            max_length=64
        )
    elif args.dataset_name == 'ViVQA-X':
        dataset = ViVQAXDataset(
            ann_path="data/vivqa-x/ViVQA-X_train.json",
            img_dir="data/MSCOCO/train2014",
            text_processor=text_processor,
            vis_processor=vis_processor,
            max_length=64
        )
    
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    
    # Initialize VLM client
    logger.info("Initializing VLM client...")
    dotenv.load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")  # Keep same env var for consistency
    if not api_key:
        api_key = "your_api_key_here"  # Fallback for ollama
        logger.warning("GEMINI_API_KEY not found, using default for ollama")
    
    gemini_client = VlmAugmentClient(api_key=api_key)
    
    # Initialize translator (always enabled)
    logger.info(f"Initializing translator: {args.translator_model}")
    try:
        translator = Translator(model_name=args.translator_model)
        logger.info("Translator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize translator: {e}")
        logger.info("Continuing without translation...")
        translator = None
    
    # Generate augmented dataset
    logger.info("Generating augmented dataset with auto translation...")
    output_dir = Path(args.output_dir) / args.dataset_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    augmented_output_path = output_dir / f"{args.dataset_name}_simple_augmented.json"
    augmented_data = generate_simple_augmented_dataset(
        dataset=dataset,
        gemini_client=gemini_client,
        translator=translator,
        output_path=augmented_output_path,
        num_samples=args.num_samples,
        num_augmentations_per_sample=args.num_augmentations_per_sample,
        lambda_value=args.lambda_value,
        request_delay=args.request_delay,
        use_translation=True,  # Always enabled
    )
    
    # Save metadata
    metadata = {
        "dataset_name": args.dataset_name,
        "total_original_samples": len(dataset),
        "processed_samples": args.num_samples if args.num_samples else len(dataset),
        "num_augmentations_per_sample": args.num_augmentations_per_sample,
        "lambda_value": args.lambda_value,
        "use_translation": True,  # Always enabled
        "translator_model": args.translator_model,
        "total_augmentations_generated": len(augmented_data),
        "request_delay": args.request_delay,
        "augmentation_methods": {
            "vlm_paraphrase": True,
            "translation": True,  # Always enabled
            "enhanced_qa_parsing": True
        },
        "vis_model_name": args.vis_model_name,
        "text_model_name": args.text_model_name
    }
    
    metadata_path = output_dir / f"{args.dataset_name}_simple_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=convert_numpy)
    
    logger.info(f"Saved metadata to {metadata_path}")
    logger.info("Simple dataset augmentation completed!")

# Test function for parsing
def test_parsing():
    """Test the parsing function with sample data"""
    sample_data = [
        {'question': 'vi: Có bình hoa trong hình không?', 'answer': 'vi: có'},
        {'question': 'vi: Có một chiếc xe trong hình không?', 'answer': 'vi: không'},
        {'question': 'vi: Vật trong bình là gì?', 'answer': 'vi: hoacolor'},
        {'question': 'vi: Bối cảnh là gì?', 'answer': 'vi: ngoài'},
        {'question': 'vi: Hoạt động đó là gì?', 'answer': 'vi: cuộc sống tĩnh'},
        {'question': 'vi: Có bao nhiêu bông hoa?', 'answer': 'vi: 5. THỰC HIỆN CÁC CHƯƠNG TRÌNH, DỰ ÁN ĐẦU TƯ'}
    ]
    
    parsed = parse_translated_qa_pairs(sample_data)
    print("=== Test Results ===")
    for i, pair in enumerate(parsed):
        print(f"{i+1}. Q: {pair['question']}")
        print(f"   A: {pair['answer']}")
    
    return parsed

if __name__ == "__main__":
    # Uncomment to test parsing
    # test_parsing()
    
    main()