import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import time
import logging
from pathlib import Path
import dotenv
import csv

from dataset import ViVQADataset, OpenViVQADataset, ViVQAXDataset
from augmentation.augment_client import VlmAugmentClient
from augmentation.translator import Translator
from transformers import AutoTokenizer, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === API Key Management ===
def load_api_keys(file_path='api_keys.txt'):
    """Load API keys from file, one per line. Ignore comments and empty lines."""
    api_keys = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    api_keys.append(line)
    return api_keys

def call_with_retry(func, max_retries=3, retry_delay=10, api_keys=None, current_key_idx=0):
    """
    Call a function with retry logic. If it fails, retry for 10 seconds.
    If still fails, switch to next API key and try again.
    
    Args:
        func: Function to call (should accept api_key parameter)
        max_retries: Max retries per API key
        retry_delay: Delay in seconds before retry
        api_keys: List of API keys
        current_key_idx: Current API key index
        
    Returns:
        (result, new_key_idx) - Result from function and updated key index
    """
    if not api_keys or len(api_keys) == 0:
        raise ValueError("No API keys available")
    
    total_keys = len(api_keys)
    keys_tried = 0
    
    while keys_tried < total_keys:
        api_key = api_keys[current_key_idx]
        
        for attempt in range(max_retries):
            try:
                result = func(api_key)
                return result, current_key_idx
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        # All retries failed, switch to next key
        keys_tried += 1
        if keys_tried < total_keys:
            current_key_idx = (current_key_idx + 1) % total_keys
            logger.warning(f"Switching to API key #{current_key_idx + 1}")
    
    raise Exception("All API keys exhausted")

def convert_numpy(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def parse_translated_questions(translated_data):
    """
    Parse translated question-answer pairs from the translation output.
    Handles the format: [{'question': 'vi: ...'}, ...]
    
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
            print(f"Could not parse translated data as JSON: {translated_data[:100]}...")
            return []
    
    # Handle list of QA pairs
    if isinstance(translated_data, list):
        for item in translated_data:
            if isinstance(item, dict) and 'question' in item:
                question = item['question']
                
                # Remove language prefix if present
                if question.startswith('vi: '):
                    question = question[4:].strip()
                elif question.startswith('en: '):
                    question = question[4:].strip()
                
                # Clean up formatting issues
                question = question.strip()
                
                parsed_pairs.append({
                    'question': question,
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
    dataset=None, 
    image_dir: str = None,
    gemini_client=None, 
    translator=None, 
    output_path=None, 
    num_samples=None, 
    num_augmentations_per_sample=3, 
    lambda_value=0.7,
    request_delay=2.0,
    use_translation=False,
    api_keys=None,
):
    """
    Generate augmented dataset with improved translation parsing.
    Process original samples until reaching the target number of augmented samples.
    
    Args:
        dataset: Dataset to augment
        gemini_client: VLM augmentation client
        translator: Translation client (will be used automatically)
        output_path: Path to save augmented dataset
        num_samples: Target number of augmented samples to generate (None = process all original samples)
        num_augmentations_per_sample: Number of augmentations per sample
        lambda_value: Lambda value for augmentation intensity
        request_delay: Delay between API requests
        use_translation: Whether to use translation augmentation (auto enabled)
        api_keys: List of API keys for rotation
    """
    augmented_data = []
    failed_augmentations = 0
    api_calls_count = 0
    translation_calls_count = 0
    current_key_idx = 0
    
    # Loading mapping answer file
    answer_mapping = {}
    with open('answer_mapping.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                question, answer = row
                answer_mapping[question.strip()] = answer.strip()

    # Determine source of images: dataset or image_dir
    img_paths = []
    if image_dir:
        image_dir_path = Path(image_dir)
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
        for e in exts:
            for p in sorted(image_dir_path.glob(e)):
                img_paths.append(str(p))
        total_original_samples = len(img_paths)
        print(f"Using image directory mode. Found {total_original_samples} images in {image_dir}")
    elif dataset is not None:
        total_original_samples = len(dataset)
        # dataset.data is used later to index
        img_paths = None
        print(f"Using dataset mode. Dataset has {total_original_samples} samples")
    else:
        raise ValueError("Either dataset or image_dir must be provided")

    target_augmented_samples = num_samples  # Target number of augmented samples to generate

    if target_augmented_samples is None:
        # Process all original samples
        target_augmented_samples = float('inf')
        print(f"Processing all available original samples")
    else:
        print(f"Target: Generate {target_augmented_samples} augmented samples")
    
    print(f"Generating {num_augmentations_per_sample} augmentations per original sample")
    print(f"Lambda value: {lambda_value}")
    print(f"Request delay: {request_delay}s")
    print(f"Translation enabled: Each LLM-generated question will be automatically translated")
    logger.info(f"Loaded {len(api_keys) if api_keys else 0} API keys for rotation")
    
    # Use tqdm with total as target samples (show target if finite)
    pbar_total = target_augmented_samples if target_augmented_samples != float('inf') else total_original_samples
    pbar = tqdm(total=pbar_total, desc="Generating augmented samples")

    idx = 0
    while idx < total_original_samples and len(augmented_data) < target_augmented_samples:
        try:
            # Get original data
            if img_paths is not None:
                img_path = img_paths[idx]
                original_answer = None
            else:
                original_data = dataset.data
                img_path = original_data['img_paths'][idx]
                original_answer = original_data['answers'][idx]
            
            # === VLM-based augmentation ===
            try:
                # Generate augmented questions using VLM client with retry logic
                def generate_with_key(api_key):
                    # Update client API key
                    gemini_client.api_key = api_key
                    return gemini_client.generate_questions(img_path, "vqa_aug_system_VI")
                
                # Call with retry and key rotation
                if api_keys and len(api_keys) > 0:
                    augmented_questions_text, current_key_idx = call_with_retry(
                        generate_with_key,
                        max_retries=3,
                        retry_delay=10,
                        api_keys=api_keys,
                        current_key_idx=current_key_idx
                    )
                else:
                    # Fallback to direct call if no API keys list
                    augmented_questions_text = gemini_client.generate_questions(img_path, "vqa_aug_system_EN")
                
                print(f"VLM Response: {augmented_questions_text[:200]}...")  # Debug print
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
                print(f"VLM augmentation failed for sample {idx}: {e}")
                failed_augmentations += num_augmentations_per_sample
                if request_delay > 0:
                    time.sleep(request_delay)  # Still sleep on error
            
            # === Translation augmentation (Auto enabled) ===
            if len(questions_to_use) > 0:
                if use_translation :
                    try:
                        print(f"Translating {len(questions_to_use)} questions for sample {idx}")
                        
                        try:
                            # Translation with retry logic (simple retry, no key rotation for translator)
                            max_translation_retries = 2
                            translated_text = None
                            for trans_attempt in range(max_translation_retries):
                                try:
                                    translated_text = translator.translate_qa_pairs(questions_to_use, src_lang='en')
                                    break
                                except Exception as trans_err:
                                    if trans_attempt < max_translation_retries - 1:
                                        logger.warning(f"Translation attempt {trans_attempt+1} failed: {trans_err}, retrying in 5s...")
                                        time.sleep(5)
                                    else:
                                        raise
                                
                            translation_calls_count += 1
                                
                            # Parse translated output - could be string or list of questions
                            # Prepare answers list. If original dataset provided, prefer original_answer(s).
                            # Otherwise for image-dir mode we use the generated questions as keys for mapping.
                            answers = [item["answer"] for item in questions_to_use]

                            parsed_questions = parse_translated_questions(translated_text)
                                
                            # Use parsed QA pairs
                            for question, answer in zip(parsed_questions, answers):
                                # Stop if we've reached the target
                                if len(augmented_data) >= target_augmented_samples:
                                    break
                                # Get mapped answer - try multiple fallbacks
                                mapped_answer = answer_mapping.get(answer)
                                # also try mapping by the translated question text
                                if mapped_answer is None and isinstance(question, dict):
                                    # If parsed_questions returned dicts unexpectedly
                                    qtext = question.get('question')
                                    mapped_answer = answer_mapping.get(qtext)
                                if mapped_answer is None and isinstance(question, str):
                                    mapped_answer = answer_mapping.get(question)
                                if mapped_answer is None:
                                    print(f"Skipping question - no mapping found for answer/question: {answer} / {question}")
                                    failed_augmentations += 1
                                    continue
                                
                                translation_entry = {
                                    "img_path": str(img_path),
                                    "augmented_question": question,
                                    "answer": mapped_answer
                                }
                                print(f"{translation_entry}")  # Debug print
                                augmented_data.append(translation_entry)
                                pbar.update(1)
                            
                            # Small delay between individual translations
                            if request_delay > 0:
                                time.sleep(request_delay / 2)
                                    
                        except Exception as e:
                            print(f"Translation failed for question in sample {idx}: {e}")
                            failed_augmentations += 1
                            continue
                        
                    except Exception as e:
                        print(f"Translation batch failed for sample {idx}: {e}")
                        failed_augmentations += len(questions_to_use)
                else:
                    print("Translation augmentation is disabled, skipping translation step.")
                    question = [item["question"] for item in questions_to_use]
                    answers = [item["answer"] for item in questions_to_use]
                    for q, a in zip(question, answers):
                        # Stop if we've reached the target
                        if len(augmented_data) >= target_augmented_samples:
                            break
                        # Check answer exists in answer pool
                        if a not in answer_mapping.values():
                            print(f"Skipping question - no mapping found for answer: {a}")
                            failed_augmentations += 1
                            continue
                        translation_entry = {
                           "img_path": str(img_path),
                           "augmented_question": q,
                           "answer": a
                       }
                        augmented_data.append(translation_entry)
                        pbar.update(1)


        except Exception as e:
            print(f"Failed to process sample {idx}: {e}")
            total_augmentations_for_sample = num_augmentations_per_sample + (num_augmentations_per_sample if use_translation else 0)
            failed_augmentations += total_augmentations_for_sample
        
        # Move to next original sample
        idx += 1
    
    pbar.close()
    
    # Save augmented dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving augmented dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2, default=convert_numpy)
    
    print(f"Saved {len(augmented_data)} augmented samples to {output_path}")
    print(f"Processed {idx} original samples to generate {len(augmented_data)} augmented samples")
    print(f"Used {api_calls_count} VLM API calls")
    print(f"Used {translation_calls_count} Translation API calls")
    print(f"Failed/Skipped augmentations: {failed_augmentations} (includes samples without valid answer mapping)")
    
    if target_augmented_samples != float('inf'):
        print(f"Target reached: {len(augmented_data)}/{target_augmented_samples} augmented samples (with valid answers)")
    
    return augmented_data

def main():
    parser = argparse.ArgumentParser(description="Generate simple augmented dataset with auto translation")
    parser.add_argument('--dataset_name', type=str, choices=['ViVQA', 'OpenViVQA', 'ViVQA-X'],
                        default='ViVQA', help='Dataset name')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Target number of augmented samples to generate (default: 5)')
    parser.add_argument('--num_augmentations_per_sample', type=int, default=3,
                        help='Number of paraphrase augmentations per sample (default: 3)')
    parser.add_argument('--lambda_value', type=float, default=0.7,
                        help='Lambda value for augmentation intensity (0.0-1.0, default: 0.7)')
    parser.add_argument('--request_delay', type=float, default=2.0,
                        help='Delay in seconds between API requests (default: 2.0)')
    parser.add_argument('--output_dir', type=str, default='simple_augmented_datasets',
                        help='Output directory for augmented datasets')
    parser.add_argument('--use_translation', action='store_true',
                        help='Use translation model for augmentation')
    parser.add_argument('--translator_model', type=str, default='VietAI/envit5-translation',
                        help='Translation model name')
    parser.add_argument('--vis_model_name', type=str, default='google/vit-base-patch16-224',
                        help='Vision model name')
    parser.add_argument('--text_model_name', type=str, default='vinai/bartpho-syllable-base',
                        help='Text model name')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory of images to augment (optional). If provided, dataset loading is skipped.')
    
    args = parser.parse_args()
    
    # Setup
    print(f"Simple dataset augmentation starting...")
    print(f"Dataset: {args.dataset_name}")
    
    # Load processors for dataset
    vis_processor = AutoProcessor.from_pretrained(args.vis_model_name, use_fast=True)
    text_processor = AutoTokenizer.from_pretrained(args.text_model_name)
    
    # Load dataset
    if args.dataset_name == 'ViVQA':
        dataset = ViVQADataset(
            ann_path="data/vivqa/train.csv",
            img_dir="data/vivqa/vivqa/images",
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
    
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Load API keys from file
    logger.info("Loading API keys from api_keys.txt...")
    api_keys = load_api_keys('api_keys.txt')
    if not api_keys:
        logger.warning("No API keys found in api_keys.txt, trying environment variable...")
        dotenv.load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            api_keys = [api_key]
        else:
            api_keys = ["your_api_key_here"]
            logger.warning("No API keys found, using default")
    
    logger.info(f"Loaded {len(api_keys)} API key(s)")
    
    # Initialize VLM client with first API key
    print("Initializing VLM client...")
    gemini_client = VlmAugmentClient(api_key=api_keys[0])
    
    # Initialize translator (always enabled)
    if args.use_translation:
        print(f"Initializing translator: {args.translator_model}")
        try:
            translator = Translator(model_name=args.translator_model)
            print("Translator initialized successfully")
        except Exception as e:
            print(f"Failed to initialize translator: {e}")
            print("Continuing without translation...")
            translator = None
    else:
        translator = None
        print("Translation augmentation is disabled.")
    
    # Generate augmented dataset
    print("Generating augmented dataset with auto translation...")
    output_dir = Path(args.output_dir) / args.dataset_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    augmented_output_path = output_dir / f"{args.dataset_name}_simple_augmented.json"
    augmented_data = generate_simple_augmented_dataset(
        dataset=(dataset if args.image_dir is None else None),
        image_dir=args.image_dir,
        gemini_client=gemini_client,
        translator=translator,
        output_path=augmented_output_path,
        num_samples=args.num_samples,
        num_augmentations_per_sample=args.num_augmentations_per_sample,
        lambda_value=args.lambda_value,
        request_delay=args.request_delay,
        use_translation=args.use_translation,  # Always enabled
        api_keys=api_keys,  # Pass API keys for rotation
    )
    
    # Save metadata
    metadata = {
        "dataset_name": args.dataset_name,
        "total_original_samples": len(dataset),
        "image_dir": args.image_dir,
        "target_augmented_samples": args.num_samples if args.num_samples else "all",
        "actual_augmented_samples": len(augmented_data),
        "num_augmentations_per_sample": args.num_augmentations_per_sample,
        "lambda_value": args.lambda_value,
        "use_translation": True,  # Always enabled
        "translator_model": args.translator_model,
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
    
    print(f"Saved metadata to {metadata_path}")
    print("Simple dataset augmentation completed!")

if __name__ == "__main__":

    main()