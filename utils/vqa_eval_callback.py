"""
VQA Evaluation Callback for VLM training.
Computes accuracy via generation after each epoch.
"""

import torch
from transformers import TrainerCallback

from utils.vlm_utils import clear_memory


class VQAEvalCallback(TrainerCallback):
    """
    Callback to evaluate VQA accuracy after each epoch.
    Generates answers and compares with ground truth (giống train.py).
    """
    
    def __init__(
        self, 
        processor, 
        eval_dataset, 
        device: str = "cuda",
        max_eval_samples: int = None,
        max_new_tokens: int = 32
    ):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.device = device
        self.max_eval_samples = max_eval_samples
        self.max_new_tokens = max_new_tokens
        
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.best_accuracy = 0.0
        self.history = []  # Store metrics history
        
    def on_epoch_end(self, args, state, control, **kwargs):
        self.current_epoch = int(state.epoch)
        
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Called after trainer.evaluate() - compute VQA accuracy here."""
        if metrics is None:
            return
            
        eval_loss = metrics.get('eval_loss', 0)
        
        # Compute VQA accuracy via generation
        print(f"\n🔄 Computing VQA accuracy (generating answers)...")
        accuracy_results = self._compute_vqa_accuracy(model)
        accuracy = accuracy_results['accuracy']
        correct = accuracy_results['correct']
        total = accuracy_results['total']
        
        # Add to metrics (modify in place)
        metrics['eval_accuracy'] = accuracy
        metrics['eval_correct'] = correct
        metrics['eval_total'] = total
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"📊 EPOCH {self.current_epoch} EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"   Eval Loss: {eval_loss:.4f}")
        print(f"   Eval Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            print(f"   ✅ New best eval loss!")
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print(f"   ✅ New best accuracy!")
        
        print(f"{'='*60}\n")
        
        # Store history
        self.history.append({
            'epoch': self.current_epoch,
            'eval_loss': eval_loss,
            'eval_accuracy': accuracy,
            'correct': correct,
            'total': total
        })
    
    def _compute_vqa_accuracy(self, model) -> dict:
        """Compute VQA accuracy via generation."""
        model.eval()
        
        # Get model dtype for casting inputs
        model_dtype = next(model.parameters()).dtype
        
        # Determine if we need autocast (for mixed precision)
        use_autocast = model_dtype in [torch.float16, torch.bfloat16]
        
        correct = 0
        total = 0
        
        num_samples = len(self.eval_dataset)
        
        with torch.no_grad():
            for idx in range(num_samples):
                try:
                    # Get conversation and raw item
                    conversation = self.eval_dataset[idx]
                    raw_item = self.eval_dataset.get_raw_item(idx)
                    
                    # Prepare input (system + user only, no assistant)
                    text = self.processor.apply_chat_template(
                        conversation[0:2],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    # Get image - ensure it's in a list for processor
                    image = conversation[1]["content"][0]["image"]
                    
                    # Process inputs
                    inputs = self.processor(
                        text=[text],
                        images=[image],  # IMPORTANT: processor expects list of images
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # Move inputs to device and ensure correct dtype
                    processed_inputs = {}
                    for k, v in inputs.items():
                        v = v.to(self.device)
                        # Cast pixel_values and any float tensors to model dtype
                        if k == 'pixel_values' or v.dtype in [torch.float32, torch.float64, torch.float16]:
                            v = v.to(model_dtype)
                        processed_inputs[k] = v
                    inputs = processed_inputs
                    
                    # Generate with autocast if using mixed precision
                    if use_autocast:
                        with torch.cuda.amp.autocast(dtype=model_dtype):
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=self.max_new_tokens,
                                do_sample=False,
                                pad_token_id=self.processor.tokenizer.pad_token_id
                            )
                    else:
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.pad_token_id
                        )
                    
                    # Decode
                    generated_text = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )[0]
                    
                    # Extract answer
                    pred_answer = self._extract_answer(generated_text)
                    gt_answer = str(raw_item['answer']).strip().lower()
                    
                    # Check exact match
                    if pred_answer == gt_answer:
                        correct += 1
                    total += 1
                    
                    # Clear memory periodically
                    del inputs, generated_ids
                    if idx % 50 == 0 and idx > 0:
                        clear_memory()
                        
                except Exception as e:
                    import traceback
                    if idx < 3:  # Print first 3 errors
                        print(f"\n  Error at sample {idx}: {str(e)}")
                        traceback.print_exc()
                    total += 1
                    continue
        
        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy, "correct": correct, "total": total}
    
    def _extract_answer(self, generated_text: str) -> str:
        """Extract the answer from generated text."""
        text = generated_text.strip()
        
        # Try to extract after "assistant" marker
        if "assistant" in text.lower():
            parts = text.lower().split("assistant")
            text = parts[-1].strip()
        
        # Remove common prefixes
        prefixes = ["answer:", "the answer is", "câu trả lời:", "câu trả lời là", "đáp án:", "đáp án là"]
        text_lower = text.lower()
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        text = text.strip().lower()
        
        # Remove quotes
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
        
        return text