from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator:
    def __init__(self, model_name="VietAI/envit5-translation", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device = device

    def translate_qa_pairs(self, qa_list, src_lang="en", max_length=128):
        """
        qa_list: list of dict [{"question": "...", "answer": "..."}, ...]
        return: translated list with same structure
        """
        # Prepare texts with language prefix
        texts = [f"{src_lang}: {item['question']}" for item in qa_list] + \
                [f"{src_lang}: {item['answer']}" for item in qa_list]

        # Encode batch
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        # Generate
        outputs = self.model.generate(encoded.input_ids, max_length=max_length)

        # Decode
        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Ghép lại thành list dict
        n = len(qa_list)
        translated_data = []
        for i in range(n):
            translated_data.append({
                "question": translations[i],
                "answer": translations[i + n]
            })

        return translated_data