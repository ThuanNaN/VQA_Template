import os
import litellm
import base64
import json
from .translator import Translator
from .models.output_question import ListQuestions

class VlmAugmentClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = litellm

    def generate_questions(self, image_path: str, prompt_name: str) -> str:
        # Load and convert the image to base64
        image_url = self._encode_image(image_path)
        
        # Load system prompt
        system_prompt = self._load_prompt(prompt_name)

        response = self.client.completion(
            model = "gemini/gemini-flash-lite-latest",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                                    # {
                                    #     "type": "text",
                                    #     "text": system_prompt
                                    # },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image_url
                                        }
                                    }
                                ]
                }
            ],
            response_format=ListQuestions,
            temperature=1,
            api_key=self.api_key
            )
        return response.choices[0].message.content
    
    
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"
    

    def _load_prompt(self, name: str) -> str:
        # Get absolute path to the prompts directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, "prompts", f"{name}.txt")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()


# if __name__ == "__main__":
#     client = VlmAugmentClient(api_key="your_api_key_here")
#     translator = Translator()
#     response = client.generate_questions(image_path="../data/MSCOCO/train2014/COCO_train2014_000000000030.jpg")
#     print(response)
#     # try:
# #     #     list_qa = json.loads(response)
# #     #     result = translator.translate_qa_pairs(list_qa, src_lang="en")
# #     #     print(response)
# #     #     print(result)
# #     # except json.JSONDecodeError:
# #     #     print(response)
   
   

