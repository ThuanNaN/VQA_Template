import os
from google import genai
from google.genai import types
from models.output_question import OutputQuestion
import dotenv
from PIL import Image

class VlmAugmentClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = genai.Client(api_key=self.api_key)

    def generate_questions(self, image: Image.Image, question: str, lamda: float = 1.0) -> str:
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            config=types.GenerateContentConfig(
                system_instruction="""You are a paraphrasing tool inspired by SapAugment.  
                I will give you a sentence and a parameter λ ∈ [0,1].  

                Rules:
                - The higher λ → the more paraphrases you should generate, with stronger changes in wording and structure.  
                - The lower λ → fewer paraphrases, with minimal changes.  

                Mapping:
                - λ ∈ [0.0, 0.2]: generate 1 paraphrase, very light modification (almost identical to original).  
                - λ ∈ (0.2, 0.5]: generate 2 paraphrases, small to moderate changes.  
                - λ ∈ (0.5, 0.8]: generate 3 paraphrases, moderate to strong changes.  
                - λ ∈ (0.8, 1.0]: generate 5 paraphrases, strong changes (new vocabulary, different structure), but **preserve meaning**.  

                Output:
                - Always return a numbered list of paraphrased sentences.  
                - Do not add explanations, just the sentences.  

                Example Input:
                λ = 0.75  
                Sentence: "The quick brown fox jumps over the lazy dog."

                Example Output:
                1. The fast brown fox leaps over the lazy dog.  
                2. A swift fox with brown fur jumps across the sluggish dog.  
                3. The brown fox quickly hops over a tired dog.  
                """,
                response_schema=OutputQuestion,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
            contents=[
            image,
            f"λ = {lamda}\nSentence: {question}"
        ]
        )
        return response.text
