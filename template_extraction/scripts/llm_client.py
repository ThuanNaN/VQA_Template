"""
LLM Client để phân loại câu hỏi và trích xuất template tiếng Việt.
Sử dụng liteLLM để gọi các LLM model khác nhau.
"""
import litellm
from typing import List
import logging

logger = logging.getLogger(__name__)


class TemplateLLMClient:
    """Client để gọi LLM cho việc phân loại câu hỏi và trích xuất template."""
    
    def __init__(self, api_key: str, model: str = "gemini/gemini-1.5-flash"):
        """
        Khởi tạo LLM client.
        
        Args:
            api_key: API key cho LLM service
            model: Tên model (ví dụ: gemini/gemini-1.5-flash, gpt-4o-mini)
        """
        self.api_key = api_key
        self.model = model
        self.client = litellm
    
    def classify_question_type(
        self, 
        question: str, 
        question_types: List[str]
    ) -> str:
        """
        Phân loại loại câu hỏi tiếng Việt.
        
        Args:
            question: Câu hỏi cần phân loại
            question_types: Danh sách các loại câu hỏi có thể
            
        Returns:
            Tên loại câu hỏi
        """
        types_str = ", ".join(question_types)
        
        system_prompt = f"""Bạn là một công cụ phân loại loại câu hỏi trong bài toán VQA tiếng Việt.
Hãy phân loại câu hỏi vào một trong các loại sau: [{types_str}].
Chỉ trả về tên loại, không kèm giải thích."""
        
        user_prompt = f'Câu hỏi: "{question}"'
        
        try:
            response = self.client.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Deterministic
                api_key=self.api_key
            )
            
            result = response.choices[0].message.content.strip()
            
            # Validate result
            if result not in question_types:
                # Try to find closest match
                for qt in question_types:
                    if qt in result or result in qt:
                        return qt
                logger.warning(f"LLM returned invalid type: {result}, defaulting to first type")
                return question_types[0]
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying question: {e}")
            return question_types[0]  # Default fallback
    
    def extract_template(self, question: str) -> str:
        """
        Trích xuất template tổng quát tiếng Việt từ câu hỏi.
        
        Args:
            question: Câu hỏi cần trích xuất template
            
        Returns:
            Template với các placeholder
        """
        system_prompt = """Bạn là một công cụ trích xuất mẫu câu hỏi (template) tiếng Việt cho bài toán VQA.
Hãy thay thế các danh từ riêng, đối tượng cụ thể, màu sắc, số lượng, địa điểm bằng placeholder trong ngoặc vuông.

Ví dụ:
- "Chiếc xe màu đỏ đang ở đâu?" → "Chiếc [đối_tượng] màu [màu_sắc] đang ở đâu?"
- "Có bao nhiêu người đang chơi bóng?" → "Có bao nhiêu [đối_tượng] đang [hành_động]?"
- "Con mèo nằm trên ghế sofa phải không?" → "Con [đối_tượng] nằm trên [vị_trí] phải không?"

Chỉ trả về template tiếng Việt, không kèm giải thích."""
        
        user_prompt = f'Câu hỏi: "{question}"'
        
        try:
            response = self.client.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Slightly creative
                api_key=self.api_key
            )
            
            template = response.choices[0].message.content.strip()
            
            # Remove quotes if present
            if template.startswith('"') and template.endswith('"'):
                template = template[1:-1]
            
            return template
            
        except Exception as e:
            logger.error(f"Error extracting template: {e}")
            return question  # Return original question as fallback
