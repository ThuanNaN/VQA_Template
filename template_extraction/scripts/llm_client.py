"""
LLM Client để phân loại câu hỏi và trích xuất template tiếng Việt.
Sử dụng liteLLM để gọi các LLM model khác nhau.
"""
import litellm
from typing import List
import logging
from augmentation.models.output_question import ListQuestions
from template_extraction.models.output_template import OutputTemplate

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
        
        system_prompt = f"""Phân loại câu hỏi VQA vào ĐÚNG 1 loại trong: {types_str}

QUY TẮC (kiểm tra theo thứ tự):

1. màu_sắc: Có "màu", "màu sắc", "tone màu"
   ✓ Chiếc xe màu gì?
   ✓ Màu của áo là gì?
   ✓ Bức tường có màu gì?

2. đếm: Có "bao nhiêu", "mấy", "số lượng", "có ... không" (đếm số)
   ✓ Có bao nhiêu người?
   ✓ Số xe buýt là bao nhiêu?
   ✓ Có mấy con chó?

3. có_không: Kết thúc "không?", "phải không?", "có ... không?"
   ✓ Có người trong ảnh không?
   ✓ Đây là bãi biển phải không?
   ✓ Người đàn ông đang cười phải không?

4. hành_động: Có "làm gì", "đang làm", "chơi gì", động từ hỏi hành động
   ✓ Người này đang làm gì?
   ✓ Họ đang chơi gì?
   ✓ Cậu bé đang ăn gì?

5. vị_trí: Có "ở đâu", "chỗ nào", "tại đâu", "nơi nào"
   ✓ Cái bàn ở đâu?
   ✓ Bức ảnh chụp ở đâu?
   ✓ Chiếc xe đậu chỗ nào?

6. đối_tượng: Hỏi "là gì", "loại gì", "con gì", "cái gì" (hỏi tên/loại)
   ✓ Đây là gì?
   ✓ Loại chim gì?
   ✓ Con vật này tên là gì?

7. trạng_thái: Hỏi trạng thái, tính chất, "hay" (lựa chọn)
   ✓ Pin yếu hay mạnh?
   ✓ Đèn bật hay tắt?
   ✓ Thời tiết thế nào?

Chỉ trả về TÊN LOẠI (ví dụ: màu_sắc, đếm, hành_động), không giải thích."""
        
        user_prompt = f'Câu hỏi: "{question}"\nLoại:'
        
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
            
            result = response.choices[0].message.content.strip().lower()
            
            # Map English names back to Vietnamese if needed
            english_to_vietnamese = {
                "color": "màu_sắc",
                "count": "đếm",
                "object": "đối_tượng",
                "action": "hành_động",
                "location": "vị_trí",
                "yes_no": "có_không",
                "state": "trạng_thái",
            }
            
            # Check if result is already valid
            if result in question_types:
                return result
            
            # Try to map from English
            result_lower = result.lower()
            if result_lower in english_to_vietnamese:
                mapped = english_to_vietnamese[result_lower]
                if mapped in question_types:
                    return mapped
            
            # Try to find partial match
            for qt in question_types:
                if qt in result or result in qt:
                    return qt
            
            # Check if any English keyword in result
            for eng, vie in english_to_vietnamese.items():
                if eng in result_lower and vie in question_types:
                    return vie
            
            logger.warning(f"LLM returned invalid type: {result}, defaulting to first type")
            return question_types[0]
            
        except Exception as e:
            logger.error(f"Error classifying question: {e}")
            return question_types[0]  # Default fallback
    
    def correct_template(self, raw_template: str, original_question: str) -> str:
        """
        Sửa lỗi và chuẩn hóa template đã trích xuất.
        
        Args:
            raw_template: Template thô từ bước extraction
            original_question: Câu hỏi gốc để tham khảo
            
        Returns:
            Template đã được sửa lỗi và chuẩn hóa
        """
        system_prompt = """Sửa lỗi template VQA. CHỈ sửa những lỗi rõ ràng, GIỮ NGUYÊN phần đúng.

LỖI CẦN SỬA:
1. Bỏ classifiers: "Con gì" → "[đối_tượng] gì", "cái [đối_tượng]" → "[đối_tượng]"
2. Bỏ "các" trước [số_lượng]: "các [số_lượng]" → "[số_lượng]"
3. Chữ hoa đầu câu → chữ thường
4. Dấu câu cuối (?, !, .) → bỏ
5. Placeholder sai: thay "màu" thành [màu_sắc] → sửa lại "màu" (không thay)
6. Placeholder sai: thay "số" thành [số_lượng] → sửa lại "số" (không thay)

KHÔNG SỬA (giữ nguyên nếu đúng):
- [hành_động] nếu câu gốc có động từ cụ thể (chơi bóng, nằm trên)
- [màu_sắc] nếu câu gốc có màu cụ thể
- Cấu trúc câu đúng

VÍ DỤ:

Gốc: "Cái bàn ở đâu?"
Lỗi: "cái [đối_tượng] ở đâu?"
Sửa: "[đối_tượng] ở đâu"

Gốc: "2 con chó đang ở đâu?"
Lỗi: "các [số_lượng] [đối_tượng] đang ở đâu?"
Sửa: "[số_lượng] [đối_tượng] đang ở đâu"

Gốc: "Con gì đang nằm trên ghế?"
Lỗi: "Con [đối_tượng] đang [hành_động] trên [vị_trí]"
Sửa: "[đối_tượng] gì đang [hành_động] trên [vị_trí]"

Gốc: "Chiếc xe màu gì?"
Lỗi: "[màu_sắc] của [đối_tượng]"
Sửa: "[đối_tượng] màu gì"

Gốc: "Số xe buýt là bao nhiêu?"
Lỗi: "[số_lượng] [đối_tượng] là bao nhiêu"
Sửa: "số [đối_tượng] là bao nhiêu"

Nếu template không có lỗi rõ ràng, TRẢ VỀ NGUYÊN."""

        user_prompt = f'Gốc: "{original_question}"\nLỗi: "{raw_template}"\nSửa:'
        
        try:
            response = self.client.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                api_key=self.api_key,
                response_format=OutputTemplate,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON
            import json
            try:
                data = json.loads(content)
                corrected = data.get("template", "").strip()
            except json.JSONDecodeError:
                corrected = content
            
            # Clean up
            corrected = corrected.strip('"').rstrip('?.!').lower()
            
            return corrected
            
        except Exception as e:
            logger.error(f"Error correcting template: {e}")
            return raw_template  # Return raw template if correction fails
    
    def extract_template(self, question: str) -> str:
        """
        Trích xuất template tổng quát tiếng Việt từ câu hỏi.
        
        Args:
            question: Câu hỏi cần trích xuất template
            
        Returns:
            Template với các placeholder
        """
        system_prompt = """Trích xuất template VQA. Thay CHÍNH XÁC theo các ví dụ.

THAY:
- Danh từ cụ thể (xe, mèo, người, bàn, áo, chó, bình) → [đối_tượng]
- Số cụ thể (2, 3, ba) → [số_lượng]
- Màu CỤ THỂ (đỏ, xanh, vàng) → [màu_sắc]
- Động từ có tân ngữ (chơi bóng, nằm trên ghế, mặc áo) → [hành_động]
- Vị trí cụ thể (ghế, sân, phòng) → [vị_trí]

GIỮ NGUYÊN (TUYỆT ĐỐI không thay):
- Từ hỏi: gì, nào, đâu
- Từ chỉ định: này, đó, những, các
- Cụm: có bao nhiêu, số (từ), màu (từ), làm gì, ở đâu, màu gì

BỎ (classifiers):
- con, chiếc, cái

LƯU Ý QUAN TRỌNG:
❌ SAI: "con [đối_tượng]" → ✅ ĐÚNG: "[đối_tượng]" (phải bỏ "con")
❌ SAI: "[đối_tượng] này" → ✅ ĐÚNG: "những [đối_tượng] này" (phải giữ "những")
❌ SAI: "[màu_sắc]" trong "màu gì" → ✅ ĐÚNG: "màu gì" (không thay)
❌ SAI: "[số_lượng]" thay "có bao nhiêu" → ✅ ĐÚNG: "có bao nhiêu" (giữ nguyên)

Ví dụ:

Con mèo đang làm gì? → [đối_tượng] đang làm gì
Chiếc xe màu gì? → [đối_tượng] màu gì
Cái bàn ở đâu? → [đối_tượng] ở đâu
Những người này đang làm gì? → những [đối_tượng] này đang làm gì
Những người đó đang làm gì? → những [đối_tượng] đó đang làm gì
Các người này đang chơi gì? → các [đối_tượng] này đang [hành_động] gì
Các con mèo đó ở đâu? → các [đối_tượng] đó ở đâu
Có bao nhiêu người đang chơi bóng? → có bao nhiêu [đối_tượng] đang [hành_động]
Màu của chiếc bình là gì? → màu của [đối_tượng] là gì
Số xe buýt là bao nhiêu? → số [đối_tượng] là bao nhiêu
2 con chó đang ở đâu? → [số_lượng] [đối_tượng] đang ở đâu
Con gì đang nằm trên ghế? → [đối_tượng] gì đang [hành_động] trên [vị_trí]
Cậu bé mặc áo màu gì? → [đối_tượng] mặc [đối_tượng] màu gì

Chỉ trả template, chữ thường."""
        
        user_prompt = f'Q: {question}\nA:'
        
        try:
            response = self.client.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                api_key=self.api_key,
                response_format=OutputTemplate,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON structured output
            import json
            try:
                data = json.loads(content)
                template = data.get("template", "").strip()
            except json.JSONDecodeError:
                # Fallback: treat as plain text
                template = content
            
            # Remove quotes if present
            if template.startswith('"') and template.endswith('"'):
                template = template[1:-1]
            
            # Remove trailing punctuation
            template = template.rstrip('?.!')
            
            # Correct template with second LLM call
            # template = self.correct_template(template, question)
            
            return template
            
        except Exception as e:
            logger.error(f"Error extracting template: {e}")
            return question  # Return original question as fallback
