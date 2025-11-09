🎯 Mục tiêu

Xây dựng pipeline để tự động trích xuất template tổng quát từ những câu hỏi khó trong bài toán VQA tiếng Việt.
Pipeline này không sinh dữ liệu mới, chỉ dừng ở bước phân loại loại câu hỏi và rút trích template tiếng Việt để phục vụ phân tích hoặc augmentation thủ công sau này.

🧭 Quy trình pipeline
1️⃣ Tính độ khó câu hỏi

Dùng model baseline đã huấn luyện để tính độ khó (entropy, confidence) cho từng mẫu.

Input: checkpoint model baseline + dataset gốc.

Output: file .jsonl chứa:

{
  "question_id": 12345,
  "question_text": "Chiếc xe trong ảnh có màu gì?",
  "gt_answer": "Màu đỏ",
  "image_id": "COCO_0001",
  "entropy": 2.13,
  "confidence": 0.54
}

2️⃣ Chọn mẫu câu hỏi khó

Chọn ra các câu hỏi có entropy cao hoặc confidence thấp (ví dụ top 10%).

Output: hard_samples.jsonl

3️⃣ Trích xuất template bằng LLM (bước chính)

Input: hard_samples.jsonl

Mục tiêu:

Nếu mẫu chưa có question_type, LLM sẽ phân loại loại câu hỏi theo danh sách truyền vào (ví dụ: “màu sắc”, “số lượng”, “đối tượng”, “hành động”, “vị trí”, “lý do”…).

Sau đó, LLM rút trích template tiếng Việt tổng quát bằng cách thay thế danh từ, địa điểm, đối tượng... bằng placeholder [đối tượng], [địa điểm], [màu sắc], v.v.

Gom nhóm các template theo question_type.

Đếm số lần xuất hiện của mỗi template, chỉ giữ template có tần suất ≥ N.

Mỗi template được lưu kèm một vài ví dụ thật từ dataset.

Output: templates.json, ví dụ:

{
  "màu_sắc": [
    {
      "template": "Chiếc [đối tượng] có màu gì?",
      "count": 11,
      "examples": [
        "Chiếc xe có màu gì?",
        "Con mèo trong hình có màu gì?",
        "Chiếc áo của cô gái có màu gì?"
      ]
    }
  ],
  "số_lượng": [
    {
      "template": "Có bao nhiêu [đối tượng] trong ảnh?",
      "count": 8,
      "examples": [
        "Có bao nhiêu người đang đứng?",
        "Có bao nhiêu con chó trong ảnh?",
        "Có bao nhiêu cái ghế được xếp?"
      ]
    }
  ]
}

📁 Cấu trúc thư mục
template_extraction/
│
├── baseline/
│   └── model_checkpoint/
│
├── data/
│   ├── train.json
│   ├── hard_samples.jsonl
│   └── templates.json
│
├── scripts/
│   ├── compute_difficulty.py
│   ├── select_hard_samples.py
│   ├── extract_templates_llm.py   # trích template tiếng Việt
│   └── run_pipeline.sh
│
└── copilot-context.md

🧰 Các function Copilot nên sinh code
1️⃣ Load & save utilities
load_jsonl(path: str) -> List[Dict]
save_json(data: Dict, path: str)

2️⃣ Phân loại loại câu hỏi bằng LLM
classify_question_type_vi(question: str, types: List[str], llm_model: str) -> str

3️⃣ Trích template tiếng Việt bằng LLM
extract_vietnamese_template(question: str, llm_model: str) -> str

4️⃣ Pipeline xử lý batch mẫu
extract_templates_with_classification_vi(
    samples: List[Dict],
    llm_client,
    question_types: List[str],
    min_count: int
) -> Dict[str, List[Dict]]

🧠 Prompt ví dụ cho LLM

(1) Phân loại loại câu hỏi tiếng Việt:

Bạn là một công cụ phân loại loại câu hỏi trong bài toán VQA tiếng Việt.
Câu hỏi: "Chiếc xe trong ảnh có màu gì?"
Hãy phân loại câu hỏi này vào một trong các loại sau: [màu_sắc, số_lượng, đối_tượng, hành_động, vị_trí, lý_do].
Chỉ trả về tên loại.


(2) Trích xuất template tổng quát tiếng Việt:

Bạn là một công cụ trích xuất mẫu câu hỏi (template) tiếng Việt cho bài toán VQA.
Cho câu hỏi: "Chiếc xe màu đỏ đang ở đâu?"
Hãy thay thế các danh từ riêng, đối tượng cụ thể, hoặc địa điểm bằng placeholder trong ngoặc vuông.
Ví dụ: "Chiếc [đối tượng] màu [màu_sắc] đang ở [địa_điểm]?"
Chỉ trả về template tiếng Việt, không kèm giải thích.

⚙️ Tham số CLI gợi ý cho extract_templates_llm.py
Tham số	Kiểu	Mô tả
--input	str	Đường dẫn tới file hard_samples.jsonl
--output	str	Đường dẫn để lưu templates.json
--question_types	List[str]	Danh sách loại câu hỏi tiếng Việt do user truyền vào
--min_count	int	Số lần xuất hiện tối thiểu để giữ template
--batch_size	int	Kích thước batch khi gọi LLM
--llm_model	str	Tên model LLM, ví dụ gpt-4o-mini hoặc gemini-1.5-flash
✅ Yêu cầu khi Copilot sinh code

Code cần:

Có docstring tiếng Việt đầu file mô tả mục đích script.

Sử dụng argparse cho CLI.

Có logging rõ ràng (số lượng mẫu xử lý, số template được giữ...).

Có type hint cho toàn bộ hàm.

Không hard-code đường dẫn hoặc model.

Code viết theo hướng functional pipeline: load → process → save.

Mỗi template lưu các trường: template, count, examples.