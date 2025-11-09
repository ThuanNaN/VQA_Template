"""
Các hàm tiện ích để load/save dữ liệu cho pipeline trích xuất template.
"""
import json
from typing import List, Dict, Any
from pathlib import Path


def load_jsonl(path: str) -> List[Dict]:
    """
    Load dữ liệu từ file JSONL.
    
    Args:
        path: Đường dẫn tới file .jsonl
        
    Returns:
        Danh sách các dictionary đọc được
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str) -> None:
    """
    Lưu dữ liệu vào file JSONL.
    
    Args:
        data: Danh sách các dictionary cần lưu
        path: Đường dẫn tới file .jsonl
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_json(data: Dict[str, Any], path: str) -> None:
    """
    Lưu dữ liệu vào file JSON.
    
    Args:
        data: Dictionary cần lưu
        path: Đường dẫn tới file .json
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """
    Load dữ liệu từ file JSON.
    
    Args:
        path: Đường dẫn tới file .json
        
    Returns:
        Dictionary đọc được
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
