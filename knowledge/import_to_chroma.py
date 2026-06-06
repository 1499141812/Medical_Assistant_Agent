#!/usr/bin/env python
# knowledge/scripts/import_to_chroma.py
import sys
from pathlib import Path

def find_project_root(start_path=None):
    """向上查找包含 'knowledge' 子目录的目录作为项目根"""
    if start_path is None:
        start_path = Path(__file__).resolve().parent
    current = start_path
    while current != current.parent:
        if (current / "knowledge").exists():
            return current
        current = current.parent
    raise RuntimeError("无法找到项目根目录（包含 knowledge 目录的父目录）")

# 确定项目根目录
PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

# 现在可以导入项目模块
from chroma_kb import KnowledgeBase

def load_documents_from_directory(doc_dir_path):
    """从目录加载所有 .txt 文件"""
    doc_dir = Path(doc_dir_path)
    if not doc_dir.exists():
        raise FileNotFoundError(f"目录不存在: {doc_dir_path}")
    
    docs = []
    for txt_file in sorted(doc_dir.glob("*.txt")):
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        doc = {
            "content": content,
            "metadata": {
                "source": txt_file.name,
                "type": infer_type_from_filename(txt_file.name)
            }
        }
        docs.append(doc)
    return docs

def infer_type_from_filename(filename):
    name = filename.lower()
    if "lifestyle" in name:
        return "lifestyle"
    elif "icd" in name or "disease" in name:
        return "disease_classification"
    elif "guideline" in name:
        return "clinical_guideline"
    else:
        return "general"

def main():
    doc_dir = PROJECT_ROOT / "knowledge" / "data" / "documents"
    print(f"📚 加载文档目录: {doc_dir}")
    if not doc_dir.exists():
        print(f"❌ 目录不存在，请先创建: {doc_dir}")
        return
    
    docs = load_documents_from_directory(doc_dir)
    if not docs:
        print("❌ 未找到任何 .txt 文件")
        return
    
    print(f"✅ 加载了 {len(docs)} 个文档")
    type_counts = {}
    for d in docs:
        t = d["metadata"]["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, cnt in type_counts.items():
        print(f"   - {t}: {cnt}")
    
    print("\n💾 正在导入到 ChromaDB...")
    kb = KnowledgeBase()
    num = kb.add_documents(docs)
    print(f"✅ 成功导入 {num} 条文档到 ChromaDB")

if __name__ == "__main__":
    main()