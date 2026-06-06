# knowledge/chroma_kb.py
import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

# knowledge/chroma_kb.py
import os
import chromadb
from typing import List, Dict, Any, Optional, Union

class ChromaKnowledgeBase:
    def __init__(self, persist_dir: str = "./knowledge/data/chroma_db"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="medical_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(
        self, 
        docs: Optional[List[Any]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> int:
        """
        添加文档。支持两种调用方式：
        1. add_documents(docs) 其中 docs 是对象列表，每个对象有 page_content (或 content/text) 和 metadata 属性/字典
        2. add_documents(texts=[...], metadatas=[...], ids=[...])
        """
        # 情况1：传入了 docs 列表（对象列表）
        if docs is not None:
            if not isinstance(docs, list):
                raise TypeError("docs must be a list")
            texts = []
            metadatas = []
            ids = []
            for i, doc in enumerate(docs):
                # 提取文本内容：支持多种常见属性名
                if hasattr(doc, 'page_content'):
                    text = doc.page_content
                elif hasattr(doc, 'content'):
                    text = doc.content
                elif isinstance(doc, dict) and ('content' in doc or 'text' in doc):
                    text = doc.get('content') or doc.get('text')
                else:
                    raise ValueError(f"Document object must have 'page_content', 'content', or be a dict with 'content'/'text'")
                
                # 提取元数据
                if hasattr(doc, 'metadata'):
                    meta = doc.metadata.copy() if doc.metadata else {}
                elif isinstance(doc, dict) and 'metadata' in doc:
                    meta = doc['metadata'].copy() if doc['metadata'] else {}
                else:
                    meta = {}
                
                # 确保 type 字段存在（如果没有，尝试从文档类型推断或设为 'general'）
                if 'type' not in meta:
                    # 可以从文档来源文件名或其他信息推断，这里简单设为 'general'
                    meta['type'] = 'general'
                
                texts.append(text)
                metadatas.append(meta)
                ids.append(f"doc_{i}")
        
        # 情况2：传入 texts, metadatas, ids（原有方式）
        else:
            if texts is None:
                raise ValueError("Either docs or texts must be provided")
            if metadatas is None:
                metadatas = [{}] * len(texts)
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(texts))]
        
        # 批量添加到 Chroma
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        return len(texts)
    
    def search(self, query: str, top_k: int = 5, filter_type: str = None, **kwargs):
        """检索相似文档，可按类型过滤（元数据中的 'type' 字段）"""
        where_filter = None
        if filter_type:
            where_filter = {"type": filter_type}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        output = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                output.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "score": 1 - results['distances'][0][i]
                })
        return output
    
    def delete_collection(self):
        self.client.delete_collection("medical_knowledge")


KnowledgeBase = ChromaKnowledgeBase