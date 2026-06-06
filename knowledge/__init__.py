"""
医学知识库模块

提供基于 Milvus 的向量数据库封装
"""
# from .chroma_kb import KnowledgeBase
# __all__ = ['KnowledgeBase']
from .chroma_kb import ChromaKnowledgeBase as KnowledgeBase
__all__ = ['KnowledgeBase']
