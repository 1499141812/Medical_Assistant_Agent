"""
医学知识库模块

提供基于 Chroma 的向量数据库封装
"""
from .chroma_kb import ChromaKnowledgeBase as KnowledgeBase
__all__ = ['KnowledgeBase']
