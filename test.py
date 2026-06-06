# #!/usr/bin/env python
# # test_kb_and_skill.py
# import sys
# from pathlib import Path

# # 将项目根目录添加到 Python 路径（确保能导入 knowledge 等模块）
# project_root = Path(__file__).resolve().parent
# sys.path.insert(0, str(project_root))

# # 现在可以导入知识库模块
# from knowledge.chroma_kb import KnowledgeBase

# # 技能的实际路径：.claude/skills/search-knowledge/script/search.py
# # 我们直接导入该模块（注意路径中的连字符需要处理）
# # 方法1：用 importlib 动态导入（推荐，因为文件名/路径有连字符）
# import importlib.util
# skill_path = project_root / ".claude" / "skills" / "search-knowledge" / "script" / "search.py"
# spec = importlib.util.spec_from_file_location("search_knowledge_skill", skill_path)
# skill_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(skill_module)
# search_knowledge = skill_module.search_knowledge  # 异步函数

# async def test():
#     # 1. 测试知识库直接检索
#     print("=" * 60)
#     print("测试知识库直接检索")
#     kb = KnowledgeBase()
#     results = kb.search("糖尿病饮食", top_k=2)
#     if results:
#         print("检索成功，第一个文档字段:", results[0].keys())
#         print("内容预览:", results[0]['text'][:100])
#     else:
#         print("检索结果为空，请检查知识库是否有数据")

#     # 2. 测试 search_knowledge skill
#     print("\n" + "=" * 60)
#     print("测试 search_knowledge skill")
#     result = await search_knowledge("糖尿病饮食", max_results=3)
#     print("返回的 answer 长度:", len(result['answer']))
#     print("前200字符:\n", result['answer'][:200])

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(test())
from core.config import LLM_CONFIG
print(LLM_CONFIG)
