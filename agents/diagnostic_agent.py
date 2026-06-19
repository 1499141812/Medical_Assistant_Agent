"""
DiagnosticAgent：症状诊断推理 Agent

这是第一个 WorkerAgent 实现，展示如何：
1. 参与 Swarm 协作
2. 自主认领任务
3. 调用医疗工具
4. 将结果写入 SharedContext
"""
from typing import Dict, Any, Optional
from loguru import logger

from .base_agent import BaseAgent
from .skill_registry_mixin import SkillRegistryMixin
from core import LLMClient


class DiagnosticAgent(BaseAgent, SkillRegistryMixin):
    """
    诊断 Agent

    职责：
    - 复杂症状的鉴别诊断
    - 多系统关联分析
    - 诊断思路推理（类似医生的临床思维）

    能力标签：
    - symptom_analysis
    - differential_diagnosis
    - clinical_reasoning
    """

    def __init__(
        self,
        agent_id: str = "diagnostic_agent",
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None
    ):
        config = config or {}
        config.setdefault('max_iterations', 5)

        super().__init__(agent_id, config, llm_client)

        # 设置能力标签（Swarm 协作用）
        self.set_capabilities([
            "symptom_analysis",
            "differential_diagnosis",
            "clinical_reasoning",
            "multi_system_analysis"
        ])

    def register_tools(self):
        """注册所有 9 个 Skills（共享实现，来自 SkillRegistryMixin）"""
        self.register_all_skills()


    def get_system_prompt(self) -> str:
        """获取系统提示词（精简版）"""
        return """你是专业的诊断推理 Agent。职责：分析症状模式、生成鉴别诊断、评估可能性。

可用 Skills: search_knowledge, recommend_lifestyle, assess_risk, analyze_symptoms, disease_code, clinical_guideline, deep_research, search_history, search_similar_cases

策略：先用 assess_risk 评估风险，再用 analyze_symptoms 分析模式。最多2-3次 Skill 调用后给出诊断思路。
永远不做确诊，只提供诊断推理。常见病优先但勿忽略危险疾病。

输出格式：
【风险评估】风险等级+紧急程度
【症状分析】症状类别+关联性
【鉴别诊断】1. 诊断A（可能性%）- 支持/反对证据 2. 诊断B ...
【建议检查】检查项目列表
【推理过程】简述推理逻辑
"""

    async def post_process_result(
        self,
        result: Dict[str, Any],
        final_response: str
    ) -> Dict[str, Any]:
        """
        结果后处理：提取结构化诊断信息

        这里可以添加更复杂的解析逻辑
        """
        # 尝试提取风险等级
        risk_level = "unknown"
        if "风险等级" in final_response:
            if "高" in final_response or "HIGH" in final_response:
                risk_level = "high"
            elif "中" in final_response or "MEDIUM" in final_response:
                risk_level = "medium"
            elif "低" in final_response or "LOW" in final_response:
                risk_level = "low"

        result.update({
            "risk_level": risk_level,
            "diagnosis_provided": True
        })

        return result


# 便捷函数
async def diagnose(question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    便捷函数：快速使用 DiagnosticAgent

    Args:
        question: 症状描述
        context: 额外上下文（年龄、既往史等）

    Returns:
        诊断结果
    """
    agent = DiagnosticAgent()
    input_data = {'question': question}
    if context:
        input_data['context'] = context

    return await agent.process(input_data)
