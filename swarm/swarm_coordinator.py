"""
SwarmCoordinator：Swarm 入口和智能路由

注意：这不是编排器！
- 只负责路由决策：简单问题 → 单 Agent，复杂问题 → Swarm
- 不控制 Agent 执行
- 不编排任务顺序

类比：交通信号灯，决定车辆走哪条路，但不控制车辆如何行驶
"""
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger

from core import LLMClient
from .shared_context import SharedContext
from .lead_agent import LeadAgent
from .events import Event, EventType
from agents import ConsultationAgent, DiagnosticAgent, ResearchAgent
from memory import SessionSummaryManager, SessionSummary, ShortTermMemory, LongTermMemory


# ── 快速路由：常见简单问题的关键词匹配 ──
# 跳过 LeadAgent LLM 调用，直接路由到 ConsultationAgent
_SIMPLE_QUESTION_KEYWORDS = [
    # 感冒相关
    "感冒", "着凉", "受凉", "流感", "伤风",
    # 发烧
    "发烧", "发热", "体温", "退烧", "低烧",
    # 咳嗽
    "咳嗽", "干咳", "有痰", "咳痰",
    # 头痛
    "头痛", "头疼", "偏头痛",
    # 常见轻症
    "鼻塞", "流鼻涕", "打喷嚏", "嗓子疼", "喉咙痛",
    "拉肚子", "腹泻", "便秘", "消化不良",
    "失眠", "睡不着", "睡眠不好",
    "牙疼", "牙痛", "口腔溃疡",
    "过敏", "皮肤痒", "起疹子",
    # 健康科普
    "什么是", "是什么意思", "怎么预防", "如何预防",
    "饮食注意", "吃什么", "不能吃", "多吃",
    "运动", "锻炼", "减肥", "控制体重",
    # 用药咨询（非处方）
    "布洛芬", "对乙酰氨基酚", "感冒药", "退烧药",
    # 单一简短问题（≤10字大概率是简单问题）
]


def _is_simple_question(question: str) -> bool:
    """快速判断是否为简单问题，避免 LLM 调用"""
    q = question.strip()
    # 短问题 + 包含关键词 → 简单
    if len(q) <= 15:
        return True
    # 包含简单关键词
    for kw in _SIMPLE_QUESTION_KEYWORDS:
        if kw in q:
            return True
    return False


def _is_complex_question(question: str) -> bool:
    """快速判断是否为复杂问题"""
    q = question.strip()
    complex_indicators = [
        # 多症状分隔符
        "、", "和", "加", "还有", "同时", "伴有", "并且",
        # 严重症状
        "胸痛", "胸闷", "呼吸困难", "昏厥", "晕倒", "心悸",
        "吐血", "便血", "剧烈疼痛", "视力模糊", "瘫痪",
        # 复杂关键词
        "鉴别诊断", "诊断标准", "诊疗指南", "最新研究",
        "并发症", "后遗症", "慢性病",
        # 需要分析的
        "严重吗", "需要就医吗", "要不要去医院", "会不会是",
    ]
    count = sum(1 for ind in complex_indicators if ind in q)
    return count >= 2 or len(q) > 100


class SwarmCoordinator:
    """
    Swarm 协调器

    职责：
    1. 智能路由（简单 → 单 Agent，复杂 → Swarm）
    2. 初始化 SharedContext
    3. 启动和监控 Swarm
    4. 生成 SessionSummary

    不做：
    - 不编排 Worker 执行顺序
    - 不直接调用 Worker
    - 不控制任务分配
    """

    # 简单问题的响应缓存（同会话内有效）
    _response_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        enable_swarm: bool = True
    ):
        self.llm_client = llm_client or LLMClient()
        self.enable_swarm = enable_swarm

        # 初始化 Agent（懒加载，首次使用时创建）
        self._lead_agent: Optional[LeadAgent] = None
        self._consultation_agent: Optional[ConsultationAgent] = None
        self._diagnostic_agent: Optional[DiagnosticAgent] = None
        self._research_agent: Optional[ResearchAgent] = None

        # 记忆管理器
        self.session_manager = SessionSummaryManager()
        self.short_term_memory = ShortTermMemory(storage_type="memory")
        self.long_term_memory = LongTermMemory()

        logger.info("SwarmCoordinator initialized (agents lazy-loaded)")

    # ── Agent 懒加载属性 ──
    @property
    def lead_agent(self) -> LeadAgent:
        if self._lead_agent is None:
            self._lead_agent = LeadAgent(llm_client=self.llm_client)
        return self._lead_agent

    @property
    def consultation_agent(self) -> ConsultationAgent:
        if self._consultation_agent is None:
            self._consultation_agent = ConsultationAgent()
            self._consultation_agent.loop.short_term_memory = self.short_term_memory
            self._consultation_agent.loop.max_iterations = 3
            self._consultation_agent.loop.max_tool_calls = 2
        return self._consultation_agent

    @property
    def diagnostic_agent(self) -> DiagnosticAgent:
        if self._diagnostic_agent is None:
            self._diagnostic_agent = DiagnosticAgent()
            self._diagnostic_agent.loop.short_term_memory = self.short_term_memory
            self._diagnostic_agent.loop.max_iterations = 3
            self._diagnostic_agent.loop.max_tool_calls = 2
        return self._diagnostic_agent

    @property
    def research_agent(self) -> ResearchAgent:
        if self._research_agent is None:
            self._research_agent = ResearchAgent()
            self._research_agent.loop.short_term_memory = self.short_term_memory
            self._research_agent.loop.max_iterations = 3
            self._research_agent.loop.max_tool_calls = 2
        return self._research_agent

    @property
    def worker_pool(self) -> List[Any]:
        return [self.consultation_agent, self.diagnostic_agent, self.research_agent]

    def _get_agent_by_id(self, agent_id: str):
        """根据 agent_id 返回对应的 Agent 实例"""
        mapping = {
            "consultation_agent": self.consultation_agent,
            "diagnostic_agent": self.diagnostic_agent,
            "research_agent": self.research_agent
        }
        return mapping.get(agent_id)

    async def process(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理用户问题

        Args:
            question: 用户问题
            context: 额外上下文（年龄、既往史等）
            session_id: 会话ID（如果不提供，将自动生成）

        Returns:
            处理结果
        """
        start_time = datetime.now()
        if session_id is None:
            session_id = f"{start_time.strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"

        logger.info(f"Processing question (session={session_id}): {question[:50]}...")

        # ── 响应缓存：相同问题直接返回（5分钟内有效）──
        cache_key = f"{session_id}:{question.strip()}"
        if cache_key in self._response_cache:
            cached = self._response_cache[cache_key]
            age = (start_time - cached["_cached_at"]).total_seconds()
            if age < 300:  # 5 分钟内有效
                logger.info(f"⚡ Cache hit (age={age:.0f}s)")
                cached["total_time"] = 0.01  # 缓存响应几乎瞬时
                return cached
            else:
                del self._response_cache[cache_key]

        # ── 快速路由：关键词匹配，跳过 LLM ──
        use_fast_path = _is_simple_question(question) and not _is_complex_question(question)

        # ── Swarm 关闭时：直接走 ConsultationAgent，跳过 LeadAgent ──
        if not self.enable_swarm and use_fast_path:
            logger.info(f"⚡ Swarm disabled + simple question → ConsultationAgent directly")
            result = await self.consultation_agent.process({
                'question': question,
                'context': context or {},
                'session_id': session_id
            })
            end_time = datetime.now()
            result.update({
                'swarm_enabled': False, 'session_id': session_id,
                'total_time': (end_time - start_time).total_seconds(),
                'suggestions': result.get('suggestions', []),
                'disclaimer': result.get('disclaimer',
                    "⚠️ 以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。"),
                '_cached_at': datetime.now(),
            })
            self._response_cache[cache_key] = result
            asyncio.create_task(self._save_long_term_memory(
                session_id, question, result.get('answer', ''), "disabled_fast", 0,
                result['total_time']
            ))
            return result

        # ── 简单问题：关键词快速路由 ──
        if use_fast_path:
            # 简单问题：直接路由到 ConsultationAgent，仅检索短期记忆
            logger.info(f"⚡ Fast path: routing directly to ConsultationAgent")
            recent_history = self.short_term_memory.get_recent_messages(
                session_id=session_id, limit=6
            )
            enhanced_context = context or {}
            if recent_history:
                enhanced_context["recent_history"] = [
                    {"role": msg.get("role", ""), "content": msg.get("content", "")}
                    for msg in recent_history
                ]

            result = await self.consultation_agent.process({
                'question': question,
                'context': enhanced_context,
                'session_id': session_id
            })
            final_answer = result.get('answer', '')
            result.update({
                'swarm_enabled': False,
                'session_id': session_id,
                'route_reason': 'fast_path: simple question → ConsultationAgent',
                'suggestions': result.get('suggestions', []),
            })
            if 'disclaimer' not in result:
                result['disclaimer'] = "⚠️ 以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。"

            end_time = datetime.now()
            result['total_time'] = (end_time - start_time).total_seconds()
            result['_cached_at'] = datetime.now()
            self._response_cache[cache_key] = result

            # 异步保存长期记忆（不阻塞响应）
            asyncio.create_task(self._save_long_term_memory(
                session_id, question, final_answer, "fast_path", 0,
                result['total_time']
            ))
            return result

        # ── 复杂/不明问题：完整流水线 ──
        # 并行检索记忆
        recent_history_task = asyncio.create_task(
            asyncio.to_thread(self.short_term_memory.get_recent_messages, session_id, 10)
        )
        similar_memories_task = asyncio.create_task(
            asyncio.to_thread(self.long_term_memory.search_similar_sessions, question, 3)
        )

        # 构建增强上下文
        enhanced_context = context or {}
        recent_history = await recent_history_task
        if recent_history:
            enhanced_context["recent_history"] = [
                {"role": msg.get("role", ""), "content": msg.get("content", "")}
                for msg in recent_history
            ]
            logger.info(f"Loaded {len(recent_history)} recent messages from short-term memory")

        similar_memories = await similar_memories_task
        if similar_memories:
            enhanced_context["historical_cases"] = [
                {"summary": mem["content"], "score": mem["score"]}
                for mem in similar_memories
            ]
            logger.info(f"Found {len(similar_memories)} similar historical cases from long-term memory")

        # Step 1: LeadAgent 分解任务
        assessment = await self.lead_agent.assess_and_decompose(question, enhanced_context)
        subtasks = assessment.get("subtasks", [])

        logger.info(f"LeadAgent 分解任务：{len(subtasks)} 个")

        # Step 2: 根据任务数量路由
        final_answer = None
        mode = None

        if len(subtasks) == 1:
            # 单任务 → 直接调用对应 Agent
            task = subtasks[0]
            agent_id = task.get("assigned_agent")
            agent = self._get_agent_by_id(agent_id)

            if agent is None:
                logger.warning(f"Unknown agent_id: {agent_id}, fallback to ConsultationAgent")
                agent = self.consultation_agent

            logger.info(f"Route: Single Agent ({agent_id})")
            mode = "single_agent"
            result = await agent.process({
                'question': question,
                'context': enhanced_context,
                'session_id': session_id
            })
            final_answer = result.get('answer', '')

            result.update({
                'swarm_enabled': False,
                'session_id': session_id,
                'route_reason': f'单任务路由到 {agent_id}'
            })
            if 'disclaimer' not in result:
                result['disclaimer'] = "⚠️ 以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。"
            if 'suggestions' not in result:
                result['suggestions'] = []

        elif len(subtasks) >= 2 and self.enable_swarm:
            logger.info(f"Route: Swarm (Multi-Agent Collaboration) - {len(subtasks)} tasks")
            mode = "swarm"
            result = await self._process_with_swarm(
                question=question,
                context=enhanced_context,
                assessment=assessment,
                session_id=session_id,
                start_time=start_time
            )
            return result  # Swarm 模式内已保存长期记忆

        else:
            if len(subtasks) == 0:
                logger.warning("No subtasks generated, fallback to ConsultationAgent")
                mode = "fallback"
            else:
                logger.info("Swarm disabled, fallback to ConsultationAgent")
                mode = "disabled_swarm"

            result = await self.consultation_agent.process({
                'question': question,
                'context': enhanced_context,
                'session_id': session_id
            })
            final_answer = result.get('answer', '')
            result.update({
                'swarm_enabled': False,
                'session_id': session_id
            })

        # 保存长期记忆 & 缓存
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        result['total_time'] = total_time
        result['_cached_at'] = datetime.now()
        self._response_cache[cache_key] = result

        asyncio.create_task(self._save_long_term_memory(
            session_id, question, final_answer, mode, len(subtasks), total_time
        ))

        return result

    async def _save_long_term_memory(
        self, session_id: str, question: str, answer: str,
        mode: str, subtasks_count: int, total_time: float
    ):
        """异步保存长期记忆（不阻塞主响应）"""
        try:
            self.long_term_memory.add_session_summary(
                session_id=session_id,
                question=question,
                answer=answer,
                metadata={
                    "mode": mode,
                    "subtasks_count": subtasks_count,
                    "total_time": total_time,
                }
            )
            logger.debug(f"Saved to long-term memory (session={session_id}, mode={mode})")
        except Exception as e:
            logger.error(f"Failed to save to long-term memory: {e}")

    async def _process_with_swarm(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        assessment: Dict[str, Any],
        session_id: str,
        start_time: datetime
    ) -> Dict[str, Any]:
        """
        使用 Swarm 处理复杂问题

        这是群体智能的核心流程

        注意：context 已经包含了长短期记忆（在 process() 中注入）
        """
        # context 已经包含 recent_history 和 historical_cases
        # 无需重复检索

        # 创建 SharedContext
        shared_context = SharedContext(session_id=session_id)

        # 附加 SharedContext 到所有 Worker
        for worker in self.worker_pool:
            worker.attach_shared_context(shared_context)

        # 发布 Swarm 启动事件
        shared_context.publish_event(Event(
            type=EventType.SWARM_STARTED,
            source_agent="swarm_coordinator",
            data={
                "question": question,
                "num_subtasks": len(assessment.get("subtasks", []))
            }
        ))

        # Step 1: LeadAgent 分解任务
        subtasks = self.lead_agent.create_subtasks(assessment, shared_context)
        logger.info(f"Created {len(subtasks)} subtasks")

        # Step 2: Worker 执行分配的任务（并行）
        tasks = []
        for worker in self.worker_pool:
            task = asyncio.create_task(
                self._worker_execute_assigned_tasks(worker, shared_context)
            )
            tasks.append(task)

        # 等待所有 Worker 完成（或超时）
        timeout_occurred = False
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=60.0  # 60 秒超时（从 90 秒减少）
            )
        except asyncio.TimeoutError:
            timeout_occurred = True
            logger.warning("Swarm execution timeout (60s)")
            # 记录哪些 Agent 已完成，哪些未完成
            completed_agents = list(shared_context.agent_contributions.keys())
            claimed_tasks = [
                (subtask.assigned_to, subtask.type)
                for subtask in shared_context.task_decomposition.values()
                if subtask.status.value == "claimed"
            ]
            logger.info(f"Completed agents: {completed_agents}")
            logger.info(f"Timed out tasks: {claimed_tasks}")

        # Step 3: LeadAgent 汇总结果
        # 即使超时，也尝试汇总已完成的部分结果
        final_answer = await self.lead_agent.synthesize_results(
            question=question,
            shared_context=shared_context,
            timeout_occurred=timeout_occurred
        )

        end_time = datetime.now()

        # Step 4: 生成 SessionSummary
        try:
            summary = SessionSummary.from_shared_context(
                session_id=session_id,
                question=question,
                shared_context=shared_context,
                final_answer=final_answer,
                start_time=start_time,
                end_time=end_time
            )
            self.session_manager.save_summary(summary)
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")

        # 注意：短期记忆已经在 Agent Loop 中保存了，这里不需要重复保存
        # Agent Loop 保存了完整的对话历史（user + assistant + tool messages）

        # 保存到 Mem0 长期记忆
        try:
            # 保存会话总结
            self.long_term_memory.add_session_summary(
                session_id=session_id,
                question=question,
                answer=final_answer,
                metadata={
                    "mode": "swarm",
                    "agents_count": len(shared_context.agent_contributions),
                    "total_time": (end_time - start_time).total_seconds(),
                    "timeout_occurred": timeout_occurred
                }
            )

            logger.info(f"Saved to Mem0 long-term memory (session={session_id})")

        except Exception as e:
            logger.error(f"Failed to save to Mem0: {e}")

        # 发布 Swarm 完成事件
        shared_context.publish_event(Event(
            type=EventType.SWARM_COMPLETED,
            source_agent="swarm_coordinator",
            data={
                "duration": (end_time - start_time).total_seconds(),
                "agents_count": len(shared_context.agent_contributions)
            }
        ))

        # 返回结果
        completed_agents = list(shared_context.agent_contributions.keys())
        result = {
            'answer': final_answer,
            'swarm_enabled': True,
            'session_id': session_id,
            'agents_involved': completed_agents,
            'subtasks_completed': len(shared_context.get_all_completed_subtasks()),
            'total_time': (end_time - start_time).total_seconds(),
            'swarm_metadata': shared_context.get_summary(),
            'timeout_occurred': timeout_occurred
        }

        # 提取建议和免责声明（简化实现）
        result['suggestions'] = self._extract_suggestions(final_answer)

        # 根据是否超时调整免责声明
        if timeout_occurred and not completed_agents:
            result['disclaimer'] = "由于系统超时，未能提供完整分析。建议简化问题重试，或在紧急情况下立即就医。"
        elif timeout_occurred:
            result['disclaimer'] = f"以上分析基于 {len(completed_agents)} 个 Agent 的部分协作结果（部分分析模块超时未完成），仅供参考，不能替代医生诊断。"
        else:
            result['disclaimer'] = "以上分析基于多个专业 Agent 的协作，仅供参考，不能替代医生诊断。"

        # 缓存 swarm 结果（5分钟有效）
        cache_key = f"{session_id}:{question.strip()}"
        result['_cached_at'] = end_time
        self._response_cache[cache_key] = result

        return result

    async def _worker_execute_assigned_tasks(
        self,
        worker: Any,
        shared_context: SharedContext
    ):
        """
        Worker 执行分配给它的任务

        简化后的流程：
        - 查找分配给自己的任务
        - 执行任务
        - 记录结果
        """
        try:
            # 获取分配给该 Agent 的任务
            assigned_tasks = shared_context.get_subtasks_for_agent(worker.agent_id)

            if not assigned_tasks:
                logger.debug(f"{worker.agent_id}: No assigned tasks")
                return

            # 并行执行所有分配的任务
            tasks = []
            for subtask in assigned_tasks:
                logger.info(f"{worker.agent_id}: Starting {subtask.type}")
                shared_context.start_subtask(subtask.id)

                task = asyncio.create_task(
                    self._execute_single_subtask(worker, subtask, shared_context)
                )
                tasks.append(task)

            # 等待所有任务完成
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"{worker.agent_id}: Error processing subtask: {e}")

    async def _execute_single_subtask(self, worker, subtask, shared_context):
        """执行单个子任务"""
        try:
            result = await worker.process_subtask(subtask)
            shared_context.complete_subtask(subtask.id, worker.agent_id, result)
            logger.info(f"{worker.agent_id}: Completed {subtask.type}")
        except Exception as e:
            logger.error(f"{worker.agent_id}: Error in {subtask.type}: {e}")

    def _extract_suggestions(self, final_answer: str) -> List[str]:
        """从最终答案中提取建议（简化实现）"""
        suggestions = []

        # 简单的文本匹配
        if "【核心建议】" in final_answer:
            # 提取核心建议部分
            start_idx = final_answer.find("【核心建议】")
            end_idx = final_answer.find("【", start_idx + 1)
            if end_idx == -1:
                end_idx = len(final_answer)

            suggestions_text = final_answer[start_idx:end_idx]

            # 提取编号列表
            import re
            matches = re.findall(r'\d+\.\s*([^\n]+)', suggestions_text)
            suggestions = matches[:5]  # 最多5条

        return suggestions or ["请遵循医嘱，注意休息和营养"]

async def process_with_swarm(
    question: str,
    context: Optional[Dict[str, Any]] = None,
    enable_swarm: bool = True,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    便捷函数：使用 Swarm 处理问题

    后端自动判断问题复杂度：
    - 简单问题（感冒、发烧等常见症状）→ 单 Agent 快速回复
    - 复杂问题（多症状组合、需鉴别诊断等）→ 多 Agent Swarm 协作

    Args:
        question: 用户问题
        context: 额外上下文
        enable_swarm: 是否允许 Swarm 协作（默认 True，后端自动判断何时真正启用）
        session_id: 会话ID（如果提供，将使用该ID而不是生成新的）

    Returns:
        处理结果
    """
    coordinator = SwarmCoordinator(enable_swarm=enable_swarm)
    return await coordinator.process(question, context, session_id=session_id)
