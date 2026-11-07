#!/usr/bin/env python3
"""
多智能体医疗助手 - Web 界面
基于 Streamlit 的单页应用

运行方式:
    streamlit run app.py

功能:
    - 对话区：用户输入健康问题，后端调用 Swarm 多智能体协作系统
    - 历史记录：展示当前会话的完整对话历史
    - 侧边栏：显示 Agent 状态、会话信息、操作按钮
    - 进度提示：处理过程中显示状态信息
"""

import asyncio
import concurrent.futures
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 添加项目根目录到路径，确保可以导入 swarm 和 agents 模块
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import streamlit as st
from loguru import logger

from swarm import SwarmCoordinator, process_with_swarm
from swarm.shared_context import SharedContext


# ============================================================================
# 页面配置
# ============================================================================
st.set_page_config(
    page_title="多智能体医疗助手",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "多智能体医疗助手 v0.1.0\n\n"
            "基于 Swarm 架构的智能医疗咨询系统，"
            "集成健康咨询、诊断推理、循证医学三大专业 Agent。"
        )
    },
)


# ============================================================================
# 自定义 CSS
# ============================================================================
def inject_custom_css() -> None:
    """注入自定义 CSS 样式，美化界面"""
    st.markdown(
        """
    <style>
    /* 顶部标题横幅 */
    .main-header {
        text-align: center;
        padding: 1.2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        color: white;
        font-size: 1.8rem;
        margin: 0;
        letter-spacing: 2px;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* 执行时间标记 */
    .time-badge {
        display: inline-block;
        background: #e9ecef;
        color: #495057;
        padding: 1px 8px;
        border-radius: 8px;
        font-size: 0.75rem;
        margin-left: 6px;
    }

    /* 免责声明样式 */
    .disclaimer-box {
        background-color: #fff8e1;
        border-left: 4px solid #ffa000;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 10px;
        font-size: 0.82rem;
        color: #6d4c00;
    }

    /* Agent 卡片 */
    .agent-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 10px 12px;
        margin: 6px 0;
        border-left: 4px solid #667eea;
        transition: all 0.2s;
    }
    .agent-card:hover {
        background: #eef0ff;
        border-left-color: #764ba2;
    }
    .agent-card h4 {
        margin: 0;
        font-size: 0.9rem;
    }
    .agent-card p {
        margin: 3px 0 0 0;
        font-size: 0.78rem;
        color: #6c757d;
        line-height: 1.4;
    }

    /* 欢迎消息 */
    .welcome-box {
        text-align: center;
        padding: 2rem;
        background: #f0f4ff;
        border-radius: 12px;
        margin: 1rem 0;
    }
    .welcome-box h3 {
        color: #495057;
        margin-bottom: 0.5rem;
    }
    .welcome-box p {
        color: #6c757d;
        font-size: 0.9rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# ============================================================================
# 会话状态初始化
# ============================================================================
def init_session_state() -> None:
    """初始化 Streamlit 会话状态变量"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"
        )

    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0

    if "coordinator" not in st.session_state:
        # enable_swarm 始终为 True，后端根据问题复杂度自动判断单/多 Agent
        st.session_state.coordinator = SwarmCoordinator(enable_swarm=True)


# ============================================================================
# 异步 → 同步适配
# ============================================================================
def run_async(coro):
    """
    安全地运行异步协程，兼容已有 / 没有事件循环的环境。
    Streamlit 环境没有运行中的事件循环，直接使用 asyncio.run()。

    Args:
        coro: 一个 awaitable 协程对象

    Returns:
        协程的返回值
    """
    try:
        # 检查当前线程是否有运行中的事件循环
        asyncio.get_running_loop()
        # 有运行中的循环（极少见）→ 在独立线程中执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=90)
    except RuntimeError:
        # 没有运行中的事件循环 → 直接使用 asyncio.run()（Streamlit 的常见路径）
        return asyncio.run(coro)


def ask_agent(
    user_input: str,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    同步封装：调用 SwarmCoordinator 处理用户健康问题。
    后端自动判断问题复杂度：简单问题用单 Agent，复杂问题用多 Agent 协作。

    Args:
        user_input: 用户输入的健康问题
        session_id: 会话 ID，用于记忆连续性

    Returns:
        包含 answer, swarm_enabled, agents_involved, suggestions, disclaimer 等字段的字典
    """
    coordinator: SwarmCoordinator = st.session_state.coordinator

    logger.info(f"Processing query: {user_input[:60]}...")

    result = run_async(
        coordinator.process(user_input, session_id=session_id)
    )

    # 确保返回结果包含必要字段
    result.setdefault("answer", "抱歉，未能生成回复。")
    result.setdefault("swarm_enabled", False)
    result.setdefault("agents_involved", [])
    result.setdefault("suggestions", [])
    result.setdefault("disclaimer",
        "⚠️ 以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。")
    result.setdefault("total_time", 0.0)
    result.setdefault("timeout_occurred", False)

    return result


# ============================================================================
# 侧边栏
# ============================================================================
AGENT_DISPLAY_NAME = {
    "consultation_agent": "💬 健康咨询 Agent",
    "diagnostic_agent": "🔍 诊断推理 Agent",
    "research_agent": "📚 循证医学 Agent",
    "lead_agent": "🎯 协调 Agent",
}

AGENT_DESCRIPTIONS = {
    "consultation_agent": "常见疾病科普、健康建议、生活方式指导",
    "diagnostic_agent": "症状模式分析、鉴别诊断、临床推理",
    "research_agent": "临床指南检索、文献综合、循证医学证据",
    "lead_agent": "任务分解、结果汇总、多视角整合",
}

AGENT_CAPABILITIES = {
    "consultation_agent": ["健康咨询", "风险评估", "症状分诊"],
    "diagnostic_agent": ["症状分析", "鉴别诊断", "临床推理", "多系统分析"],
    "research_agent": ["文献检索", "证据综合", "事实核查", "指南查询"],
    "lead_agent": ["任务编排", "结果合成"],
}


def render_sidebar() -> None:
    """渲染侧边栏：会话信息、Agent 状态、操作按钮"""
    with st.sidebar:
        st.title("🏥 控制台")
        st.markdown("---")

        # ── 会话信息 ──
        st.subheader("📋 会话信息")
        sid = st.session_state.session_id
        st.caption(f"会话 ID: `{sid[:20]}...`")
        st.metric("对话轮次", st.session_state.conversation_count)

        st.markdown("---")

        # ── Agent 状态面板 ──
        st.subheader("🤖 智能体列表")

        for agent_id in [
            "consultation_agent",
            "diagnostic_agent",
            "research_agent",
            "lead_agent",
        ]:
            name = AGENT_DISPLAY_NAME.get(agent_id, agent_id)
            desc = AGENT_DESCRIPTIONS.get(agent_id, "")
            caps = AGENT_CAPABILITIES.get(agent_id, [])
            caps_str = " · ".join(caps[:3])

            st.markdown(
                f"""
            <div class="agent-card">
                <h4>{name}</h4>
                <p>{desc}</p>
                <small>🏷️ {caps_str}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── 操作按钮 ──
        st.subheader("🔧 操作")
        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("🔄 新会话", use_container_width=True):
                new_id = (
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
                    f"{str(uuid.uuid4())[:8]}"
                )
                st.session_state.messages = []
                st.session_state.session_id = new_id
                st.session_state.conversation_count = 0
                # 重建 coordinator 以清理短期记忆
                st.session_state.coordinator = SwarmCoordinator(enable_swarm=True)
                st.rerun()

        with col_b:
            if st.button("🗑️ 清屏", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        st.markdown("---")

        # ── 使用提示 ──
        with st.expander("💡 使用提示"):
            st.markdown(
                """
            | 问题类型 | 示例 | 系统行为 |
            |---|---|---|
            | 简单咨询 | "感冒怎么办" | 自动单 Agent 快速回复 |
            | 复杂症状 | "头痛+恶心+视力模糊" | 自动多 Agent 协作 |
            | 循证查询 | "高血压最新诊疗指南" | Research Agent 检索 |
            | 风险评估 | "胸闷气短冒冷汗严重吗" | Diagnostic Agent 评估 |

            > 系统会根据问题复杂度**自动判断**使用单 Agent 还是多 Agent 协作，无需手动切换。
            """
            )

        # ── 页脚 ──
        st.markdown("---")
        st.caption(
            "⚠️ 本系统仅供健康科普参考，"
            "不能替代专业医生的诊断和治疗。"
            "如有身体不适，请及时就医。"
        )


# ============================================================================
# 对话区
# ============================================================================
def render_welcome() -> None:
    """当还没有对话历史时，显示欢迎信息"""
    if len(st.session_state.messages) > 0:
        return

    st.markdown(
        """
    <div class="welcome-box">
        <h3>👋 您好！我是智能体医疗助手</h3>
        <p>由 <b>健康咨询</b>、<b>诊断推理</b>、<b>循证医学</b> 三大专业 Agent 协同驱动</p>
        <p style="margin-top:0.8rem;font-size:0.85rem;">
            您可以直接描述您的健康问题或症状，系统会自动判断是否需要多 Agent 协作分析。
        </p>
        <p style="font-size:0.8rem;color:#999;">
            示例：<i>"我感冒了怎么办"</i> ·
            <i>"头痛一周越来越严重需要就医吗"</i> ·
            <i>"高血压最新治疗方案是什么"</i>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    """渲染对话历史中的全部消息"""
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            # 正文
            st.markdown(msg["content"])

            # 元数据行（仅在 assistant 消息上显示）
            if role == "assistant" and msg.get("metadata"):
                _render_message_metadata(msg)

            # 建议（可折叠）
            if msg.get("suggestions"):
                with st.expander("💡 核心建议"):
                    for i, suggestion in enumerate(msg["suggestions"], 1):
                        st.markdown(f"{i}. {suggestion}")

            # 免责声明
            if msg.get("disclaimer"):
                st.markdown(
                    f'<div class="disclaimer-box">{msg["disclaimer"]}</div>',
                    unsafe_allow_html=True,
                )


def _render_message_metadata(msg: Dict[str, Any]) -> None:
    """渲染一条 assistant 消息的元数据标记"""
    meta = msg["metadata"]
    parts: List[str] = []

    # 模式标记
    if meta.get("swarm_enabled"):
        agents = meta.get("agents_involved", [])
        agent_names = [
            AGENT_DISPLAY_NAME.get(a, a).split(" ")[0] for a in agents
        ]
        parts.append(f"🐝 **Swarm 协作** · {', '.join(agent_names)} 参与")
    else:
        parts.append("🤖 **单 Agent 模式**")

    # 耗时
    elapsed = meta.get("total_time", 0)
    if elapsed:
        parts.append(f"⏱️ {elapsed:.1f}s")

    # 超时告警
    if meta.get("timeout_occurred"):
        parts.append("⚠️ 部分超时")

    st.caption(" · ".join(parts))


# ============================================================================
# 用户输入处理
# ============================================================================
def handle_user_input(prompt: str) -> None:
    """
    处理用户输入：追加到对话历史，调用后端，追加回复。

    Args:
        prompt: 用户在聊天框输入的文本
    """
    # 1. 记录用户消息
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    # 2. 显示用户消息（当前 run 内可见）
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. 调用后端
    st.session_state.conversation_count += 1

    with st.chat_message("assistant"):
        with st.spinner("🤔 正在分析您的问题，Agent 工作中..."):
            try:
                result = ask_agent(
                    prompt,
                    session_id=st.session_state.session_id,
                )
            except Exception as exc:
                logger.error(f"处理请求出错: {exc}")
                result = {
                    "answer": (
                        f"❌ 抱歉，处理您的问题时出现内部错误。\n\n"
                        f"错误详情：`{exc}`\n\n"
                        f"请简化问题后重试，或点击侧边栏「新会话」按钮。"
                    ),
                    "swarm_enabled": False,
                    "agents_involved": [],
                    "suggestions": ["请简化问题重试", "如持续出错请联系管理员"],
                    "disclaimer": "系统出现技术故障，分析结果可能不完整。",
                    "total_time": 0.0,
                    "timeout_occurred": False,
                }

        # 显示回答正文
        answer = result.get("answer", "抱歉，未能生成回复。")
        st.markdown(answer)

        # 元数据
        meta = {
            "swarm_enabled": result.get("swarm_enabled", False),
            "agents_involved": result.get("agents_involved", []),
            "total_time": result.get("total_time", 0.0),
            "timeout_occurred": result.get("timeout_occurred", False),
        }
        _render_message_metadata({"metadata": meta})

        # 建议
        suggestions: List[str] = result.get("suggestions", [])
        if suggestions:
            with st.expander("💡 核心建议"):
                for i, s in enumerate(suggestions, 1):
                    st.markdown(f"{i}. {s}")

        # 免责声明
        disclaimer: str = result.get("disclaimer", "")
        if disclaimer:
            st.markdown(
                f'<div class="disclaimer-box">{disclaimer}</div>',
                unsafe_allow_html=True,
            )

    # 4. 持久化助手消息
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "metadata": meta,
            "suggestions": suggestions,
            "disclaimer": disclaimer,
        }
    )


# ============================================================================
# 主入口
# ============================================================================
def main() -> None:
    """Streamlit 应用主入口"""
    inject_custom_css()
    init_session_state()

    # ── 顶部标题 ──
    st.markdown(
        """
    <div class="main-header">
        <h1>🏥 医疗助手Agent</h1>
        <p>健康咨询 · 诊断推理 · 循证医学 — 三大 Agent 群体协作</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── 侧边栏 ──
    render_sidebar()

    # ── 对话区 ──
    render_welcome()
    render_chat_history()

    # ── 输入框（固定底部）──
    if prompt := st.chat_input(
        "💬 请输入您的健康问题...",
        key="chat_input",
    ):
        handle_user_input(prompt)


if __name__ == "__main__":
    main()
