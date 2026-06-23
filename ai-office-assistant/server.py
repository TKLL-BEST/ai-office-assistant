#!/usr/bin/env python3
"""
AI Office Assistant - Backend Server
基于 OpenClaw 的个人精简版 AI 办公助手
后端 API + 技能调度服务
"""

import os
import sys
import json
import time
import uuid
import shutil
import subprocess
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── 配置 ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SKILLS_DIR = BASE_DIR / "skills"
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"

# DeepSeek API 配置
DEEPSEEK_API_KEY = "your_api_key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 确保数据目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "uploads").mkdir(exist_ok=True)
(DATA_DIR / "outputs").mkdir(exist_ok=True)
(DATA_DIR / "kb").mkdir(exist_ok=True)

app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def call_llm(prompt, system_prompt=None, use_feedback=True):
    """调用 DeepSeek 大模型

    参数：
        prompt: 用户输入
        system_prompt: 可选，自定义系统提示词
        use_feedback: 是否自动加载基于评分的个性化系统提示词
    """
    import urllib.request
    import urllib.error

    messages = []
    
    if system_prompt:
        # 如果显式传了 system_prompt，用它
        messages.append({"role": "system", "content": system_prompt})
    elif use_feedback:
        # 自动从反馈生成个性化 system prompt
        try:
            from skills.feedback_engine import FeedbackEngine
            engine = FeedbackEngine()
            fb_prompt = engine.generate_system_prompt()
            if fb_prompt:
                messages.append({"role": "system", "content": fb_prompt})
        except Exception:
            pass  # 如果没有反馈数据，就不加 system prompt

    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4096,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        return f"[LLM Error] {e}"


def run_skill_script(script_name, *args):
    """执行 skill 目录下的 Python 脚本"""
    script_path = SKILLS_DIR / script_name
    if not script_path.exists():
        return {"error": f"Skill script not found: {script_name}"}
    
    result = subprocess.run(
        [sys.executable, str(script_path)] + list(args),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


# ═══════════════════════════════════════════════════════════════════
# API 路由
# ═══════════════════════════════════════════════════════════════════

# ── 状态检查 ──────────────────────────────────────────────────────
@app.route("/api/status", methods=["GET"])
def api_status():
    """检查服务状态"""
    return jsonify({
        "status": "ok",
        "openclaw": "connected",
        "model": "deepseek-chat",
        "modules": ["doc-generator", "pdf-parser", "meeting-summary", "rag-qa"],
    })


# ── 模块1: AI文档生成助手 ─────────────────────────────────────────
@app.route("/api/doc-generator/templates", methods=["GET"])
def get_templates():
    """获取文档模板列表"""
    templates = {
        "weekly_report": {
            "name": "个人周报",
            "description": "生成个人周工作总结汇报",
        },
        "prd": {
            "name": "产品需求文档(精简版)",
            "description": "生成产品需求文档精简版",
        },
        "proposal": {
            "name": "方案建议书",
            "description": "生成项目/产品方案建议",
        },
        "essay": {
            "name": "短文/博客",
            "description": "生成短文、博客或文章",
        },
        "official_doc": {
            "name": "公文/通知",
            "description": "生成正式公文或通知",
        },
    }
    return jsonify({"templates": templates})


@app.route("/api/doc-generator/generate", methods=["POST"])
def generate_doc():
    """生成文档"""
    data = request.json
    template = data.get("template", "essay")
    requirements = data.get("requirements", "")
    extra_instructions = data.get("extra_instructions", "")

    if not requirements:
        return jsonify({"error": "请填写需求描述"}), 400

    # 模板提示词
    template_prompts = {
        "weekly_report": """你正在写一份【个人周工作总结汇报】。
请使用清晰的分段结构：本周工作内容、关键成果、遇到的问题及解决、下周计划。
语言正式、简洁，适合向上级汇报。""",
        "prd": """你正在写一份【产品需求文档(精简版)】。
结构包含：背景与目标、用户场景、功能需求列表、非功能需求、验收标准。
专业、简洁，适合产品设计场景。""",
        "proposal": """你正在写一份【方案建议书】。
结构包含：项目背景、方案概述、实施计划、预期效果、所需资源。
逻辑清晰、说服力强。""",
        "essay": """你正在写一篇【短文/博客/文章】。
结构包含：引言、正文(2-3个要点)、结语。
语言流畅、引人入胜。""",
        "official_doc": """你正在写一份【正式公文/通知】。
使用正式公文格式：标题、正文(原因、内容、要求)、落款。
语言正式、准确、规范。""",
    }

    prompt = template_prompts.get(template, template_prompts["essay"])
    full_prompt = f"{prompt}\n\n需求说明：{requirements}"
    if extra_instructions:
        full_prompt += f"\n\n补充要求：{extra_instructions}"
    full_prompt += "\n\n请直接输出文档内容，不要加额外说明。"

    content = call_llm(full_prompt)
    
    # 保存到文件
    doc_id = str(uuid.uuid4())[:8]
    output_path = DATA_DIR / "outputs" / f"doc_{doc_id}.txt"
    output_path.write_text(content, encoding="utf-8")

    return jsonify({
        "doc_id": doc_id,
        "content": content,
        "filename": f"doc_{doc_id}.txt",
    })


@app.route("/api/doc-generator/optimize", methods=["POST"])
def optimize_doc():
    """优化文档（润色/改写/缩写）"""
    data = request.json
    content = data.get("content", "")
    action = data.get("action", "polish")  # polish | rewrite | shorten

    if not content:
        return jsonify({"error": "请提供文档内容"}), 400

    prompts = {
        "polish": "请润色以下文本，优化表达但不改变原意：\n\n",
        "rewrite": "请改写以下文本，用不同方式表达相同内容：\n\n",
        "shorten": "请缩写以下文本，保留核心信息，将长度缩短一半：\n\n",
    }

    full_prompt = prompts.get(action, prompts["polish"]) + content
    result = call_llm(full_prompt)

    return jsonify({
        "content": result,
        "action": action,
    })


# ── 模块2: 文档智能解析助手 ───────────────────────────────────────
@app.route("/api/pdf-parser/upload", methods=["POST"])
def upload_and_parse():
    """上传并解析文档"""
    if "file" not in request.files:
        return jsonify({"error": "请上传文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    # 保存上传文件
    ext = os.path.splitext(file.filename)[1].lower()
    upload_id = str(uuid.uuid4())[:8]
    save_path = DATA_DIR / "uploads" / f"{upload_id}{ext}"
    file.save(save_path)

    # 调用解析脚本
    from skills.parse_document import parse_document
    result = parse_document(str(save_path))

    return jsonify({
        "upload_id": upload_id,
        "filename": file.filename,
        "text": result.get("text", ""),
        "tables": result.get("tables", []),
        "structure": result.get("structure", []),
    })


@app.route("/api/pdf-parser/extract", methods=["POST"])
def extract_key_info():
    """提取关键信息"""
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "请提供文档文本"}), 400

    prompt = f"""从以下文档文本中提取关键信息，以 JSON 格式输出：

文档文本：
{text[:8000]}

请提取：
1. 核心主题/标题
2. 关键要点（列表）
3. 重要数据/数字
4. 结论或建议

输出格式：
{{
  "title": "核心主题",
  "key_points": ["要点1", "要点2", ...],
  "key_data": [{{"item": "描述", "value": "数值"}}, ...],
  "conclusion": "结论或建议"
}}
"""
    result = call_llm(prompt)
    
    # 尝试解析 JSON
    try:
        import re
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            extracted = json.loads(json_match.group())
        else:
            extracted = {"raw": result}
    except:
        extracted = {"raw": result}

    return jsonify({
        "extracted": extracted,
    })


# ── 模块3: 智能会议助手 ─────────────────────────────────────────
@app.route("/api/meeting-summary/upload", methods=["POST"])
def upload_audio():
    """上传音频文件"""
    if "file" not in request.files:
        return jsonify({"error": "请上传音频文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".mp3", ".wav", ".m4a", ".ogg"):
        return jsonify({"error": "不支持的音频格式，支持 MP3/WAV/M4A/OGG"}), 400

    upload_id = str(uuid.uuid4())[:8]
    save_path = DATA_DIR / "uploads" / f"{upload_id}{ext}"
    file.save(save_path)

    return jsonify({
        "upload_id": upload_id,
        "filename": file.filename,
        "size": os.path.getsize(save_path),
        "message": "音频上传成功，点击生成纪要开始处理",
    })


@app.route("/api/meeting-summary/generate", methods=["POST"])
def generate_meeting_summary():
    """
    生成会议纪要
    注意：语音转写需要额外配置 ASR API（百度/阿里云/讯飞）
    这里使用模拟转写 + 大模型纪要生成
    """
    data = request.json
    upload_id = data.get("upload_id", "")
    
    # 这里简化处理：如果用户提供文本则直接处理，否则模拟
    manual_text = data.get("text", "")

    if manual_text:
        transcript = manual_text
    else:
        # 模拟转写文本（演示用）
        transcript = """[00:00] 张三：大家好，今天我们讨论一下新项目的开发计划。
[00:05] 李四：我觉得我们可以在两周内完成第一阶段。
[00:10] 张三：好的，那我们先确定一下第一阶段的里程碑。
[00:15] 王五：我建议把需求评审放在下周一下午。
[00:20] 李四：没问题，我会准备好原型设计。
[00:25] 张三：还有测试计划，我们要提前安排好。
[00:30] 王五：测试计划我来负责，周三前出一版。
[00:35] 张三：好的，那本周目标就是完成需求评审和原型。
[00:40] 李四：大家还有别的事吗？
[00:45] 张三：下周产品上线，各部分进展情况我们要跟进。"""

    # 用大模型生成纪要
    prompt = f"""请从以下会议转写文本中提取会议纪要，包括：
1. 核心议题
2. 关键结论
3. 待办事项（带负责人）

转写文本：
{transcript}

请按以下格式输出：
📋 会议纪要
────────
**核心议题：**
...

**关键结论：**
...

**待办事项：**
- 事项1 (负责人：XXX)
- 事项2 (负责人：XXX)"""

    summary = call_llm(prompt)

    return jsonify({
        "transcript": transcript,
        "summary": summary,
    })


@app.route("/api/meeting-summary/export", methods=["POST"])
def export_meeting():
    """导出会议纪要"""
    data = request.json
    transcript = data.get("transcript", "")
    summary = data.get("summary", "")

    content = f"""会议转写文本
{'='*30}

{transcript}

{'='*30}
AI 生成的会议纪要
{'='*30}

{summary}
"""

    export_id = str(uuid.uuid4())[:8]
    output_path = DATA_DIR / "outputs" / f"meeting_{export_id}.txt"
    output_path.write_text(content, encoding="utf-8")

    return jsonify({
        "export_id": export_id,
        "filename": f"meeting_{export_id}.txt",
        "content": content,
    })


# ── 模块4: 私有知识库 RAG（向量检索版） ───────────────────────────

@app.route("/api/rag-qa/upload", methods=["POST"])
def upload_kb_doc():
    """上传文档到知识库（向量检索版）"""
    if "file" not in request.files:
        return jsonify({"error": "请上传文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".txt", ".docx", ".md"):
        return jsonify({"error": "不支持的格式，支持 PDF/TXT/DOCX/MD"}), 400

    # 保存上传文件
    save_path = DATA_DIR / "uploads" / file.filename
    file.save(save_path)

    try:
        from skills.rag_engine import RagEngine
        engine = RagEngine()
        result = engine.add_document(str(save_path))
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"入库失败: {e}"}), 500


@app.route("/api/rag-qa/list", methods=["GET"])
def list_kb_docs():
    """列出知识库中的文档"""
    try:
        from skills.rag_engine import RagEngine
        engine = RagEngine()
        docs = engine.list_documents()
        return jsonify({"documents": docs})
    except Exception as e:
        return jsonify({"documents": []})


@app.route("/api/rag-qa/delete", methods=["POST"])
def delete_kb_doc():
    """删除知识库中的文档"""
    data = request.json
    doc_id = data.get("doc_id", "")

    try:
        from skills.rag_engine import RagEngine
        engine = RagEngine()
        result = engine.delete_document(doc_id)
        if "error" in result:
            return jsonify(result), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"删除失败: {e}"}), 500


@app.route("/api/rag-qa/ask", methods=["POST"])
def ask_kb():
    """向知识库提问（RAG 向量检索）"""
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "请输入问题"}), 400

    try:
        from skills.rag_engine import RagEngine
        engine = RagEngine()
        result = engine.query(question)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"查询失败: {e}"}), 500


@app.route("/api/rag-qa/clear", methods=["POST"])
def clear_kb():
    """清空知识库"""
    try:
        from skills.rag_engine import RagEngine
        engine = RagEngine()
        result = engine.clear()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"清空失败: {e}"}), 500



@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    data = request.json
    prompt = data.get("prompt", "")
    response = data.get("response", "")
    rating = data.get("rating", 0)
    module = data.get("module", "unknown")
    feedback_text = data.get("feedback_text", "")
    tags = data.get("tags", [])
    if not prompt or not response:
        return jsonify({"error": "prompt and response required"}), 400
    if not 1 <= rating <= 5:
        return jsonify({"error": "rating must be 1-5"}), 400
    try:
        from skills.feedback_engine import FeedbackEngine
        engine = FeedbackEngine()
        result = engine.add_feedback(
            prompt=prompt, response=response, rating=rating,
            module=module, feedback_text=feedback_text, tags=tags,
            metadata={"model": "deepseek-chat"}
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/feedback/list", methods=["GET"])
def list_feedback():
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    module = request.args.get("module", None)
    try:
        from skills.feedback_engine import FeedbackEngine
        engine = FeedbackEngine()
        feedbacks = engine.list_feedbacks(limit=limit, offset=offset, module=module)
        stats = engine.get_stats()
        dpo_stats = engine.get_dpo_stats()
        return jsonify({"feedbacks": feedbacks, "stats": stats, "dpo_stats": dpo_stats})
    except Exception as e:
        return jsonify({"feedbacks": [], "error": str(e)})

@app.route("/api/feedback/stats", methods=["GET"])
def feedback_stats():
    try:
        from skills.feedback_engine import FeedbackEngine
        engine = FeedbackEngine()
        stats = engine.get_stats()
        dpo_stats = engine.get_dpo_stats()
        pref = engine.get_preference_summary()
        return jsonify({"stats": stats, "dpo_stats": dpo_stats, "preference": pref})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/feedback/dpo/export", methods=["GET"])
def export_dpo_data():
    min_rating_diff = request.args.get("min_rating_diff", 2, type=int)
    try:
        from skills.feedback_engine import FeedbackEngine
        engine = FeedbackEngine()
        dpo_pairs = engine.export_dpo(min_rating_diff=min_rating_diff)
        import time
        filename = "dpo_dataset_" + time.strftime("%Y%m%d_%H%M%S") + ".jsonl"
        filepath = engine.save_dpo_export(dpo_pairs, filename)
        return jsonify({
            "dpo_pairs": len(dpo_pairs), "filepath": str(filepath),
            "filename": filename, "format": "JSONL",
            "usage": "For DPO training, see rlhf_learning/dpo_demo.py"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# RLHF 模拟器路由 (用 DeepSeek API 模拟完整 RLHF 流程)
# ============================================================

# 全局单例
_rlhf_sim = None

def get_rlhf_sim():
    global _rlhf_sim
    if _rlhf_sim is None:
        from skills.rlhf_simulator import RLHF_Simulator
        _rlhf_sim = RLHF_Simulator()
    return _rlhf_sim


@app.route("/api/rlhf/sim/status", methods=["GET"])
def rlhf_sim_status():
    """获取 RLHF 模拟器状态"""
    try:
        sim = get_rlhf_sim()
        status = sim.get_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rlhf/sim/train", methods=["POST"])
def rlhf_sim_train():
    """
    执行一步 RLHF 训练（用 DeepSeek API 模拟 PPO 流程）
    
    请求体：
    {
        "prompt": "用户输入",  # 必填
        "beta": 0.1           # 可选，KL 惩罚系数
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "缺少 prompt 参数"}), 400
        beta = data.get("beta", 0.1)
        
        sim = get_rlhf_sim()
        result = sim.train_step(prompt, beta=beta)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rlhf/sim/reward", methods=["POST"])
def rlhf_sim_reward():
    """
    用 DeepSeek API 模拟 Reward Model 评分
    
    请求体：
    {
        "prompt": "用户问题",
        "response": "AI 回答"
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt", "")
        response = data.get("response", "")
        if not prompt or not response:
            return jsonify({"error": "缺少 prompt 或 response"}), 400
        
        sim = get_rlhf_sim()
        scores = sim.get_reward_model_response(prompt, response)
        return jsonify(scores)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rlhf/sim/compare", methods=["POST"])
def rlhf_sim_compare():
    """
    DPO 风格偏好对比
    
    请求体：
    {
        "prompt": "用户问题",
        "response_a": "回答A",
        "response_b": "回答B"
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt", "")
        a = data.get("response_a", "")
        b = data.get("response_b", "")
        if not prompt or not a or not b:
            return jsonify({"error": "缺少 prompt/response_a/response_b"}), 400
        
        sim = get_rlhf_sim()
        result = sim.reward_model.compare(prompt, a, b)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 旧版 RLHF 路由（偏好学习）
# ============================================================

@app.route("/api/rlhf/status", methods=["GET"])
def rlhf_status():
    try:
        from skills.rlhf_engine import RLHF_Engine
        engine = RLHF_Engine()
        info = engine.get_current_version_info()
        return jsonify({
            "status": "ok",
            "current_version": info["version_id"],
            "score": info["score"],
            "created_at": info["created_at"],
            "total_versions": info["total_versions"],
            "styles": info["styles"],
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/feedback/preference", methods=["GET"])
def get_preference():
    """获取基于评分的偏好分析结果"""
    try:
        from skills.feedback_engine import FeedbackEngine
        engine = FeedbackEngine()
        pref = engine.get_preference_summary()
        return jsonify(pref)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rlhf/train", methods=["POST"])
def rlhf_train():
    """
    训练入口：根据评分优化系统提示词 + 更新偏好分析
    """
    try:
        from skills.feedback_engine import FeedbackEngine
        engine = FeedbackEngine()
        pref = engine.get_preference_summary()
        system_prompt = engine.generate_system_prompt()
        
        return jsonify({
            "success": True,
            "message": "基于评分的系统提示词已更新",
            "signal_strength": pref["signal_strength"],
            "total_feedbacks": pref["total"],
            "system_prompt": system_prompt or "暂无足够反馈数据生成个性化提示词",
            "preferences": pref["preferences"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/rlhf/evaluate", methods=["GET"])
def rlhf_evaluate():
    try:
        from skills.rlhf_engine import RLHF_Engine
        engine = RLHF_Engine()
        report = engine.evaluate()
        return jsonify(report)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/rlhf/history", methods=["GET"])
def rlhf_history():
    try:
        from skills.rlhf_engine import RLHF_Engine
        engine = RLHF_Engine()
        history = engine.history(limit=20)
        return jsonify({
            "versions": history,
            "total": len(history),
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/rlhf/rollback", methods=["POST"])
def rlhf_rollback():
    try:
        data = request.get_json(silent=True) or {}
        version_id = data.get("version_id", None)

        from skills.rlhf_engine import RLHF_Engine
        engine = RLHF_Engine()
        result = engine.rollback(version_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
def export_file(filename):
    """下载导出文件"""
    file_path = DATA_DIR / "outputs" / filename
    if not file_path.exists():
        return jsonify({"error": "文件不存在"}), 404
    return send_file(file_path, as_attachment=True)


# ── 前端静态文件 ──────────────────────────────────────────────────
@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/<path:path>")
def static_files(path):
    return app.send_static_file(path)


# ═══════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Office Assistant Server")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()
    
    print(f"🚀 AI Office Assistant Server running on http://{args.host}:{args.port}")
    print(f"📧 Backend API: http://127.0.0.1:{args.port}/api/")
    print(f"🖥 Frontend: http://127.0.0.1:{args.port}/")
    
    app.static_folder = str(FRONTEND_DIR)
    app.static_url_path = ""
    app.run(host=args.host, port=args.port, debug=True)
