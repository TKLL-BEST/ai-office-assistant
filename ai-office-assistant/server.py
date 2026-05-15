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
DEEPSEEK_API_KEY = "your_API"
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

def call_llm(prompt, system_prompt=None):
    """调用 DeepSeek 大模型"""
    import urllib.request
    import urllib.error

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
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


# ── 导出 ──────────────────────────────────────────────────────────
@app.route("/api/export/<filename>", methods=["GET"])
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
