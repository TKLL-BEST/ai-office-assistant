with open('server.py', encoding='utf-8') as f:
    lines = f.readlines()

routes = []
routes.append('\n')
routes.append('# ---- RLHF Feedback System (Module 8) ----\n')
routes.append('\n')
routes.append('@app.route("/api/feedback", methods=["POST"])\n')
routes.append('def submit_feedback():\n')
routes.append('    data = request.json\n')
routes.append('    prompt = data.get("prompt", "")\n')
routes.append('    response = data.get("response", "")\n')
routes.append('    rating = data.get("rating", 0)\n')
routes.append('    module = data.get("module", "unknown")\n')
routes.append('    feedback_text = data.get("feedback_text", "")\n')
routes.append('    tags = data.get("tags", [])\n')
routes.append('    if not prompt or not response:\n')
routes.append('        return jsonify({"error": "prompt and response required"}), 400\n')
routes.append('    if not 1 <= rating <= 5:\n')
routes.append('        return jsonify({"error": "rating must be 1-5"}), 400\n')
routes.append('    try:\n')
routes.append('        from skills.feedback_engine import FeedbackEngine\n')
routes.append('        engine = FeedbackEngine()\n')
routes.append('        result = engine.add_feedback(\n')
routes.append('            prompt=prompt, response=response, rating=rating,\n')
routes.append('            module=module, feedback_text=feedback_text, tags=tags,\n')
routes.append('            metadata={"model": "deepseek-chat"}\n')
routes.append('        )\n')
routes.append('        return jsonify(result)\n')
routes.append('    except Exception as e:\n')
routes.append('        return jsonify({"error": str(e)}), 500\n')
routes.append('\n')
routes.append('@app.route("/api/feedback/list", methods=["GET"])\n')
routes.append('def list_feedback():\n')
routes.append('    limit = request.args.get("limit", 50, type=int)\n')
routes.append('    offset = request.args.get("offset", 0, type=int)\n')
routes.append('    module = request.args.get("module", None)\n')
routes.append('    try:\n')
routes.append('        from skills.feedback_engine import FeedbackEngine\n')
routes.append('        engine = FeedbackEngine()\n')
routes.append('        feedbacks = engine.list_feedbacks(limit=limit, offset=offset, module=module)\n')
routes.append('        stats = engine.get_stats()\n')
routes.append('        dpo_stats = engine.get_dpo_stats()\n')
routes.append('        return jsonify({"feedbacks": feedbacks, "stats": stats, "dpo_stats": dpo_stats})\n')
routes.append('    except Exception as e:\n')
routes.append('        return jsonify({"feedbacks": [], "error": str(e)})\n')
routes.append('\n')
routes.append('@app.route("/api/feedback/stats", methods=["GET"])\n')
routes.append('def feedback_stats():\n')
routes.append('    try:\n')
routes.append('        from skills.feedback_engine import FeedbackEngine\n')
routes.append('        engine = FeedbackEngine()\n')
routes.append('        stats = engine.get_stats()\n')
routes.append('        dpo_stats = engine.get_dpo_stats()\n')
routes.append('        return jsonify({"stats": stats, "dpo_stats": dpo_stats})\n')
routes.append('    except Exception as e:\n')
routes.append('        return jsonify({"error": str(e)}), 500\n')
routes.append('\n')
routes.append('@app.route("/api/feedback/dpo/export", methods=["GET"])\n')
routes.append('def export_dpo_data():\n')
routes.append('    min_rating_diff = request.args.get("min_rating_diff", 2, type=int)\n')
routes.append('    try:\n')
routes.append('        from skills.feedback_engine import FeedbackEngine\n')
routes.append('        engine = FeedbackEngine()\n')
routes.append('        dpo_pairs = engine.export_dpo(min_rating_diff=min_rating_diff)\n')
routes.append('        filename = f"dpo_dataset_{time.strftime(\'%Y%m%d_%H%M%S\')}.jsonl"\n')
routes.append('        filepath = engine.save_dpo_export(dpo_pairs, filename)\n')
routes.append('        return jsonify({\n')
routes.append('            "dpo_pairs": len(dpo_pairs), "filepath": str(filepath),\n')
routes.append('            "filename": filename,\n')
routes.append('            "format": "JSONL (prompt / chosen / rejected)",\n')
routes.append('            "usage": "For DPO training, see rlhf_learning/dpo_demo.py"\n')
routes.append('        })\n')
routes.append('    except Exception as e:\n')
routes.append('        return jsonify({"error": str(e)}), 500\n')
routes.append('\n')

for i in range(len(lines)):
    if '/api/export/<filename>' in lines[i]:
        insert_at = i
        break
for r in reversed(routes):
    lines.insert(insert_at, r)

with open('server.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('Routes inserted at line', insert_at)

# === RLHF Training 路由说明 ===
# 已直接集成到 server.py 中，无需运行本脚本
#
# 新增路由:
#   /api/rlhf/status    GET    查看模型版本状态
#   /api/rlhf/train     POST   触发 DPO 训练
#   /api/rlhf/evaluate  GET    评估当前模型
#   /api/rlhf/history   GET    查看版本历史
#   /api/rlhf/rollback  POST   回滚到历史版本
#
# 核心代码: skills/rlhf_engine.py
# 完整文档: RLHF_EVOLUTION.md
