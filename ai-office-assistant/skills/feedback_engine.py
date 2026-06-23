#!/usr/bin/env python3
"""
RLHF 偏好反馈收集引擎
================================
用途：收集用户对 LLM 回答的偏好，构建 DPO 训练数据集

核心概念：
  - 每条反馈包含：prompt + chosen_response + rejected_response + 评分
  - 积累了反馈数据后，可导出为 DPO 格式用于模型训练
  - 你之前学的 rlhf_learning/dpo_demo.py 就是用这种数据格式

DPO 偏好数据格式（导出时生成）：
  {
    "prompt": "用户的输入 / 系统提示词",
    "chosen": "高质量的回答（用户点赞/高分）",
    "rejected": "低质量的回答（用户踩/低分）"
  }

使用方法（命令行 / API）：
  # 收集反馈
  python feedback_engine.py add --prompt "..." --response "..." --rating 5

  # 查看反馈列表
  python feedback_engine.py list

  # 导出 DPO 偏好数据集
  python feedback_engine.py export --min_rating_diff 2

  # 查看统计
  python feedback_engine.py stats
"""

import json
import time
import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

# ---- 配置 ----
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FEEDBACK_DIR = DATA_DIR / "feedback"
FEEDBACK_INDEX = FEEDBACK_DIR / "feedback.json"
DPO_EXPORT_DIR = FEEDBACK_DIR / "dpo_exports"

FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
DPO_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 反馈引擎核心类 ----
class FeedbackEngine:
    """
    偏好反馈引擎

    数据模型（feedback.json）：
    {
      "feedbacks": [
        {
          "id": "feedback_xxx",
          "timestamp": "2024-01-01T12:00:00",
          "module": "doc-generator",        -- 来源模块
          "prompt": "用户输入/系统提示词",
          "response": "LLM 生成的回答",
          "rating": 5,                       -- 1-5 评分
          "feedback_text": "用户备注（可选）",
          "tags": ["准确", "流畅"],           -- 用户打的标签
          "metadata": {
            "model": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 4096,
            "source": "api_call"             -- 调用来源
          }
        }
      ],
      "stats": {
        "total_feedbacks": 100,
        "avg_rating": 3.8,
        "module_counts": {"doc-gen": 30, "rag-qa": 50, "meeting": 20},
        "last_updated": "2024-01-01T12:00:00"
      }
    }
    """

    def __init__(self):
        self.data = self._load()

    def _load(self) -> dict:
        """加载反馈数据库"""
        if FEEDBACK_INDEX.exists():
            return json.loads(FEEDBACK_INDEX.read_text(encoding="utf-8"))
        return {"feedbacks": [], "stats": {
            "total_feedbacks": 0, "avg_rating": 0.0,
            "module_counts": {}, "last_updated": ""
        }}

    def _save(self):
        """保存反馈数据库"""
        FEEDBACK_INDEX.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def _update_stats(self):
        """更新统计信息"""
        fbs = self.data["feedbacks"]
        if not fbs:
            self.data["stats"] = {
                "total_feedbacks": 0, "avg_rating": 0.0,
                "module_counts": {}, "last_updated": ""
            }
            return

        ratings = [fb["rating"] for fb in fbs]
        module_counts = {}
        for fb in fbs:
            m = fb.get("module", "unknown")
            module_counts[m] = module_counts.get(m, 0) + 1

        self.data["stats"] = {
            "total_feedbacks": len(fbs),
            "avg_rating": round(sum(ratings) / len(ratings), 2),
            "module_counts": module_counts,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S")
        }

    def add_feedback(
        self,
        prompt: str,
        response: str,
        rating: int,
        module: str = "unknown",
        feedback_text: str = "",
        tags: List[str] = None,
        metadata: Dict = None
    ) -> Dict:
        """
        收集一条反馈

        参数:
            prompt: 用户输入/系统提示词
            response: LLM 生成的回答
            rating: 评分 1-5（1=很差, 5=很好）
            module: 来源模块（doc-generator/rag-qa/meeting-summary/...）
            feedback_text: 用户备注
            tags: 标签列表
            metadata: 额外信息
        """
        if not 1 <= rating <= 5:
            return {"error": "评分必须在 1-5 之间"}

        feedback_id = hashlib.md5(
            f"{prompt}{response}{time.time()}".encode()
        ).hexdigest()[:12]

        feedback = {
            "id": f"fb_{feedback_id}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "module": module,
            "prompt": prompt,
            "response": response,
            "rating": rating,
            "feedback_text": feedback_text,
            "tags": tags or [],
            "metadata": metadata or {},
        }

        self.data["feedbacks"].append(feedback)
        self._update_stats()
        self._save()

        return {"feedback_id": feedback["id"], "message": f"反馈已记录，评分: {rating}/5"}

    def list_feedbacks(self, limit: int = 50, offset: int = 0, module: str = None) -> List[Dict]:
        """查询反馈列表"""
        fbs = self.data["feedbacks"]
        if module:
            fbs = [fb for fb in fbs if fb.get("module") == module]
        return list(reversed(fbs))[offset:offset + limit]

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.data["stats"]

    def export_dpo(self, min_rating: int = 4, max_rating: int = 2, min_rating_diff: int = 2) -> List[Dict]:
        """
        导出 DPO 偏好数据集

        策略：
          - chosen = 评分 >= min_rating 的回答（高质量）
          - rejected = 评分 <= max_rating 的回答（低质量）
          - 同一个 prompt 有多个回答时，自动配对

        参数:
            min_rating: chosen 的最低评分（默认 4）
            max_rating: rejected 的最高评分（默认 2）
            min_rating_diff: chosen 和 rejected 的最小评分差（默认 2）

        返回:
            DPO 格式的偏好对列表
        """
        fbs = self.data["feedbacks"]
        if not fbs:
            return []

        # 按 prompt 分组
        from collections import defaultdict
        prompt_groups = defaultdict(list)
        for fb in fbs:
            # 使用 prompt 前 200 字符作为分组 key
            prompt_key = fb["prompt"][:200] + fb.get("module", "")
            prompt_groups[prompt_key].append(fb)

        dpo_pairs = []
        for prompt_key, group in prompt_groups.items():
            chosen = [fb for fb in group if fb["rating"] >= min_rating]
            rejected = [fb for fb in group if fb["rating"] <= max_rating]

            for c in chosen:
                for r in rejected:
                    if abs(c["rating"] - r["rating"]) >= min_rating_diff:
                        dpo_pairs.append({
                            "prompt": c["prompt"],
                            "chosen": c["response"],
                            "rejected": r["response"],
                            "rating_diff": abs(c["rating"] - r["rating"]),
                            "module": c.get("module", ""),
                        })

        return dpo_pairs

    def save_dpo_export(self, dpo_pairs: List[Dict], filename: str = None) -> str:
        """
        将 DPO 偏好数据保存为 JSONL 文件

        JSONL 格式每行一个 JSON 对象，可以直接用于 DPO 训练（见 dpo_demo.py）
        """
        if not filename:
            filename = f"dpo_dataset_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

        filepath = DPO_EXPORT_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            for pair in dpo_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        return str(filepath)

    def get_dpo_stats(self) -> Dict:
        """获取 DPO 数据概览"""
        dpo_pairs = self.export_dpo()
        if not dpo_pairs:
            return {"dpo_pairs": 0, "avg_rating_diff": 0, "modules": []}

        diffs = [p["rating_diff"] for p in dpo_pairs]
        modules = list(set(p["module"] for p in dpo_pairs))

        return {
            "dpo_pairs": len(dpo_pairs),
            "avg_rating_diff": round(sum(diffs) / len(diffs), 2),
            "modules": modules,
        }

    def generate_system_prompt(self) -> str:
        """
        根据所有历史反馈，自动生成个性化系统提示词。

        分析用户的评分习惯，提取偏好特征（详略、格式、语气等），
        生成优化后的 system prompt，让大模型按用户喜欢的风格输出。
        """
        fbs = self.data["feedbacks"]
        if not fbs:
            return ""

        # 只分析评分差异明显的反馈（评分 4-5 或 1-2）
        high_rated = [fb for fb in fbs if fb["rating"] >= 4]
        low_rated = [fb for fb in fbs if fb["rating"] <= 2]

        preferences = []

        # 分析详略偏好
        if high_rated:
            # 检查高分的回答长度特征
            long_count = 0
            structured_count = 0
            for fb in high_rated:
                resp = fb.get("response", "")
                if len(resp) > 200:
                    long_count += 1
                # 检测是否包含结构化元素（编号、列表、表格等）
                if any(c in resp for c in ["1)", "1.", "-", "\n\n", "|", "**"]):
                    structured_count += 1

            total_high = len(high_rated)
            if long_count / total_high > 0.6:
                preferences.append("用户偏好详细、完整的回答")
            else:
                preferences.append("用户偏好简洁、精炼的回答")

            if structured_count / total_high > 0.6:
                preferences.append("用户偏好结构化输出（分点、分段、列举）")
            else:
                preferences.append("用户偏好自然段落式输出")

        # 分析差评原因
        if low_rated:
            short_count = 0
            casual_count = 0
            for fb in low_rated:
                resp = fb.get("response", "")
                if len(resp) < 100:
                    short_count += 1
                # 检测是否过于口语化
                casual_words = ["可以", "行", "好", "嗯", "ok"]
                if any(w in resp for w in casual_words) and len(resp) < 150:
                    casual_count += 1

            total_low = len(low_rated)
            if short_count / total_low > 0.4:
                preferences.append("用户不认可过于简略的回答")
            if casual_count / total_low > 0.3:
                preferences.append("用户不认可过于随意的表达")

        # 组装系统提示词
        instructions = []
        for pref in preferences:
            if "结构化" in pref:
                instructions.append("请使用结构化输出：分段清晰、适当使用编号、表格或列表来组织内容")
            elif "详细" in pref:
                instructions.append("请提供详尽、全面的回答，包含具体细节和示例")
            elif "简洁" in pref:
                instructions.append("请保持回答简洁精炼，重点突出")
            elif "自然段落" in pref:
                instructions.append("请使用自然段落式写作，保持流畅易读")

        if not instructions:
            return ""

        # 标记反馈数据量
        signal_strength = "较强" if len(fbs) >= 10 else "初步" if len(fbs) >= 5 else "较弱"

        prompt = (
            f"你是一位专业的 AI 办公助手。基于用户反馈（{signal_strength}偏好信号），"
            f"请遵循以下风格指导：\n\n"
        )
        prompt += "\n".join(f"- {inst}" for inst in instructions)
        prompt += (
            f"\n\n（当前基于 {len(fbs)} 条用户反馈自动优化，评分率 {len(high_rated)}赞/{len(low_rated)}踩）"
        )

        return prompt

    def get_preference_summary(self) -> Dict:
        """获取偏好摘要，用于前端展示"""
        fbs = self.data["feedbacks"]
        if not fbs:
            return {"status": "no_data", "total": 0, "preferences": [], "system_prompt": ""}

        high = [fb for fb in fbs if fb["rating"] >= 4]
        mid = [fb for fb in fbs if fb["rating"] == 3]
        low = [fb for fb in fbs if fb["rating"] <= 2]

        prompt = self.generate_system_prompt()

        # 提取偏好关键词
        pref_lines = [l for l in prompt.split("\n") if l.startswith("- ")]

        return {
            "status": "active",
            "total": len(fbs),
            "high_ratings": len(high),
            "mid_ratings": len(mid),
            "low_ratings": len(low),
            "avg_rating": round(sum(fb["rating"] for fb in fbs) / len(fbs), 2),
            "preferences": pref_lines,
            "system_prompt": prompt,
            "signal_strength": "strong" if len(fbs) >= 10 else "medium" if len(fbs) >= 5 else "weak",
        }


# ---- 命令行接口 ----
def print_separator():
    print("\n" + "=" * 60)

if __name__ == "__main__":
    engine = FeedbackEngine()

    if len(sys.argv) < 2:
        print("RLHF 偏好反馈收集引擎")
        print("=" * 40)
        print("用法:")
        print("  python feedback_engine.py add --prompt '...' --response '...' --rating 5")
        print("  python feedback_engine.py list [--limit 20] [--module rag-qa]")
        print("  python feedback_engine.py stats")
        print("  python feedback_engine.py export [--min_rating_diff 2]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "add":
        # python feedback_engine.py add --prompt "..." --response "..." --rating 5
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--prompt", required=True)
        parser.add_argument("--response", required=True)
        parser.add_argument("--rating", type=int, required=True)
        parser.add_argument("--module", default="cli")
        parser.add_argument("--feedback_text", default="")
        parser.add_argument("--tags", default="")
        args = parser.parse_args(sys.argv[2:])

        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        result = engine.add_feedback(
            prompt=args.prompt,
            response=args.response,
            rating=args.rating,
            module=args.module,
            feedback_text=args.feedback_text,
            tags=tags,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif command == "list":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--limit", type=int, default=20)
        parser.add_argument("--offset", type=int, default=0)
        parser.add_argument("--module", default=None)
        args = parser.parse_args(sys.argv[2:])

        fbs = engine.list_feedbacks(limit=args.limit, offset=args.offset, module=args.module)
        print_separator()
        print(f"反馈列表（共 {len(fbs)} 条）\n")
        for fb in fbs:
            rating_bar = "\u2605" * fb["rating"] + "\u2606" * (5 - fb["rating"])
            print(f"  [{fb['module']}] {rating_bar} ({fb['rating']}/5)")
            print(f"  Prompt: {fb['prompt'][:80]}...")
            print(f"  Response: {fb['response'][:80]}...")
            print(f"  ID: {fb['id']} | {fb['timestamp']}\n")

    elif command == "stats":
        stats = engine.get_stats()
        print_separator()
        print("反馈统计\n")
        print(f"  总反馈数: {stats['total_feedbacks']}")
        print(f"  平均评分: {stats['avg_rating']}/5")
        print(f"  模块分布: {stats['module_counts']}")
        print(f"  最后更新: {stats['last_updated']}")

        # DPO 数据概览
        dpo_stats = engine.get_dpo_stats()
        print(f"\n  DPO 偏好数据概览:")
        print(f"    可导出偏好对: {dpo_stats['dpo_pairs']}")
        print(f"    平均评分差: {dpo_stats['avg_rating_diff']}")
        print(f"    覆盖模块: {dpo_stats['modules']}")

    elif command == "train":
        """触发 DPO 训练"""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--verbose", action="store_true", default=True)
        args = parser.parse_args(sys.argv[2:])

        from skills.rlhf_engine import RLHF_Engine, VersionManager
        rlhf_engine = RLHF_Engine()
        result = rlhf_engine.train(epochs=args.epochs, verbose=args.verbose)

        print_separator()
        print("RLHF 训练结果\n")
        if result.success:
            print(f"  ✅ 训练成功")
            if result.version_id:
                print(f"     新版本: {result.version_id}")
                print(f"     评分提升: {result.eval_score:.3f} (之前 {result.prev_score:.3f})")
            print(f"     训练样本: {result.samples_used}")
            print(f"     训练轮次: {result.epochs_trained}")
            print(f"     耗时: {result.duration_seconds:.1f}s")
            if not result.improved:
                print(f"  ⚠️  模型未显著提升，未部署新版本")
        else:
            print(f"  ❌ 训练失败: {result.error}")

        # 显示版本历史
        vm = VersionManager()
        versions = vm.list_versions()
        if len(versions) > 1:
            print(f"\n  版本历史:")
            for v in versions[-5:]:
                marker = "◀ 当前" if v == versions[-1] and result.success and result.version_id else ""
                print(f"    {v.version_id}: 评分 {v.score:.3f} | {v.created_at[:19]} {marker}")

    elif command == "export":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--min_rating_diff", type=int, default=2)
        parser.add_argument("--filename", default=None)
        args = parser.parse_args(sys.argv[2:])

        dpo_pairs = engine.export_dpo(min_rating_diff=args.min_rating_diff)
        filepath = engine.save_dpo_export(dpo_pairs, args.filename)

        print_separator()
        print(f"DPO 偏好数据导出完成")
        print(f"  偏好对数量: {len(dpo_pairs)}")
        print(f"  文件路径: {filepath}")
        print(f"  数据格式: JSONL（每行一对 prompt/chosen/rejected）")
        print(f"  可用于: dpo_demo.py 训练 / HuggingFace TRL DPOTrainer")

        if dpo_pairs:
            print(f"\n  示例偏好对:")
            example = dpo_pairs[0]
            print(f"    prompt: {example['prompt'][:60]}...")
            print(f"    chosen (好评): {example['chosen'][:60]}...")
            print(f"    rejected (差评): {example['rejected'][:60]}...")
            print(f"    评分差: {example['rating_diff']}")

    else:
        print(f"未知命令: {command}")
        sys.exit(1)