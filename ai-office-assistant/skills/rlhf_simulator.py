#!/usr/bin/env python3
"""
RLHF 模拟器 — 用 DeepSeek API 演示完整的强化学习流程
=========================================================

为什么叫"模拟器"？
  真正的 RLHF 需要训练模型参数（PPO/DPO loss 反向传播），
  这需要 GPU + 模型权重。在这个项目中，我们用 DeepSeek API
  来模拟 RLHF 的关键环节，展示对 RLHF 流程的理解。

模拟的 RLHF 流水线：
  1. Actor（策略模型） → DeepSeek API 根据当前 prompt 生成回答
  2. Reward Model（奖励模型） → DeepSeek API 评价生成质量
  3. PPO 更新 → 根据 reward 调整 system prompt 策略
  4. KL 惩罚 → 防止策略突变（新旧 prompt 的相似度约束）

使用方法：
  from skills.rlhf_simulator import RLHF_Simulator
  sim = RLHF_Simulator()
  result = sim.train_step("写一份周报")

面试考点：
  - DPO vs PPO 核心区别
  - Reward Model 的作用
  - KL 散度在 RLHF 中的意义
  - RLHF 如何对齐人类偏好
"""

import json
import time
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 引用主服务器的 API key 配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from server import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, call_llm


class RewardModel:
    """
    奖励模型 — 对生成内容进行质量评分。
    
    真正的 RLHF 中，Reward Model 是一个单独训练的模型（通常是
    transformer 分类器），在人类标注数据上训练，学会预测人类偏好。
    
    这里我们用 DeepSeek API 模拟 Reward Model 的行为。
    """
    
    # 评分维度
    DIMENSIONS = {
        "completeness": "回答是否完整、全面覆盖了问题",
        "clarity": "表达是否清晰、逻辑是否通顺",
        "structure": "结构是否合理、层次是否分明",
        "conciseness": "是否精炼、没有冗余信息",
        "professionalism": "用词是否专业、语气是否得体",
    }
    
    def score(self, prompt: str, response: str) -> Dict[str, float]:
        """
        用 DeepSeek API 当裁判，从多个维度给回答打分。
        
        返回示例：
        {
            "completeness": 0.85,
            "clarity": 0.90,
            "structure": 0.75,
            "conciseness": 0.60,
            "professionalism": 0.80,
            "overall": 0.78
        }
        """
        dim_descs = "\n".join(
            f"- {k}: {v}" for k, v in self.DIMENSIONS.items()
        )
        
        eval_prompt = f"""你是一个专业的 AI 质量评估器（Reward Model）。
请从以下维度对 AI 的回答进行评分（0-100分），并给出总分：

评分维度：
{dim_descs}

用户问题：{prompt}

AI 回答：{response}

请严格按以下 JSON 格式输出（不要有其他文字）：
{{"completeness": <0-100>, "clarity": <0-100>, "structure": <0-100>, "conciseness": <0-100>, "professionalism": <0-100>, "overall": <0-100>, "reason": "<一句话解释>"}}"""
        
        try:
            result = call_llm(eval_prompt, system_prompt="你是一个严格的评分裁判。评分要公正准确。", use_feedback=False)
            # 解析 JSON
            result = result.strip()
            # 去掉可能的 markdown 代码块
            if result.startswith("```"):
                result = result.split("\n", 1)[-1]
                result = result.rsplit("```", 1)[0]
            result = result.strip()
            scores = json.loads(result)
            # 归一化到 0-1
            for k in list(self.DIMENSIONS.keys()) + ["overall"]:
                if k in scores:
                    scores[k] = scores[k] / 100.0
            return scores
        except Exception:
            # 解析失败时返回默认值
            return {
                "completeness": 0.5,
                "clarity": 0.5,
                "structure": 0.5,
                "conciseness": 0.5,
                "professionalism": 0.5,
                "overall": 0.5,
                "reason": "评分解析失败，使用默认值",
            }
    
    def compare(self, prompt: str, response_a: str, response_b: str) -> Dict:
        """
        对比两个回答，判断哪个更好（类似 DPO 的偏好判断）。
        
        返回：
        {
            "preferred": "A" or "B",
            "confidence": <0-1>,
            "reasons": {...}
        }
        """
        eval_prompt = f"""你是一个专业的 AI 偏好判断器。

用户问题：{prompt}

回答 A：
{response_a}

回答 B：
{response_b}

请判断哪个回答更好，并说明原因。

请严格按 JSON 格式输出：
{{"preferred": "A", "confidence": 0.85, "reason": "回答A更完整地涵盖了..."}}"""
        
        try:
            result = call_llm(eval_prompt, system_prompt="你擅长对比分析。", use_feedback=False)
            result = result.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[-1]
                result = result.rsplit("```", 1)[0]
            result = result.strip()
            data = json.loads(result)
            if data.get("preferred") not in ("A", "B"):
                data["preferred"] = "A"
            return data
        except Exception:
            return {"preferred": "A", "confidence": 0.6, "reason": "对比解析失败"}


class RLHF_Simulator:
    """
    RLHF 模拟器 — 展示完整的强化学习人类反馈流程。
    
    不同于之前的 DPO 训练（只是切换模板），这个模拟器
    用 DeepSeek API 演示了 PPO 风格的三阶段流程：
    
    Phase 1 — 数据收集：
      用不同的 system prompt 策略生成多个回答，让 reward model 评分
    
    Phase 2 — 策略优化：
      根据 reward scores 调整 prompt 策略权重（模拟 PPO 的策略梯度）
    
    Phase 3 — 部署：
      将最优策略应用到下一次生成
    
    面试亮点：
    - 展示了 RLHF 的核心 pipeline：采样 → 评估 → 更新
    - 用 API 模拟了 reward model 的功能
    - 体现了 KL 惩罚的思想（新旧 prompt 的语义距离约束）
    """
    
    def __init__(self):
        self.reward_model = RewardModel()
        self.strategies = self._init_strategies()
        self.history = []
    
    def _init_strategies(self) -> Dict[str, Dict]:
        """
        初始化一组 prompt 策略（模拟策略模型的 action space）。
        
        每个策略包含：
        - name: 策略名称
        - system_prompt: 对应的系统提示词
        - weight: 策略权重（类似 PPO 中 policy 的输出概率）
        - score: 历史平均得分
        - trials: 被尝试次数
        """
        return {
            "default": {
                "name": "默认风格",
                "system_prompt": None,  # 不加额外提示词
                "weight": 0.25,
                "score": 0.0,
                "trials": 0,
            },
            "structured": {
                "name": "结构化风格",
                "system_prompt": "请使用结构化输出：分点列出、适当使用编号和分段，让内容层次清晰。",
                "weight": 0.25,
                "score": 0.0,
                "trials": 0,
            },
            "concise": {
                "name": "精炼风格",
                "system_prompt": "请用最精炼的语言回答，直接给出核心内容，避免冗余。",
                "weight": 0.25,
                "score": 0.0,
                "trials": 0,
            },
            "detailed": {
                "name": "详尽风格",
                "system_prompt": "请提供详尽全面的回答，包含具体细节、示例和解释。",
                "weight": 0.25,
                "score": 0.0,
                "trials": 0,
            },
        }
    
    def sample_strategy(self) -> Tuple[str, Dict]:
        """
        根据权重采样一个策略（模拟 PPO 从 policy 分布采样 action）。
        
        权重高的策略被选中的概率更大，但低权重的策略也有机会探索（exploration）。
        """
        strategies = list(self.strategies.values())
        weights = [s["weight"] for s in strategies]
        total = sum(weights)
        normalized = [w / total for w in weights]
        chosen = random.choices(strategies, weights=normalized, k=1)[0]
        return chosen["name"], chosen
    
    def _kl_penalty(self, old_strategy: str, new_strategy: str) -> float:
        """
        模拟 KL 散度惩罚。
        
        真正的 RLHF 中，KL 惩罚防止更新后的策略偏离原始模型太远。
        这里我们用策略名称的差异度来模拟——切换了策略就算有 KL 代价。
        
        面试知识：
        - KL 惩罚是 PPO 的关键设计
        - 在 RLHF 中，KL 惩罚约束 policy 不要偏离 SFT 模型太远
        - DPO 隐式包含了 KL 约束，不需要额外计算
        """
        if old_strategy == new_strategy:
            return 0.0
        else:
            # 策略越不同，KL 惩罚越大
            difference_map = {
                ("default", "structured"): 0.3,
                ("default", "concise"): 0.4,
                ("default", "detailed"): 0.4,
                ("structured", "concise"): 0.5,
                ("structured", "detailed"): 0.5,
                ("concise", "detailed"): 0.6,
            }
            key = (old_strategy, new_strategy)
            reverse_key = (new_strategy, old_strategy)
            return difference_map.get(key, difference_map.get(reverse_key, 0.3))
    
    def train_step(self, prompt: str, beta: float = 0.1) -> Dict:
        """
        执行一步 RLHF 训练。
        
        参数：
        - prompt: 用户输入
        - beta: KL 惩罚系数（越大越保守）
        
        流程：
        1. 使用不同策略生成多个回答（Actor 采样）
        2. Reward Model 为每个回答打分
        3. 根据分数更新策略权重（近似 PPO 的策略梯度）
        4. 应用 KL 惩罚，避免策略突变
        
        返回完整的训练日志。
        """
        step_log = {
            "prompt": prompt,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "explorations": [],
            "best_strategy": None,
            "best_score": 0,
            "strategy_updates": [],
            "kl_penalty": 0,
        }
        
        # Phase 1: 使用每种策略生成回答并评分
        for strategy_name, strategy in self.strategies.items():
            # 生成回答
            response = call_llm(prompt, system_prompt=strategy["system_prompt"], use_feedback=False)
            
            # Reward Model 评分
            scores = self.reward_model.score(prompt, response)
            overall = scores.get("overall", 0.5)
            
            # 记录
            strategy["score"] = (strategy["score"] * strategy["trials"] + overall) / (strategy["trials"] + 1)
            strategy["trials"] += 1
            
            step_log["explorations"].append({
                "strategy": strategy_name,
                "response": response[:150] + "..." if len(response) > 150 else response,
                "scores": scores,
            })
        
        # Phase 2: 找到最佳策略
        best_name = max(self.strategies, key=lambda k: self.strategies[k]["score"])
        best_score = self.strategies[best_name]["score"]
        step_log["best_strategy"] = best_name
        step_log["best_score"] = best_score
        
        # Phase 3: 策略权重更新（模拟 PPO 梯度更新）
        old_weights = {k: v["weight"] for k, v in self.strategies.items()}
        
        for name, strategy in self.strategies.items():
            if name == best_name:
                # 最好的策略增加权重
                strategy["weight"] = min(0.7, strategy["weight"] + 0.1)
            else:
                # 其他策略降低权重
                strategy["weight"] = max(0.1, strategy["weight"] - 0.05)
        
        # 归一化权重
        total_weight = sum(s["weight"] for s in self.strategies.values())
        for s in self.strategies.values():
            s["weight"] /= total_weight
        
        # KL 惩罚
        if self.history:
            prev_best = self.history[-1].get("best_strategy", "default")
            kl = self._kl_penalty(prev_best, best_name)
            step_log["kl_penalty"] = round(kl, 3)
            
            # 如果 KL 惩罚太大，部分回退（模拟 PPO 的 clipped objective）
            if kl > 0.4:
                # 回退部分权重更新
                for name in self.strategies:
                    self.strategies[name]["weight"] = (
                        self.strategies[name]["weight"] * 0.7 + old_weights[name] * 0.3
                    )
                total_weight = sum(s["weight"] for s in self.strategies.values())
                for s in self.strategies.values():
                    s["weight"] /= total_weight
                step_log["kl_clipped"] = True
            else:
                step_log["kl_clipped"] = False
        else:
            step_log["kl_penalty"] = 0
            step_log["kl_clipped"] = False
        
        step_log["strategy_updates"] = [
            {
                "name": name,
                "weight_before": round(old_weights[name], 3),
                "weight_after": round(s["weight"], 3),
                "score": round(s["score"], 3),
                "trials": s["trials"],
            }
            for name, s in self.strategies.items()
        ]
        
        self.history.append(step_log)
        return step_log
    
    def get_status(self) -> Dict:
        """获取当前 RLHF 模拟状态"""
        return {
            "strategies": [
                {
                    "name": s["name"],
                    "weight": round(s["weight"], 3),
                    "score": round(s["score"], 3),
                    "trials": s["trials"],
                }
                for s in self.strategies.values()
            ],
            "history_length": len(self.history),
            "last_step": self.history[-1] if self.history else None,
        }
    
    def batch_train(self, prompts: List[str], epochs: int = 1) -> Dict:
        """
        批量训练多个样本。
        
        真正的 RLHF 用 batch data，这里模拟同样的概念。
        """
        log = []
        for epoch in range(epochs):
            for prompt in prompts:
                result = self.train_step(prompt)
                log.append(result)
        return {
            "total_steps": len(log),
            "epochs": epochs,
            "final_strategies": self.get_status()["strategies"],
            "log": log[-5:],  # 只返回最后 5 步
        }
    
    def get_reward_model_response(self, prompt: str, response: str) -> Dict:
        """暴露 Reward Model 的评分功能（供前端调用）"""
        return self.reward_model.score(prompt, response)


# 命令行接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RLHF 模拟器")
    parser.add_argument("action", choices=["train", "status", "demo", "reward"])
    parser.add_argument("--prompt", default="写一份本周工作总结")
    parser.add_argument("--response", default="", help="用于 reward model 评分的回答")
    parser.add_argument("--epochs", type=int, default=1)
    
    args = parser.parse_args()
    
    sim = RLHF_Simulator()
    
    if args.action == "status":
        print(json.dumps(sim.get_status(), indent=2, ensure_ascii=False))
    
    elif args.action == "train":
        result = sim.train_step(args.prompt)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.action == "demo":
        print("=" * 60)
        print("RLHF 模拟演示 — 使用 DeepSeek API 展示完整 RLHF 流程")
        print("=" * 60)
        
        examples = [
            "写一份本周工作总结",
            "介绍一个项目的技术方案",
            "写一封请假邮件",
        ]
        
        for i, prompt in enumerate(examples, 1):
            print(f"\n{'─' * 50}")
            print(f"📝 Step {i}: {prompt}")
            print(f"{'─' * 50}")
            result = sim.train_step(prompt)
            
            print(f"\n🏆 最佳策略: {result['best_strategy']} (得分: {result['best_score']:.3f})")
            print(f"📊 KL 惩罚: {result['kl_penalty']}")
            print(f"\n策略权重变化:")
            for u in result["strategy_updates"]:
                arrow = "↑" if u["weight_after"] > u["weight_before"] else "↓"
                print(f"  {u['name']}: {u['weight_before']:.3f} → {u['weight_after']:.3f} {arrow}  (评分: {u['score']:.3f})")
        
        print(f"\n{'=' * 50}")
        print(f"最终策略状态:")
        for s in sim.get_status()["strategies"]:
            print(f"  {s['name']}: weight={s['weight']:.3f}, score={s['score']:.3f}, trials={s['trials']}")
        print(f"{'=' * 50}")
    
    elif args.action == "reward":
        if args.response:
            scores = sim.get_reward_model_response(args.prompt, args.response)
            print(f"Prompt: {args.prompt}")
            print(f"Response: {args.response[:100]}...")
            print(f"Scores: {json.dumps(scores, indent=2, ensure_ascii=False)}")
        else:
            print("请用 --response 提供要评分的回答")
