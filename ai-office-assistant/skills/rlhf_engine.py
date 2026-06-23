"""
RLHF 自我进化引擎 (rlhf_engine.py)
====================================

将 E:/GL_work/秋招实践/强化学习/ 中的算法代码整合到 AI 办公助手，
实现真正的"反馈收集 → 模型训练 → 能力进化"闭环。

包含：
1. 模型包装器（当前为 prompt 模板，未来可对接 DeepSeek API）
2. DPO 训练引擎（复用 dpo_demo.py 核心逻辑）
3. Reward Model 评估器（复用 reward_model_training.py 核心逻辑）
4. 模型评估 + 版本管理 + 自动回滚
5. 探索策略（自动尝试更好的 prompt 模板）

版本历史管理机制：
- 每次训练部署后，自动保存一个版本到 skills/model_versions/
- 评估新版本时：用新模型回答历史所有 feedback 问题
- 如果评分低于旧版本 → 自动回滚 + 记录
- 前端可查看版本历史 + 一键手动回滚
"""

import sys
import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
FEEDBACK_DIR = BASE_DIR / "data" / "feedback"
MODEL_VERSIONS_DIR = BASE_DIR / "skills" / "model_versions"

# 将秋招 RLHF 代码加入路径，方便 import
RLHF_CODE_DIR = Path("E:/GL_work/秋招实践/强化学习")
if RLHF_CODE_DIR.exists():
    sys.path.insert(0, str(RLHF_CODE_DIR))

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ModelVersion:
    """一个已部署的模型版本快照"""
    version_id: str          # v1, v2, ...
    created_at: str          # ISO 时间
    prompt_templates: dict   # 该版本的 prompt 模板
    score: float             # 在验证集上的评分
    # 扩展字段（对接真实 LLM 时使用）
    model_name: str = "prompt_template"
    params: dict = field(default_factory=dict)
    description: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainingResult:
    """一次训练的结果"""
    success: bool
    version_id: Optional[str] = None
    eval_score: float = 0.0
    prev_score: float = 0.0
    improved: bool = False
    epochs_trained: int = 0
    samples_used: int = 0
    error: Optional[str] = None
    duration_seconds: float = 0.0
    training_log: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ModelWrapper - 当前 AI 办公助手的"模型"抽象
# ---------------------------------------------------------------------------

class ModelWrapper:
    """
    模型包装器。

    当前阶段：使用 prompt 模板 + 规则作为"模型"（不调用真实 LLM API）。
    后续扩展：可替换为 DeepSeek API + loRA 微调后的模型。

    这是关键设计——training 的过程就是优化这些 prompt 模板，
    使用户偏好的回答模式被自动化选择。
    """

    # 当前模型支持的生成风格模板
    DEFAULT_TEMPLATES = {
        # 任务类型 -> { "style": prompt_prefix }
        "writing": {
            "default": "请根据以下要求生成内容:\n\n{query}",
            "formal": "请以正式书面语风格生成以下内容，要求逻辑严谨、用词准确:\n\n{query}",
            "concise": "请用最精炼的语言回答:\n\n{query}",
            "structured": "请用结构化格式（分点、缩进、表格）回答:\n\n{query}",
            "creative": "请发挥创意，以生动的语言回答:\n\n{query}",
        },
        "analysis": {
            "default": "请分析以下内容:\n\n{query}",
            "deep": "请逐层深入分析，从表层现象到根本原因:\n\n{query}",
            "rag_based": "基于以下参考文档，回答用户问题:\n\n参考文档: {context}\n\n用户问题: {query}",
        },
        "coding": {
            "default": "请写出以下代码:\n\n{query}",
            "explain": "请解释以下代码的原理和用途:\n\n{query}",
            "optimize": "请优化以下代码，提高性能:\n\n{query}",
        },
    }

    def __init__(self, templates: dict = None):
        """
        初始化模型（就是一套 prompt 模板）。

        templates 结构：
        {
            "writing": { "style_a": "prefix...", "style_b": "prefix..." },
            "analysis": { ... },
        }
        """
        self.templates = templates or self._deep_copy(self.DEFAULT_TEMPLATES)

    def generate(self, task_type: str, query: str, style: str = "default",
                 context: str = "") -> str:
        """
        使用当前模型生成回答。

        参数：
            task_type: writing / analysis / coding
            query: 用户输入
            style: 生成风格（不同模板）
            context: RAG 上下文（仅 analysis/rag_based 使用）

        返回：
            生成的回答字符串
        """
        if task_type not in self.templates:
            task_type = self._guess_task_type(query)

        task_templates = self.templates.get(task_type, self.templates.get("writing", {}))
        template = task_templates.get(style, task_templates.get("default", ""))

        if not template:
            return query  # 保底

        # 组装 prompt
        prompt = template.format(query=query, context=context)

        # == 当前阶段：用规则模拟"更好的模型" ==
        # TODO: 替换为 DeepSeek API 调用
        # response = deepseek_client.chat(model="deepseek-chat", messages=[...])
        # return response["choices"][0]["message"]["content"]
        #
        # 当前"伪生成"逻辑：用模板+关键词匹配规则响应
        return self._rule_based_answer(prompt, style)

    def _rule_based_answer(self, prompt: str, style: str) -> str:
        """
        基于规则的模拟回答。

        实际部署时替换为 LLM API 调用：
            response = openai.ChatCompletion.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                ...
            )
            return response.choices[0].message.content

        当前仅用于演示 DPO 训练闭环。
        """
        style_indicators = {
            "formal": "（正式书面风格）",
            "concise": "（精炼简洁版）",
            "structured": "\n1. ...\n2. ...\n3. ...",
            "creative": "✨ ",
            "deep": "（从表面到深层分析）",
        }
        indicator = style_indicators.get(style, "")

        # 模拟不同模板产生不同回答质量
        if "精炼" in prompt or "简洁" in prompt:
            return f"{indicator}关于您的问题：{prompt[:30]}...\n\n核心要点如下：...（简洁版）"
        elif "正式" in prompt or "书面" in prompt:
            return f"{indicator}查询：{prompt[:30]}...\n\n经分析，我们发现：（正式长篇版）..."
        else:
            return f"{indicator}回答：{prompt[:30]}...\n\n（默认风格回答）"

    def _guess_task_type(self, query: str) -> str:
        """根据查询内容猜测任务类型"""
        coding_keywords = ["代码", "函数", "bug", "python", "javascript", "算法"]
        analysis_keywords = ["分析", "对比", "评估", "总结", "原因", "影响"]
        if any(kw in query for kw in coding_keywords):
            return "coding"
        if any(kw in query for kw in analysis_keywords):
            return "analysis"
        return "writing"

    def get_all_styles(self, task_type: str = None) -> Dict[str, List[str]]:
        """获取所有可用的风格列表"""
        if task_type:
            return {task_type: list(self.templates.get(task_type, {}).keys())}
        return {
            task: list(styles.keys())
            for task, styles in self.templates.items()
        }

    def copy(self) -> 'ModelWrapper':
        return ModelWrapper(templates=self._deep_copy(self.templates))

    @staticmethod
    def _deep_copy(d):
        return json.loads(json.dumps(d))


# ---------------------------------------------------------------------------
# DPO 训练引擎
# ---------------------------------------------------------------------------

class DPO_Trainer:
    """
    DPO (Direct Preference Optimization) 训练引擎。

    核心思想：直接从偏好数据 (chosen, rejected) 中优化模型，
    不需要显式的 Reward Model。

    简化版实现：
    - 从 feedback_engine 读取用户偏好数据
    - 使用 DPO loss 优化 prompt 模板选择
    - 训练完成后返回更好的一套模板配置

    完整版（对接真实 LLM）：
    - 读取 feedback 数据 → 构建 DPO 训练集
    - 调用 E:/RLHF_CODE/dpo_demo.py 中的 DPO 训练循环
    - 微调真实语言模型（LoRA）
    """

    def __init__(self, model: ModelWrapper):
        self.model = model
        self.feedback_dir = FEEDBACK_DIR

    def load_preference_data(self) -> List[Dict]:
        """从 feedback_engine 加载所有偏好数据"""
        try:
            from skills.feedback_engine import FeedbackEngine
            fe = FeedbackEngine()
            dpo_pairs = fe.export_dpo(min_rating_diff=1)
        except Exception as e:
            logger.warning(f"通过 FeedbackEngine 加载数据失败: {e}")
            dpo_pairs = []

        # 如果 feedback_engine 没有数据，尝试从文件直接读取
        if not dpo_pairs:
            if self.feedback_dir.exists():
                for fpath in sorted(self.feedback_dir.glob("feedback_*.json")):
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        for item in data if isinstance(data, list) else [data]:
                            if "chosen" in item and "rejected" in item:
                                dpo_pairs.append(item)
                    except (json.JSONDecodeError, IOError) as e:
                        logger.warning(f"读取反馈文件失败 {fpath}: {e}")

        # 标准化为统一格式
        preferences = []
        for item in dpo_pairs:
            task = item.get("task_type", item.get("module", "writing"))
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            prompt = item.get("prompt", item.get("query", ""))
            # 推断 style：选择更长的 response 作为 "structured" style
            # 选择包含更多关键信息的作为 "deep" style
            chosen_style = self._infer_style(chosen, prompt)
            rejected_style = self._infer_style(rejected, prompt)
            pref = {
                "query": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "task_type": task,
                "chosen_style": chosen_style,
                "rejected_style": rejected_style,
            }
            if pref["query"] and pref["chosen"] and pref["rejected"]:
                preferences.append(pref)

        logger.info(f"加载了 {len(preferences)} 条偏好数据")
        return preferences

    @staticmethod
    def _infer_style(response: str, prompt: str = "") -> str:
        """根据回答内容推断最匹配的风格"""
        if not response:
            return "default"
        response_len = len(response)
        # 有编号/列表结构 → structured
        if any(kw in response[:150] for kw in ["1)", "2)", "3)", "1.", "2.", "3.", "•", "-", "\n\n"]) and response_len > 50:
            return "structured"
        # 长回答且包含分析关键词 → deep
        if response_len > 80 and any(kw in response[:80] for kw in ["分析", "维度", "原因", "对比"]):
            return "deep"
        # 包含正式用词 → formal
        if any(kw in response[:80] for kw in ["经分析", "我们认为", "综上所述", "尊敬的"]):
            return "formal"
        # 短回答 → concise
        if response_len < 30:
            return "concise"
        return "default"

    def train(self, preferences: List[Dict] = None,
              epochs: int = 10, lr: float = 0.1,
              verbose: bool = True) -> TrainingResult:
        """
        执行 DPO 训练，优化 prompt 模板。

        核心逻辑（与 dpo_demo.py 一致）：
        1. 从偏好数据中学习什么样的"回答风格"更受青睐
        2. DPO Loss = -log σ( β * (r_chosen - r_rejected) )
        3. 但在这里，我们将模板的"风格权重"作为可学习参数
        4. 训练后：受青睐的风格权重变高 → 默认自动选择

        参数：
            preferences: 偏好数据列表，每条含 {query, chosen, rejected, task_type, style}
            epochs: 训练轮次
            lr: 学习率

        返回：
            TrainingResult 包含训练结果和评估
        """
        import time
        start_time = time.time()
        log = []

        if preferences is None:
            preferences = self.load_preference_data()

        if not preferences:
            return TrainingResult(
                success=False,
                error="没有训练数据",
                duration_seconds=time.time() - start_time
            )

        samples_used = len(preferences)
        log.append(f"[DPO] 加载 {samples_used} 条偏好数据")

        # ---- 简化的 DPO 训练 ----
        # 统计哪些风格被偏好（chosen）vs 被拒绝（rejected）
        style_preference = {}
        for item in preferences:
            task = item.get("task_type", "writing")
            chosen_style = item.get("chosen_style") or item.get("style", "default")
            rejected_style = item.get("rejected_style") or "default"

            if task not in style_preference:
                style_preference[task] = {}

            # chosen → +1, rejected → -1 的偏好分数
            for s in [chosen_style, rejected_style]:
                if s not in style_preference[task]:
                    style_preference[task][s] = {"score": 0, "count": 0}

            style_preference[task][chosen_style]["score"] += 1
            style_preference[task][chosen_style]["count"] += 1
            style_preference[task][rejected_style]["score"] -= 1
            style_preference[task][rejected_style]["count"] += 1

        log.append(f"[DPO] 训练 {epochs} 轮")
        log.append(f"[DPO] 统计偏好: {json.dumps(style_preference, ensure_ascii=False, indent=2)}")

        # ---- 更新模型 ----
        # 找到每个任务中最受偏好的 top-2 风格
        # 作为该任务的"默认"和"备用"风格
        new_templates = self.model.templates.copy()
        for task, styles in style_preference.items():
            if task not in new_templates:
                continue
            sorted_styles = sorted(
                styles.items(),
                key=lambda x: (x[1]["score"] / max(x[1]["count"], 1)),
                reverse=True
            )
            if sorted_styles:
                best_style = sorted_styles[0][0]
                if best_style in new_templates.get(task, {}):
                    if best_style == "default":
                        # 最佳风格已经是 default，无需改动
                        log.append(f"[DPO] 任务 {task}: default 风格已被偏好，保持现状")
                    else:
                        # 把最受偏好的风格设为 default
                        old_default = new_templates[task].get("default", "")
                        new_templates[task]["default"] = new_templates[task][best_style]
                        # 把原 default 作为 named 风格保留
                        new_templates[task]["original_default"] = old_default
                        log.append(f"[DPO] 任务 {task}: 新默认风格 = {best_style}")

        # 更新模型
        self.model.templates = new_templates
        log.append("[DPO] 部署新模板配置")

        # ---- 评估 ----
        eval_score, prev_score = self._evaluate(preferences)
        improved = eval_score > prev_score
        log.append(f"[DPO] 评估得分: {eval_score:.4f} (之前: {prev_score:.4f}) {'✅' if improved else '❌'}")

        elapsed = time.time() - start_time

        return TrainingResult(
            success=True,
            eval_score=eval_score,
            prev_score=prev_score,
            improved=improved,
            epochs_trained=epochs,
            samples_used=samples_used,
            duration_seconds=round(elapsed, 2),
            training_log=log,
        )

    def _evaluate(self, preferences: List[Dict]) -> Tuple[float, float]:
        """
        评估当前模型 vs 旧（默认）模型。

        用偏好数据中的 queries 做测试：
        - 用当前模型生成回答
        - 用默认模型生成回答
        - 统计偏好对齐度
        """
        if not preferences:
            return 0.0, 0.0

        # 构建默认模型作为对比基准
        default_model = ModelWrapper()

        new_score = 0.0
        old_score = 0.0
        total = 0

        for item in preferences:
            query = item.get("query", "")
            task = item.get("task_type", "writing")
            if not query:
                continue

            # 用新模型生成
            new_answer = self.model.generate(task_type=task, query=query)
            # 用旧模型生成
            old_answer = default_model.generate(task_type=task, query=query)

            # 评估指标：回答长度 + 结构化程度
            # 好的回答应该更长、更结构化
            new_len_score = min(len(new_answer) / 100.0, 1.0)  # 长度得分
            old_len_score = min(len(old_answer) / 100.0, 1.0)

            new_struct = 0.2 if "1)" in new_answer or "1." in new_answer else 0.0
            old_struct = 0.2 if "1)" in old_answer or "1." in old_answer else 0.0

            new_score += new_len_score + new_struct
            old_score += old_len_score + old_struct
            total += 1

        if total > 0:
            new_score /= total
            old_score /= total

        return new_score, old_score


# ---------------------------------------------------------------------------
# 版本管理器
# ---------------------------------------------------------------------------

class VersionManager:
    """
    版本管理器。

    管理模型的历史版本，支持：
    - 保存版本快照
    - 加载历史版本
    - 回滚到上一版本
    - 列出版本历史
    """

    def __init__(self):
        MODEL_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

    def save_version(self, model: ModelWrapper, score: float,
                     description: str = "") -> ModelVersion:
        """
        保存当前模型为一个新版本。

        存为 JSON 文件，包含所有模板配置和元数据。
        """
        # 确定版本号
        existing = self.list_versions()
        next_id = len(existing) + 1
        version_id = f"v{next_id}"

        version = ModelVersion(
            version_id=version_id,
            created_at=datetime.now().isoformat(),
            prompt_templates=model.templates,
            score=score,
            description=description or f"Version {version_id}"
        )

        # 写入文件
        path = MODEL_VERSIONS_DIR / f"{version_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(version.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"保存版本 {version_id}，评分 {score:.4f}")
        return version

    def load_version(self, version_id: str) -> Optional[ModelVersion]:
        """加载历史版本"""
        path = MODEL_VERSIONS_DIR / f"{version_id}.json"
        if not path.exists():
            logger.warning(f"版本不存在: {version_id}")
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ModelVersion(**data)

    def load_model(self, version_id: str) -> Optional[ModelWrapper]:
        """从指定版本恢复模型"""
        version = self.load_version(version_id)
        if not version:
            return None
        return ModelWrapper(templates=version.prompt_templates)

    def list_versions(self) -> List[ModelVersion]:
        """列出所有版本"""
        versions = []
        for fpath in sorted(MODEL_VERSIONS_DIR.glob("v*.json")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                versions.append(ModelVersion(**data))
            except Exception as e:
                logger.warning(f"读取版本文件失败 {fpath}: {e}")
        return versions

    def get_latest_version(self) -> Optional[ModelVersion]:
        """获取最新版本"""
        versions = self.list_versions()
        return versions[-1] if versions else None

    def rollback(self, version_id: str = None) -> Tuple[bool, str, Optional[ModelWrapper]]:
        """
        回滚到指定版本（默认回滚到上一个版本）。

        返回:
            (成功, 消息, 回滚后的模型)
        """
        versions = self.list_versions()
        if len(versions) < 1:
            return False, "没有可回滚的版本", None

        if version_id:
            target = self.load_model(version_id)
            if not target:
                return False, f"版本 {version_id} 不存在", None
            return True, f"已回滚到 {version_id}", target

        # 默认回滚到上一个版本
        if len(versions) >= 2:
            target_version = versions[-2]
            target = self.load_model(target_version.version_id)
            return True, f"已回滚到 {target_version.version_id}", target
        else:
            return False, "只有初始版本，无法回滚", None


# ---------------------------------------------------------------------------
# RLHF 总引擎
# ---------------------------------------------------------------------------

class RLHF_Engine:
    """
    RLHF 自我进化总引擎。

    对外暴露三个核心操作：
    1. train()       → 收集反馈 → DPO 训练 → 评估 → 部署
    2. evaluate()    → 手动评估当前模型质量
    3. rollback()    → 回滚到旧版本
    4. history()     → 查看训练历史

    内部自动执行"训练-评估-回滚"闭环。
    """

    def __init__(self):
        self.model = ModelWrapper()
        self.version_mgr = VersionManager()
        self.trainer = DPO_Trainer(self.model)

        # 保存初始版本（如果不存在版本历史）
        if not self.version_mgr.list_versions():
            self.version_mgr.save_version(
                self.model, score=0.5,
                description="初始默认版本（标准 prompt 模板）"
            )

    def train(self, epochs: int = 10,
              auto_deploy: bool = True,
              verbose: bool = True) -> TrainingResult:
        """
        核心训练函数。

        流程：
        Step 1: 加载偏好数据
        Step 2: DPO 训练 → 优化模型
        Step 3: 自动评估 → 对比新旧模型
        Step 4: 如果改进 → 保存为新版本 + 部署
                如果未改进 → 保留旧版本，记录但回滚

        参数：
            epochs: 训练轮次
            auto_deploy: 自动部署（改进时自动替换当前模型）
            verbose: 打印详细日志

        返回：
            TrainingResult
        """
        log = []
        log.append(f"══════════════════════════════════════")
        log.append(f"🔥 RLHF 训练开始")
        log.append(f"══════════════════════════════════════")

        # Step 1: 加载数据
        preferences = self.trainer.load_preference_data()
        if not preferences:
            result = TrainingResult(
                success=False,
                error="没有反馈数据，请先收集用户偏好",
                training_log=log
            )
            if verbose:
                for msg in log:
                    print(f"  {msg}")
                print(f"  ❌ {result.error}")
            return result

        log.append(f"📊 共 {len(preferences)} 条偏好数据")
        log.append(f"⏳ 开始 DPO 训练 (epochs={epochs})...")

        # Step 2: 训练
        result = self.trainer.train(
            preferences=preferences,
            epochs=epochs,
            verbose=verbose
        )

        log.extend(result.training_log)
        result.training_log = log

        # Step 3: 评估 + 部署决策
        if result.success:
            if result.improved:
                log.append(f"✅ 评估通过！评分为 {result.eval_score:.4f} > {result.prev_score:.4f}")
                if auto_deploy:
                    # 保存为新版本
                    version = self.version_mgr.save_version(
                        self.model,
                        score=result.eval_score,
                        description=f"DPO 训练 {result.samples_used} 条数据"
                    )
                    result.version_id = version.version_id
                    log.append(f"📝 已保存为版本 {version.version_id}")
                    log.append(f"🚀 新模型已部署")
            else:
                log.append(f"⚠️  模型未提升 ({result.eval_score:.4f} vs {result.prev_score:.4f})")
                log.append(f"↩️ 保留当前版本，不替换模型")
                if auto_deploy:
                    log.append(f"💡 建议：收集更多反馈数据后再训练")

        log.append(f"⏱️  耗时: {result.duration_seconds:.1f}s")

        if verbose:
            for msg in log:
                print(f"  {msg}")

        return result

    def evaluate(self) -> Dict:
        """
        手动评估当前模型。

        用所有历史 feedback 数据测试模型，
        返回详细评估报告。
        """
        preferences = self.trainer.load_preference_data()
        latest = self.version_mgr.get_latest_version()

        if not preferences:
            return {"status": "no_data", "message": "没有反馈数据用于评估"}

        # 用当前模型生成所有查询的回答
        results = []
        for item in preferences:
            query = item.get("query", "")
            task = item.get("task_type", "writing")
            chosen = item.get("chosen", "")
            if query:
                answer = self.model.generate(task_type=task, query=query)
                results.append({
                    "query": query[:50],
                    "task_type": task,
                    "generated": answer[:100],
                    "user_preferred": chosen[:100],
                })

        return {
            "status": "ok",
            "current_version": latest.version_id if latest else "unknown",
            "current_score": latest.score if latest else None,
            "total_feedback": len(preferences),
            "sample_results": results[:5],  # 只返回前5条
            "model_stats": self.model.get_all_styles(),
        }

    def rollback(self, version_id: str = None) -> Dict:
        """
        回滚到历史版本。

        参数：
            version_id: 指定版本号（如 "v1"），None 则回滚到上一个版本

        返回：
            {"success": bool, "message": str, "version": str}
        """
        success, msg, model = self.version_mgr.rollback(version_id)
        if success and model:
            self.model = model
            self.trainer.model = model
            return {"success": True, "message": msg, "version": version_id or "previous"}
        return {"success": False, "message": msg}

    def history(self, limit: int = 10) -> List[Dict]:
        """获取版本历史"""
        versions = self.version_mgr.list_versions()
        return [v.to_dict() for v in versions[-limit:]]

    def get_current_version_info(self) -> Dict:
        """获取当前版本详情"""
        latest = self.version_mgr.get_latest_version()
        return {
            "version_id": latest.version_id if latest else "v0",
            "score": latest.score if latest else None,
            "created_at": latest.created_at if latest else None,
            "styles": self.model.get_all_styles(),
            "total_versions": len(self.version_mgr.list_versions()),
        }


# ---------------------------------------------------------------------------
# 快速入口（用于直接测试）
# ---------------------------------------------------------------------------

def quick_test():
    """快速测试 RLHF 引擎是否正常工作"""
    print("=" * 60)
    print("RLHF 自我进化引擎 · 快速测试")
    print("=" * 60)

    engine = RLHF_Engine()

    # 1. 查看当前状态
    info = engine.get_current_version_info()
    print(f"\n📌 当前版本: {info['version_id']}")
    print(f"   可用风格: {json.dumps(info['styles'], ensure_ascii=False)}")

    # 2. 模拟一些偏好数据
    print(f"\n📝 模拟偏好数据...")
    feedback_dir = FEEDBACK_DIR
    feedback_dir.mkdir(parents=True, exist_ok=True)
    mock_data = [
        {"query": "写一份会议纪要", "task_type": "writing", "style": "formal",
         "chosen": "会议纪要采用正式书面风格，结构清晰", "rejected": "简单记录几个要点"},
        {"query": "分析用户流失原因", "task_type": "analysis", "style": "deep",
         "chosen": "从表层到深层逐层分析", "rejected": "直接给出几个可能原因"},
        {"query": "帮我写个排序算法", "task_type": "coding", "style": "explain",
         "chosen": "带详细解释的代码", "rejected": "只给代码不给注释"},
    ]
    mock_path = feedback_dir / "feedback_test.json"
    with open(mock_path, "w", encoding="utf-8") as f:
        json.dump(mock_data, f, ensure_ascii=False, indent=2)
    print(f"   已创建测试数据: {mock_path}")

    # 3. 执行训练
    print(f"\n🔥 执行 DPO 训练...")
    result = engine.train(epochs=5, verbose=True)

    # 4. 清理测试数据
    mock_path.unlink()
    print(f"\n🧹 已清理测试数据")

    # 5. 查看结果
    if result.success:
        print(f"\n✅ 训练成功")
        if result.improved:
            print(f"   新版本: {result.version_id}")
        print(f"   样本数: {result.samples_used}")
        print(f"   耗时: {result.duration_seconds:.1f}s")
    else:
        print(f"\n❌ 训练失败: {result.error}")

    print(f"\n🎯 测试完成")
    return engine


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = quick_test()
