# 🧠 AI 办公助手 — RLHF 自我进化引擎

> 将秋招学的 RLHF 算法（PPO/DPO/RM/GAE）真正嵌入一个可用系统，
> 实现"收集反馈 → DPO 训练 → 自动评估 → 部署进化"的完整闭环。

---

## 📂 项目结构

```
ai-office-assistant/
├── server.py                      ← Flask 服务端（含 RLHF API 路由）
├── frontend/
│   └── index.html                 ← SPA 前端（含 ✨ 模型训练 Tab）
├── skills/
│   ├── feedback_engine.py         ← 偏好反馈收集引擎（已有）
│   ├── rag_engine.py              ← RAG 知识库（已有）
│   ├── parse_document.py          ← 文档解析（已有）
│   ├── rlhf_engine.py             ← ⚡ NEW: RLHF 自我进化引擎
│   └── model_versions/            ← 自动生成的模型版本快照
├── data/
│   ├── feedback/                  ← 用户反馈数据 JSON 文件
│   ├── knowledge_base/            ← RAG 知识库文件
│   └── outputs/                   ← 文档 / 报告输出
└── insert_routes.py               ← 路由注入脚本（已更新说明）
```

> 秋招学习的 RLHF 算法代码位于 `E:\GL_work\秋招实践\强化学习\`
> 引擎通过 `sys.path.insert(0, ...)` 导入其核心逻辑。

---

## 🔄 闭环流程

```
        用户使用 AI 助手
              │
              ▼
    ┌───────────────────┐
    │  反馈收集 (已有)    │ ← feedback_engine.py，收集偏好数据
    │  /api/feedback     │    每对数据包含 {chosen, rejected}
    └────────┬──────────┘
             │ 积累 >= N 条
             ▼
    ┌───────────────────┐
    │  DPO 训练 (新增)    │ ← rlhf_engine.py，调用 DPO 算法
    │  /api/rlhf/train   │    优化 prompt 模板权重
    └────────┬──────────┘
             │ 训练完成
             ▼
    ┌───────────────────┐
    │  自动评估 (新增)    │ ← 用历史反馈测试新模型
    │  /api/rlhf/evaluate│    对比新旧模型评分
    └───────┬───────────┘
          ╱        ╲
        ✅          ❌
       评分提升      评分下降
         │            │
         ▼            ▼
    ┌──────────┐  ┌──────────┐
    │ 部署新版本 │  │ 自动回滚  │
    │ + 保存    │  │ + 记录日志│
    └──────────┘  └──────────┘
```

---

## 🧬 关键设计

### 1. 什么是"模型"？

**当前阶段：** prompt 模板 = 模型。

```
"writing": {
  "default":   "请根据以下要求生成内容:\n{query}"
  "formal":    "请以正式书面语风格生成..."
  "concise":   "请用最精炼的语言回答..."
  "structured":"请用结构化格式回答..."
  "creative":  "请发挥创意..."
}
```

每个任务类型（writing/analysis/coding）都有多个**风格模板**。
DPO 训练不生成 token，而是**学习什么风格更受青睐**，自动切换。

### 2. DPO 训练做了什么？

1. 从 `data/feedback/` 读取所有 `{chosen, rejected}` 偏好对
2. 统计每个任务类型下，哪些风格被用户标记为"chosen" vs "rejected"
3. 计算偏好分数 = chose_count - rejected_count
4. 更新模型：把评分最高的风格设为该任务的 **default**
5. 保存为新版本 + 自动评估

### 3. 版本管理

| 概念 | 实现 |
|------|------|
| 版本保存 | `skills/model_versions/v1.json` — 包含模板 + 评分 |
| 自动回滚 | 新模型评分低于旧版 → 保留当前版本，不下发 |
| 手动回滚 | `POST /api/rlhf/rollback` — 回到上一个版本 |
| 版本历史 | `GET /api/rlhf/history` — 查看所有版本和评分 |

---

## 🚀 API 端点一览

| 端点 | 方法 | 用途 |
|------|------|------|
| `/api/rlhf/status` | GET | 查看当前模型版本、评分、风格 |
| `/api/rlhf/train` | POST | 触发 DPO 训练 |
| `/api/rlhf/evaluate` | GET | 手动评估当前模型 |
| `/api/rlhf/history` | GET | 查看版本历史 |
| `/api/rlhf/rollback` | POST | 回滚到历史版本 |

### 训练请求示例

```json
POST /api/rlhf/train
{
  "epochs": 10,
  "auto_deploy": true
}
```

### 训练响应示例

```json
{
  "success": true,
  "version_id": "v2",
  "eval_score": 0.723,
  "prev_score": 0.500,
  "improved": true,
  "samples_used": 25,
  "epochs_trained": 10,
  "duration_seconds": 0.35,
  "training_log": ["[DPO] 加载 25 条偏好数据", ...]
}
```

---

## 🎯 秋招面试价值

这个系统展示了你具备 **工程 + 算法的综合能力**：

1. **RLHF 理论理解**：DPO、PPO、Reward Model、GAE 等核心概念
2. **系统设计能力**：训练-评估-部署-回滚的完整闭环
3. **工程落地能力**：将算法代码集成到真实 Web 应用中
4. **迭代思维**：版本管理、自动回滚的容错机制

**面试示例回答：**
> "我实现了一个 RLHF 自我进化引擎，用户对 AI 回答的偏好反馈会积累下来，
> 达到一定量后自动触发 DPO 训练，优化模型的回答风格。训练后自动评估，
> 效果变差就回滚，变好就部署为新版本。虽然现在是 prompt 级别优化，
> 但这个架构可以无缝升级到真正的模型微调。"

---

## 🔜 未来扩展方向

| 升级路径 | 改动量 | 效果 |
|---------|--------|------|
| 接入 DeepSeek API | `ModelWrapper.generate()` 替换 1 个函数 | 真正 LLM 回答 |
| LoRA 微调 | DPO 训练换成真实 DPO loss | 模型参数级优化 |
| Reward Model | 训练一个独立的 RM，代替模板评分 | 更精确的评估 |
| 自动阈值 | feedback 积累到 50/100/200 条自动触发训练 | 全自动化 |
| AB 测试 | 新模型只对 10% 用户生效，验证效果后全量 | 安全部署 |

---

**代码位置：** `skills/rlhf_engine.py` (24KB 纯 Python，零额外依赖)
