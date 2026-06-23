# AI Office Assistant — 智能办公助手

> **文档生成 · PDF 解析 · 会议纪要 · 知识库问答 · 偏好学习**
>
> 核心特色：**根据你的评分自动优化 AI 写作风格**——用得越多，越懂你。

---

## 📋 目录

- [快速开始](#-快速开始)
- [功能总览](#-功能总览)
- [五大模块详解](#-五大模块详解)
  - [1. AI 写作助手](#1-ai-写作助手)
  - [2. PDF 智能解析](#2-pdf-智能解析)
  - [3. 会议纪要](#3-会议纪要)
  - [4. 知识库 Q&A](#4-知识库-qa)
  - [5. 偏好学习](#5-偏好学习)
- [API 路由一览](#-api-路由一览)
- [项目结构](#-项目结构)
- [技术架构](#-技术架构)
- [常见问题](#-常见问题)
- [开发计划](#-开发计划)
- [许可证](#-许可证)

---

## 🚀 快速开始

### 前置要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.8+ | 运行时环境 |
| pip | 20+ | 包管理器 |

### 安装

```bash
# 1. 进入项目目录
cd ai-office-assistant

# 2. 安装依赖
pip install flask flask-cors numpy

# 3. 启动服务
python3 server.py
```

访问 `http://localhost:5000` 即可使用。

### 启动参数

```bash
# 自定义端口
python3 server.py --port 8080

# 自定义监听地址（默认 0.0.0.0 允许局域网访问）
python3 server.py --host 127.0.0.1

# 后台运行
nohup python3 server.py > server.log 2>&1 &
```

### 验证运行

```bash
curl http://localhost:5000/api/status
# 返回: {"status": "running", "modules": ["docgen", "pdfparse", "meeting", "ragqa", "rlhf"], "version": "1.0.0"}
```

---

## 🎯 功能总览

| 功能 | 前端 Tab | 一句话 |
|------|----------|--------|
| 🔤 AI 写作 | 文档生成 | 周报/PRD/方案/论文/公文 |
| 📄 PDF 解析 | PDF 解析 | PDF 内容提取与结构化解析 |
| 🎤 会议纪要 | 会议纪要 | 录音转写 + 结构化记录 |
| 📚 知识库问答 | 知识库 Q&A | 上传文档 → 向量检索 → 问答 |
| 🧠 偏好学习 | 模型训练 | 评分越多，AI 越懂你的风格 |

---

## 📦 五大模块详解

### 1. AI 写作助手

**支持类型：**

- 📋 **周报** — 本周工作、下周计划、风险与问题
- 📊 **产品 PRD** — 背景、目标、功能需求、技术方案
- 💼 **商务方案** — 市场分析、产品介绍、实施计划
- 🎓 **论文** — 摘要、引言、方法、实验、结论
- 📝 **公文** — 通知、报告、请示、批复
- 🔄 **润色** — 改善表达、修正语法、提升专业性
- ✂️ **改写** — 调整风格、压缩篇幅
- 📐 **缩写** — 提取核心要点

**调用 DeepSeek 大模型生成内容**，生成时自动携带基于评分的个性化提示词。

**API 端点：**

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/doc-generator/templates` | GET | 获取可用模板列表 |
| `/api/doc-generator/generate` | POST | 根据模板生成文档 |
| `/api/doc-generator/optimize` | POST | 对已有文本进行润色/改写/缩写 |

**示例：**

```bash
curl -X POST http://localhost:5000/api/doc-generator/generate \
  -H "Content-Type: application/json" \
  -d '{
    "type": "周报",
    "query": "本周完成了用户模块开发和API编写，下周开始测试工作"
  }'
```

---

### 2. PDF 智能解析

**功能：**

- 📄 上传 PDF 文件（支持扫描件与电子版）
- 🔍 自动提取正文文字
- 🧩 按章节结构化解构文档
- 🎯 关键信息自动摘要（标题、作者、日期、核心观点）
- 💾 结果导出为 JSON/Markdown

**API 端点：**

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/pdf-parser/upload` | POST | 上传 PDF 并解析 |
| `/api/pdf-parser/extract` | POST | 从已解析文档中提取关键信息 |

---

### 3. 会议纪要

**支持格式：** mp3, wav, m4a, ogg, flac

**功能：**

- 🎙️ 上传录音文件
- 📝 自动生成结构化会议纪要（议题、讨论、决议、待办）
- 💾 导出为 Markdown 文件

**API 端点：**

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/meeting-summary/upload` | POST | 上传录音文件 |
| `/api/meeting-summary/generate` | POST | 生成会议纪要 |
| `/api/meeting-summary/export` | POST | 导出会议纪要 |

---

### 4. 知识库 Q&A

轻量级 **RAG（检索增强生成）** 系统，无需外挂向量数据库。

**支持格式：** txt, md, pdf, docx

**工作流程：**

```
上传文档 → 分块(chunking) → TF-IDF向量化 → 存储向量索引
                                          ↓
用户提问 → 向量检索 → 检索Top-K相关段落 → LLM生成回答(含引用)
```

**API 端点：**

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/rag-qa/upload` | POST | 上传文档到知识库 |
| `/api/rag-qa/list` | GET | 列出知识库中所有文档 |
| `/api/rag-qa/ask` | POST | 向知识库提问 |
| `/api/rag-qa/delete` | POST | 删除指定文档 |
| `/api/rag-qa/clear` | POST | 清空整个知识库 |

---

### 5. 偏好学习 + RLHF 模拟器
---

#### 第一部分：偏好学习（在线生产系统）

每次你对 AI 写作结果评分（1-5⭐），系统自动分析偏好并生成**个性化系统提示词**，在后续的 DeepSeek API 调用中自动注入。

**工作流程：**

```
你用 AI 写作 → DeepSeek 生成内容 → 你给结果评分(1-5⭐)
                                            ↓
                    FeedbackEngine 分析评分中的偏好特征
                       （详略/结构化/语气等）↓
                    生成个性化系统提示词
                                            ↓
            下次写作 → call_llm() 自动带上该提示词
                     → DeepSeek 按你喜欢的风格生成
```

**信号强度：**

| 评分条数 | 信号强度 | 效果 |
|----------|---------|------|
| 0-4 条 | 🔄 较弱 | 仅有基础分析 |
| 5-10 条 | 👍 中等 | 能识别明显偏好 |
| 10-50 条 | 💪 较强 | 精准把握写作风格 |
| 50+ 条 | 🏆 极强 | 深度个性化 |

> 评分即训练。不需要手动点任何按钮，偏好自动生效。

**技术实现：** `skills/feedback_engine.py` → `generate_system_prompt()` 方法分析历史评分、提取偏好特征（详略度、结构化程度、正式程度），组装成 system prompt。`server.py` 中的 `call_llm()` 自动加载并注入到 DeepSeek API 请求中。

---

#### 第二部分：RLHF 模拟器

> **核心理念：** 用 DeepSeek API 模拟完整的 RLHF 流程，展示对 PPO/DPO 原理的理解。

**文件：** `skills/rlhf_simulator.py` — 一个自包含的 RLHF 模拟器，不依赖 GPU、不训练模型参数，却展示了完整的强化学习对齐流程。

##### 架构设计

```
┌──────────────────────────────────────────────────────────┐
│                    RLHF Simulator                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Phase 1: 策略采样 (Actor)                                │
│    ├─ 4 种 prompt 策略：default / structured / concise /   │
│    │   detailed，各有权重                                    │
│    ├─ 按权重采样策略（模拟 PPO 从 policy 分布中采样 action）   │
│    └─ 用不同策略调用 DeepSeek API 生成回答                   │
│                                                          │
│  Phase 2: 奖励评估 (Reward Model)                          │
│    ├─ 用 DeepSeek API 当裁判，从 5 个维度评分               │
│    │   (completeness, clarity, structure, conciseness,    │
│    │    professionalism)                                  │
│    ├─ 可选的 DPO 偏好对比（compare A vs B）                │
│    └─ 综合评分确定最优策略                                  │
│                                                          │
│  Phase 3: 策略更新 (PPO-style Update)                      │
│    ├─ 最优策略权重增加 ↑ 0.1（模拟策略梯度）                 │
│    ├─ 其他策略权重降低 ↓ 0.05                              │
│    ├─ KL 惩罚：策略名之间的差异度作为 KL 散度近似             │
│    └─ KL 裁剪：超过阈值时部分回退（模拟 PPO clip）          │
│                                                          │
│  循环迭代 → 权重收敛到用户最偏好的策略                        │
└──────────────────────────────────────────────────────────┘
```

##### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/rlhf/sim/status` | GET | 获取模拟器当前状态（策略权重/评分/历史） |
| `/api/rlhf/sim/train` | POST | 执行一步 RLHF 训练 |
| `/api/rlhf/sim/reward` | POST | 用 DeepSeek 模拟 Reward Model 评分 |
| `/api/rlhf/sim/compare` | POST | DPO 风格偏好对比 |

##### 前端演示效果

在"模型训练"Tab 底部有一个独立的 RLHF 模拟器面板，点击"执行一步 RLHF 训练"即可看到：

1. 四种策略分别调用 DeepSeek 生成回答
2. 各维度的奖励分数对比（completeness/clarity/structure/conciseness/professionalism）
3. 策略权重变化（📈📉 可视化）
4. KL 惩罚值和是否触发裁剪

---
#### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/feedback` | POST | 提交评分（1-5⭐） |
| `/api/feedback/list` | GET | 获取评分历史 |
| `/api/feedback/stats` | GET | 评分统计 + 偏好分析 |
| `/api/feedback/preference` | GET | 获取当前偏好分析详情 |
| `/api/rlhf/sim/status` | GET | RLHF 模拟器状态 |
| `/api/rlhf/sim/train` | POST | 执行一步 RLHF 训练 |
| `/api/rlhf/sim/reward` | POST | Reward Model 评分 |
| `/api/rlhf/sim/compare` | POST | DPO 偏好对比 |
| `/api/rlhf/train` | POST | 手动触发偏好更新 |
| `/api/rlhf/status` | GET | 系统状态 |

---

## 🌐 API 路由一览

```
GET    /api/status                       # 系统状态
GET    /api/doc-generator/templates      # 获取写作模板
POST   /api/doc-generator/generate       # 生成文档
POST   /api/doc-generator/optimize       # 润色/改写/缩写
POST   /api/pdf-parser/upload            # 上传并解析 PDF
POST   /api/pdf-parser/extract           # 提取关键信息
POST   /api/meeting-summary/upload       # 上传录音
POST   /api/meeting-summary/generate     # 生成会议纪要
POST   /api/meeting-summary/export       # 导出会议纪要
POST   /api/rag-qa/upload               # 上传知识库文档
GET    /api/rag-qa/list                 # 知识库文档列表
POST   /api/rag-qa/delete               # 删除知识库文档
POST   /api/rag-qa/ask                  # 知识库问答
POST   /api/rag-qa/clear                # 清空知识库
POST   /api/feedback                    # 提交评分
GET    /api/feedback/list               # 评分历史
GET    /api/feedback/stats              # 评分统计+偏好分析
GET    /api/feedback/preference         # 偏好分析详情
GET    /api/rlhf/sim/status             # RLHF 模拟器状态
POST   /api/rlhf/sim/train              # 执行一步 RLHF 训练
POST   /api/rlhf/sim/reward             # Reward Model 评分
POST   /api/rlhf/sim/compare            # DPO 偏好对比
GET    /api/rlhf/status                 # 系统状态
POST   /api/rlhf/train                  # 手动触发偏好更新
```

---

## 📁 项目结构

```
ai-office-assistant/
├── server.py                 # Flask 主服务器（20 个 API 路由 + 前端静态文件）
├── frontend/
│   └── index.html            # 单页应用前端（5 个 Tab）
├── skills/
│   ├── feedback_engine.py    # 反馈收集 + 偏好分析 + 系统提示词生成
│   ├── rag_engine.py         # RAG 知识库引擎（TF-IDF 向量检索）
│   ├── parse_document.py     # 文档解析工具（PDF/TXT/MD）
│   └── model_versions/       # （历史遗留，已停用）
├── data/
│   ├── feedback/
│   │   └── feedback.json     # 评分数据持久化存储
│   ├── kb/                   # 知识库上传文档
│   ├── vector_store/         # 向量索引
│   ├── uploads/              # 临时上传文件
│   └── outputs/              # 导出结果
├── start.sh
└── README.md                 # 本文档
```

---

## 🏗 技术架构

### 后端

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | Flask | 轻量 Python Web 服务器 |
| 大模型 | DeepSeek API | 文档生成、润色、问答 |
| 跨域 | flask-cors | CORS 支持 |
| 前端 | 纯 HTML + CSS + JS | 单页应用，无额外框架依赖 |
| 向量检索 | TF-IDF + Cosine Similarity | 自实现，无需外置向量数据库 |
| PDF 解析 | PyMuPDF / pdfminer | PDF 文本提取 |

### 核心模块架构

```
┌─────────────────────────────────────────────────────────┐
│                    Flask Server                          │
│                    (server.py)                           │
├─────────┬─────────┬──────────┬───────────┬──────────────┤
│ 写作助手  │ PDF解析 │ 会议纪要  │ 知识库QA  │  偏好学习     │
│ (模板+LLM)│(规则提取)│(转写+结构化)│(RAG引擎)  │ (反馈分析)    │
├─────────┴─────────┴──────────┴───────────┴──────────────┤
│                    DeepSeek API                           │
│              (https://api.deepseek.com)                    │
└─────────────────────────────────────────────────────────┘
                                ↑
                    ┌───────────┴────────────┐
                    │   Feedback Engine       │
                    │   (评分 → 偏好 → 提示词) │
                    └────────────────────────┘
```

---

## ❓ 常见问题

### Q: 怎么让 AI 写出我想要的风格？

使用写作助手生成文档后，给结果打星评分（1-5⭐）。每次评分都会自动优化下次写作时的系统提示词。**不需要手动训练，评分即训练。**

### Q: 偏好分析准吗？

取决于评分数量。10 条以下只能识别大致倾向，20-30 条以上才能稳定把握你的风格。评得越多越准。

### Q: 启动时提示 "Port 5000 is in use"？

```bash
# 方法1：杀死旧进程
pkill -f "python3 server.py"

# 方法2：换端口启动
python3 server.py --port 5001
```

### Q: 前端文件上传失败？

检查 `data/uploads/` 目录是否存在且可写（server 会自动创建）。大文件检查 server.py 中 `MAX_CONTENT_LENGTH` 限制。

### Q: 知识库 Q&A 不准确？

使用更精确的关键词提问。TF-IDF 检索对关键词敏感，对口语化问句效果稍差。

### Q: 怎么查看当前的偏好和系统提示词？

打开"模型训练" Tab，会显示偏好分析面板和自动生成的系统提示词预览。

### Q: 评分数据存在哪里？

`data/feedback/feedback.json`，可手动查看和编辑。

---

## 🗺 开发计划

- [ ] **多轮对话上下文** — 在写作/知识库中保持会话历史
- [ ] **批量文档解析** — 一次上传多个 PDF 并批量提取
- [ ] **用户登录与多用户** — 多用户隔离的知识库与偏好数据
- [ ] **STT 语音转文字** — 集成 whisper 实现端到端会议转写
- [ ] **偏好持久化导出** — 分享/导入偏好配置
- [ ] **Docker 部署** — 一键 docker-compose 部署

---

## 📄 许可证

本项目为个人学习实践项目，仅供学习和参考使用。

---

*Made with ❤️ — 一个越用越顺手的智能助手*
