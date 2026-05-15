AI 办公助手（大模型应用开发测试）
该项目是基于 OpenClaw 开发的个人轻量化大模型应用实战案例，集成 文档生成、文档解析、会议纪要、私有知识库 RAG 四大办公高频功能。项目采用 Python + Flask 构建后端，可本地私有化部署，适合学习大模型应用开发、AI Agent、RAG 检索增强生成等技术，也可作为求职作品集项目展示。
功能概览
✅ 文档生成助手：支持周报、PRD、方案、短文、公文等模板，一键生成正式文档，支持润色 / 改写 / 缩写。
✅ 文档智能解析：支持 PDF / DOCX / TXT / MD 文件上传，自动提取文本、表格、结构化内容，AI 自动抽取关键信息、核心结论、重要数据。
✅ 智能会议纪要：支持音频上传（模拟语音转写），自动生成会议核心议题、关键结论、待办事项（含负责人），支持导出完整纪要。
✅ 私有知识库 RAG：支持文档入库、向量化、相似度检索、精准问答，所有数据本地存储，隐私可控。
技术栈
后端：Python、Flask、Flask-CORS
AI 能力：DeepSeek LLM API、文本分块、余弦相似度检索
文档处理：PyPDF2、python-docx、Pathlib
向量计算：NumPy、MD5 哈希（轻量 Embedding 替代）
部署：OpenClaw、Shell 脚本、私有化本地部署
项目结构
plaintext
ai-office-assistant/
├── server.py               # Flask后端API服务入口
├── parse_document.py       # PDF/DOCX/TXT文档解析模块
├── rag_engine.py           # RAG私有知识库核心引擎
├── start.sh                 # 一键启动脚本
├── requirements.txt         # 依赖包清单
├── .gitignore               # 忽略上传文件/缓存/密钥
└── README.md                # 项目说明文档
快速部署（本地运行）
1. 安装依赖
bash
运行
pip install flask flask-cors PyPDF2 python-docx numpy
2. 配置密钥
在项目根目录创建 .env 文件：
plaintext
DEEPSEEK_API_KEY=你的DeepSeek密钥
3. 启动服务
bash
运行
bash start.sh
# 或
python server.py
4. 访问地址
前端页面：http://127.0.0.1:5000
API 接口：http://127.0.0.1:5000/api
API 接口说明
/api/status：服务状态检测
/api/doc-generator：文档生成 / 优化
/api/pdf-parser：文档上传解析、关键信息提取
/api/meeting-summary：音频上传、会议纪要生成、导出
/api/rag-qa：知识库文档上传、删除、问答、清空
项目亮点
基于 OpenClaw 开发：完整体现大模型应用开发、技能编排、私有化部署能力。
轻量化 RAG 实现：无重型依赖，纯 Python 实现文本分块、向量检索、相似度匹配，适合学习 RAG 原理。
端到端流程完整：从文件上传 → AI 处理 → 结果输出 → 导出下载，覆盖真实办公场景。
数据本地私有化：所有文件、知识库、向量数据均本地存储，隐私安全可控。
求职友好项目：完整可演示、代码规范、文档清晰，可作为大模型应用、AI Agent、RAG 开发岗位作品集。
适用场景
个人办公自动化、文档处理、知识管理
大模型应用开发学习、RAG 技术入门
AI 产品原型快速验证
求职项目展示（大模型应用 / AI 开发 / 后端开发）
注意事项
代码中 API 密钥已脱敏，使用前需替换为自己的 DeepSeek 密钥。
语音转写功能为模拟实现，正式使用可接入百度 / 阿里云 / 讯飞 ASR API。
知识库向量数据存储在本地 data/vector_store，上传文件会自动入库。
项目基于 OpenClaw 私有部署，支持扩展更多 AI 技能（如邮件生成、图片 OCR 等）。
