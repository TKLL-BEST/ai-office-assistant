## RLHF 反馈系统 - 使用说明

### 什么是 RLHF 反馈系统？

这是 AI Office Assistant 的 RLHF（Reinforcement Learning from Human Feedback）数据收集模块。核心思路：

```
用户使用 AI 助手 -> 对回答进行评分（1-5） -> 反馈数据存储 -> 导出 DPO 偏好数据 -> 模型训练
```

### 新增文件

```
skills/feedback_engine.py    反馈收集引擎（核心）
server.py                    新增 4 个 API 路由：
  POST /api/feedback          提交反馈
  GET  /api/feedback/list     查看反馈列表
  GET  /api/feedback/stats    查看统计
  GET  /api/feedback/dpo/export 导出 DPO 偏好数据
```

### API 使用示例

#### 1. 提交反馈

```bash
curl -X POST http://localhost:5000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请写一份本周工作总结",
    "response": "本周完成了项目A的需求分析和原型设计...",
    "rating": 4,
    "module": "doc-generator",
    "feedback_text": "内容准确但格式可以优化",
    "tags": ["准确", "格式优化"]
  }'
```

#### 2. 查看反馈统计

```bash
curl http://localhost:5000/api/feedback/stats
```

返回：
```json
{
  "stats": {
    "total_feedbacks": 42,
    "avg_rating": 3.8,
    "module_counts": {"doc-generator": 15, "rag-qa": 20, "meeting-summary": 7}
  },
  "dpo_stats": {
    "dpo_pairs": 12,
    "avg_rating_diff": 3.0
  }
}
```

#### 3. 导出 DPO 偏好数据

```bash
curl http://localhost:5000/api/feedback/dpo/export?min_rating_diff=2
```

导出文件格式（JSONL，每行一个偏好对）：
```json
{"prompt": "请写一份本周工作总结", "chosen": "高质量回答...", "rejected": "低质量回答...", "rating_diff": 3}
```

### 和 RLHF 学习项目的关系

```
用户使用 AI Office Assistant
        |
        v
   收集反馈（评分 1-5）
        |
        v
   导出 DPO 偏好数据（点击服务器 API）
        |
        v
   用 rlhf_learning/dpo_demo.py 跑训练 ← 你之前学的内容
```

### 如何在前端添加评分按钮

在你的前端页面中，每次 LLM 生成回答后，添加：

```html
<div class="feedback-bar">
  <button onclick="rateResponse(5)">⭐⭐⭐⭐⭐</button>
  <button onclick="rateResponse(4)">⭐⭐⭐⭐</button>
  <button onclick="rateResponse(3)">⭐⭐⭐</button>
  <button onclick="rateResponse(2)">⭐⭐</button>
  <button onclick="rateResponse(1)">⭐</button>
</div>
```

```javascript
async function rateResponse(rating) {
  const response = await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: currentPrompt,
      response: currentResponse,
      rating: rating,
      module: currentModule
    })
  });
  const result = await response.json();
  console.log('Feedback submitted:', result);
}
```

### DPO 数据导出后的训练

导出数据后，配合你之前学的内容使用：

```bash
## 1. 导出数据
curl http://localhost:5000/api/feedback/dpo/export > dpo_data.jsonl

## 2. 用 TRL 训练（推荐）
## pip install trl transformers accelerate
## 参考: https://github.com/huggingface/trl
```

### 核心概念对应

| 反馈系统 | RLHF 理论 |
|----------|-----------|
| prompt（用户输入） | 状态 s |
| response（AI 回答） | 动作 a（token 序列） |
| rating（1-5 评分） | 奖励 r（来自 Reward Model 或人类） |
| high_rating（>=4） | chosen 回答 |
| low_rating（<=2） | rejected 回答 |
| DPO 导出 | 偏好数据集（直接用于 DPO 训练） |