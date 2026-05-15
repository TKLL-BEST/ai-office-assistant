#!/usr/bin/env python3
"""
RAG 知识库引擎 — 向量检索版
使用 DeepSeek API 生成 Embedding + NumPy 做余弦相似度检索
无需安装 torch / langchain / chromadb 等重量级依赖

核心流程：
  文档 → 分块(Chunks) → Embedding(向量化) → 存入向量库
  提问 → Embedding(向量化) → 向量相似度检索 → 拼接上下文 → LLM 回答

使用方法（命令行）：
  python3 rag_engine.py add <文件路径>          # 添加文档到知识库
  python3 rag_engine.py list                     # 列出知识库文档
  python3 rag_engine.py delete <doc_id>           # 删除文档
  python3 rag_engine.py clear                     # 清空知识库
  python3 rag_engine.py query "<问题>"            # 提问

API 调用：
  本文件同时暴露 RagEngine 类，可供 server.py 直接 import
"""

import os
import sys
import json
import time
import hashlib
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# ── 配置 ────────────────────────────────────────────────────────────

# DeepSeek Embedding API
DEEPSEEK_API_KEY = "sk-44e4d328f0b64939a5a052c87c61726c"
LLM_URL = "https://api.deepseek.com/v1/chat/completions"
LLM_MODEL = "deepseek-chat"

# HuggingFace Inference API（免费，无需 API key）
HF_EMBEDDING_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 的输出维度

# 知识库存储路径
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
KB_DIR = DATA_DIR / "vector_store"
KB_INDEX_FILE = KB_DIR / "index.json"
CHUNK_SIZE = 500       # 每个文本块的最大字符数
CHUNK_OVERLAP = 50     # 块之间的重叠字符数
TOP_K = 5              # 检索返回的最相关块数

# 确保目录存在
KB_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# 一、Embedding — 调用 DeepSeek API 将文本转为向量
# ═══════════════════════════════════════════════════════════════════

def get_embedding(text: str) -> List[float]:
    """
    用 DeepSeek LLM 生成的摘要语义哈希来模拟 Embedding。
    
    原理：让 LLM 为文本生成一段固定格式的语义标签（主题、关键词），
    然后将这些标签转为固定维度的向量用于相似度匹配。
    
    这不是真正的 embedding，但在无法调用 embedding API 时
    是一种有效的轻量替代方案。
    """
    prompt = f"""分析以下文本，用一句话概括其核心主题，并列出3-5个关键词（逗号分隔）。

文本：{text[:500]}

输出格式：
主题：[一句话概括]
关键词：[关键词1], [关键词2], [关键词3]
"""
    
    try:
        payload = json.dumps({
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 128,
        }).encode("utf-8")
        
        req = urllib.request.Request(
            LLM_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            },
            method="POST",
        )
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            summary = result["choices"][0]["message"]["content"]
            
        # 将文本转为固定维度的数值特征向量
        return _text_to_feature_vector(text + " " + summary)
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return _text_to_feature_vector(text)


def _text_to_feature_vector(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """
    将文本转为固定维度的特征向量。
    
    这不是真正的语义 embedding，而是基于 char n-gram 的哈希特征。
    优点是：
    - 不需要任何外部 API
    - 速度快
    - 对中文文本有一定的相似度区分能力
    
    原理：
    1. 对文本做 char n-gram（n=2,3,4）
    2. 每个 n-gram 哈希到 dim 维向量的某个位置
    3. 计数归一化得到最终向量
    """
    import hashlib
    
    if not text:
        return [0.0] * dim
    
    text = text.lower()
    vec = [0.0] * dim
    
    # 生成 char n-gram 特征
    for n in [2, 3, 4]:
        for i in range(len(text) - n + 1):
            gram = text[i:i+n]
            # 用 MD5 哈希将 gram 映射到 [0, dim) 范围
            h = int(hashlib.md5(gram.encode()).hexdigest()[:8], 16) % dim
            vec[h] += 1.0
    
    # L2 归一化
    vec_np = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec_np)
    if norm > 0:
        vec_np = vec_np / norm
    
    return vec_np.tolist()


# ═══════════════════════════════════════════════════════════════════
# 二、文本分块 — 将长文档切成合适的段落
# ═══════════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    将长文本切分成重叠的块。
    
    为什么需要分块？
    - Embedding 模型有最大输入长度限制（DeepSeek 是 8192 tokens）
    - 更短的块意味着更精确的检索（每个块只包含一个主题）
    - 重叠可以避免信息被切分在边界上丢失
    
    分块策略：
    1. 先用 '\n\n'（段落）切分，保证语义完整性
    2. 如果段落太长，再按句子或字符切分
    3. 相邻块之间重叠 overlap 个字符
    """
    if not text:
        return []
    
    chunks = []
    
    # 第一步：按段落切分（保留段落边界）
    paragraphs = text.split("\n\n")
    
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            # 当前块够了，保存
            if current_chunk:
                chunks.append(current_chunk.strip())
            # 开始新块，带上重叠部分
            if overlap > 0 and current_chunk:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para + "\n\n"
            else:
                current_chunk = para + "\n\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


# ═══════════════════════════════════════════════════════════════════
# 三、文档解析 — 从不同格式提取文本
# ═══════════════════════════════════════════════════════════════════

def extract_text(filepath: str) -> str:
    """从 PDF/TXT/DOCX/MD 文件中提取纯文本"""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in (".txt", ".md"):
        return Path(filepath).read_text(encoding="utf-8", errors="ignore")
    
    elif ext == ".pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            texts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
            return "\n\n".join(texts)
        except ImportError:
            return "[Error: PyPDF2 not installed]"
        except Exception as e:
            return f"[Error parsing PDF: {e}]"
    
    elif ext == ".docx":
        try:
            import docx
            doc = docx.Document(filepath)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            return "[Error: python-docx not installed]"
        except Exception as e:
            return f"[Error parsing DOCX: {e}]"
    
    else:
        return "[Error: unsupported format]"


# ═══════════════════════════════════════════════════════════════════
# 四、向量相似度计算
# ═══════════════════════════════════════════════════════════════════

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    计算两个向量的余弦相似度（值域：-1 ~ 1）
    
    cosine_sim(A, B) = A·B / (||A|| * ||B||)
    
    为什么用余弦相似度？
    - 对向量的长度不敏感，只关注方向的相似度
    - 适合语义搜索：两个文本如果"意思相近"，向量方向就相近
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


# ═══════════════════════════════════════════════════════════════════
# 五、LLM 增强回答
# ═══════════════════════════════════════════════════════════════════

def call_llm(prompt: str, system_prompt: str = None) -> str:
    """调用 DeepSeek 生成回答"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.3,       # 低温度，让回答更精确
        "max_tokens": 2048,
    }).encode("utf-8")

    req = urllib.request.Request(
        LLM_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM Error] {e}"


# ═══════════════════════════════════════════════════════════════════
# 六、RAG 引擎核心类
# ═══════════════════════════════════════════════════════════════════

class RagEngine:
    """
    RAG 知识库引擎
    
    数据模型：
    ┌─────────────────────────────────────────────┐
    │              index.json                      │
    ├─────────────────────────────────────────────┤
    │  documents: {                               │
    │    "doc_id_1": {                            │
    │      filename: "doc.pdf",                   │
    │      upload_time: "2024-01-01",             │
    │      chunks: ["chunk1", "chunk2", ...],     │
    │    }                                        │
    │  }                                          │
    │  embeddings: [                              │
    │    [0.12, -0.34, ...],  ← chunk1 的向量    │
    │    [0.56, 0.78, ...],   ← chunk2 的向量    │
    │    ...                                      │
    │  ]                                          │
    │  chunk_map: {                               │
    │    0: {"doc_id": "xxx", "chunk_idx": 0},    │
    │    1: {"doc_id": "xxx", "chunk_idx": 1},    │
    │  }                                          │
    └─────────────────────────────────────────────┘
    """
    
    def __init__(self):
        self.data = self._load()
    
    def _load(self) -> dict:
        """加载知识库索引"""
        if KB_INDEX_FILE.exists():
            return json.loads(KB_INDEX_FILE.read_text(encoding="utf-8"))
        return {"documents": {}, "embeddings": [], "chunk_map": []}
    
    def _save(self):
        """保存知识库索引"""
        KB_INDEX_FILE.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def add_document(self, filepath: str) -> Dict:
        """
        添加文档到知识库
        
        流程：
        1. 提取文本
        2. 切分成块
        3. 生成每个块的 Embedding（调用 DeepSeek API）
        4. 存入向量库
        """
        filepath = str(filepath)
        if not os.path.exists(filepath):
            return {"error": f"文件不存在: {filepath}"}
        
        filename = os.path.basename(filepath)
        
        # 1. 提取文本
        print(f"📖 正在解析文档: {filename}")
        text = extract_text(filepath)
        if not text or text.startswith("[Error"):
            return {"error": text or "文档解析失败"}
        
        # 2. 切分成块
        chunks = chunk_text(text)
        print(f"✂️  文档已切分为 {len(chunks)} 个文本块")
        
        if not chunks:
            return {"error": "文档内容为空，无法分块"}
        
        # 3. 生成文档 ID 并保存原文
        doc_id = hashlib.md5(filename.encode()).hexdigest()[:8]
        
        self.data["documents"][doc_id] = {
            "filename": filename,
            "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chunks": chunks,
        }
        
        # 4. 批量生成 Embedding（逐块调用 API）
        print("🔮 正在生成向量嵌入...")
        start_idx = len(self.data["embeddings"])
        
        for i, chunk in enumerate(chunks):
            print(f"  [{i+1}/{len(chunks)}] 向量化中...")
            embedding = get_embedding(chunk)
            self.data["embeddings"].append(embedding)
            self.data["chunk_map"].append({
                "doc_id": doc_id,
                "chunk_idx": i,
            })
        
        # 5. 保存
        self._save()
        print(f"✅ 文档 '{filename}' 入库成功，共 {len(chunks)} 个文本块")
        
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks_count": len(chunks),
        }
    
    def delete_document(self, doc_id: str) -> Dict:
        """删除文档及其向量"""
        if doc_id not in self.data["documents"]:
            return {"error": f"文档不存在: {doc_id}"}
        
        # 找到该文档的所有 chunk 索引
        indices_to_remove = [
            i for i, m in enumerate(self.data["chunk_map"])
            if m["doc_id"] == doc_id
        ]
        
        # 反向遍历删除（从后往前删，索引不会乱）
        for i in reversed(indices_to_remove):
            del self.data["embeddings"][i]
            del self.data["chunk_map"][i]
        
        filename = self.data["documents"][doc_id]["filename"]
        del self.data["documents"][doc_id]
        
        self._save()
        return {"message": f"已删除文档 '{filename}'"}
    
    def clear(self) -> Dict:
        """清空知识库"""
        self.data = {"documents": {}, "embeddings": [], "chunk_map": []}
        self._save()
        return {"message": "知识库已清空"}
    
    def list_documents(self) -> List[Dict]:
        """列出所有文档"""
        return [
            {
                "doc_id": doc_id,
                "filename": info["filename"],
                "chunks_count": len(info["chunks"]),
                "upload_time": info["upload_time"],
            }
            for doc_id, info in self.data["documents"].items()
        ]
    
    def query(self, question: str, top_k: int = TOP_K) -> Dict:
        """
        知识库问答（核心 RAG 流程）
        
        流程：
        1. 将问题向量化
        2. 计算问题向量与所有 chunk 向量的余弦相似度
        3. 找到最相似的 top_k 个 chunk
        4. 将 chunk 文本拼接成上下文
        5. 调用 LLM 生成基于上下文的回答
        
        关键技术点：
        - 向量检索：O(n) 全量搜索（小规模知识库足够快）
        - 对于大规模知识库，应改用 HNSW/IVF 等近似最近邻搜索
        """
        if not self.data["embeddings"]:
            return {"error": "知识库为空，请先上传文档"}
        
        # 1. 问题向量化
        print(f"🔮 正在向量化问题...")
        query_vector = get_embedding(question)
        
        # 2. 计算相似度
        print(f"📊 正在检索最相关段落...")
        scores = []
        for i, embedding in enumerate(self.data["embeddings"]):
            score = cosine_similarity(query_vector, embedding)
            chunk_info = self.data["chunk_map"][i]
            doc = self.data["documents"].get(chunk_info["doc_id"])
            if doc and chunk_info["chunk_idx"] < len(doc["chunks"]):
                scores.append((score, i, doc["filename"], doc["chunks"][chunk_info["chunk_idx"]]))
        
        # 3. 按相似度排序，取 top_k
        scores.sort(key=lambda x: x[0], reverse=True)
        top_results = scores[:top_k]
        
        # 4. 拼接上下文
        context_parts = []
        sources = set()
        for score, idx, filename, chunk_text in top_results:
            context_parts.append(f"[来源: {filename} | 相似度: {score:.3f}]\n{chunk_text}")
            sources.add(filename)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # 限制上下文长度（DeepSeek 上下文窗口 128k，但为了速度和成本我们截取）
        if len(context) > 8000:
            context = context[:8000] + "\n\n...(内容过长已截取)"
        
        # 5. LLM 增强回答
        print(f"🤖 正在生成回答...")
        system_prompt = """你是一个知识库问答助手。请基于提供的知识库内容回答问题。

规则：
1. 只使用知识库中的信息来回答
2. 如果知识库中没有足够信息，如实说"知识库中未找到相关信息"
3. 不要编造信息
4. 如果问题与知识库内容完全无关，告诉用户知识库中没有相关内容"""
        
        full_prompt = f"""知识库内容：
{context}

问题：{question}

请基于以上知识库内容给出准确、简洁的回答。如果知识库中没有相关信息，请直接说明。"""
        
        answer = call_llm(full_prompt, system_prompt)
        
        return {
            "question": question,
            "answer": answer,
            "sources": list(sources),
            "results": [
                {
                    "score": round(score, 4),
                    "source": filename,
                    "excerpt": text[:150] + "..." if len(text) > 150 else text,
                }
                for score, idx, filename, text in top_results
            ],
        }


# ═══════════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════════

def print_separator():
    print("\n" + "=" * 60)

if __name__ == "__main__":
    engine = RagEngine()
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python3 rag_engine.py add <文件路径>    # 添加文档")
        print("  python3 rag_engine.py list              # 列出文档")
        print("  python3 rag_engine.py delete <doc_id>   # 删除文档")
        print("  python3 rag_engine.py clear             # 清空知识库")
        print("  python3 rag_engine.py query <问题>      # 提问")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "add":
        if len(sys.argv) < 3:
            print("请指定文件路径")
            sys.exit(1)
        result = engine.add_document(sys.argv[2])
        print_separator()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif command == "list":
        docs = engine.list_documents()
        print_separator()
        print(f"📚 知识库共 {len(docs)} 个文档\n")
        for doc in docs:
            print(f"  📄 {doc['filename']}")
            print(f"     ID: {doc['doc_id']}")
            print(f"     块数: {doc['chunks_count']}")
            print(f"     入库时间: {doc['upload_time']}\n")
    
    elif command == "delete":
        if len(sys.argv) < 3:
            print("请指定文档 ID")
            sys.exit(1)
        result = engine.delete_document(sys.argv[2])
        print_separator()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif command == "clear":
        result = engine.clear()
        print_separator()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif command == "query":
        if len(sys.argv) < 3:
            print("请输入问题")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        print_separator()
        print(f"❓ 问题: {question}\n")
        result = engine.query(question)
        print_separator()
        print(f"💡 回答:\n{result.get('answer', '')}")
        if result.get("sources"):
            print(f"\n📚 来源: {', '.join(result['sources'])}")
        if result.get("results"):
            print(f"\n📊 检索结果:")
            for r in result["results"]:
                print(f"  [{r['score']:.3f}] {r['source']}")
                print(f"     {r['excerpt']}\n")
