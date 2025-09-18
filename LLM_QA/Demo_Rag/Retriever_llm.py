from llama_cpp import Llama
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from typing import List
import time

# 🧠 自定義嵌入類別
class LlamaCppEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=512,
            n_gpu_layers=100,
            use_mlock=True,
            verbose=False
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.llm.embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.llm.embed(text)

# 📚 問答主函式

def ask_llm(vectorstore, question: str):
    # 🔍 檢索向量庫
    docs = vectorstore.similarity_search(question, k=3)

    context_text = ""
    print("\n🔍 檢索結果 ：")
    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", f"Chunk{i}")
        chunk_id = doc.metadata.get("chunk_id", "-")
        print(f"【Top{i+1}】章節：{title} | Chunk：{chunk_id}")
        print(doc.page_content[:300].strip() + "...\n")
        context_text += f"【Top{i+1}：{title}】\n{doc.page_content.strip()}\n\n"

    if len(context_text) > 1500:
        context_text = context_text[:1500] + "\n（已截斷部分內容）"

    # 🤖 初始化 LLM 模型（llama.cpp）
    llm = Llama(
        model_path=r"C:\\Users\\medgreen\\.cache\\huggingface\\hub\\models--YorkieOH10--Meta-Llama-3.1-8B-Instruct-hf-Q4_K_M-GGUF\\snapshots\\a1b5ff51c81d0f8eacc89f2460a775a5b3013272\\meta-llama-3.1-8b-instruct-hf-q4_k_m.gguf",
        n_ctx=2048,
        n_gpu_layers=100,
        temperature=0.1,
        top_p=0.9,
        max_tokens=512,
        repeat_penalty=1.1,
        verbose=False
    )

    # ⏱️ 啟動模型回答
    start_time = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "你是牙科加工機的專家助手，請用繁體中文回答問題，請僅根據提供的【參考內容】回答【問題】，不要使用外部知識。使用條列式的方式解決使用者的問題。如果超出文件內容，一律回答'我不知道。"},
            # {"role": "system", "content": "你是牙科加工機的專家助手，請用繁體中文回答問題，並儘量不要重複每段文字原文。如果超出文件內容，一律回答'我不知道'"},
            {"role": "user", "content": f"請根據下列資料回答：\n\n【問題】{question}\n\n【參考內容】\n{context_text}"}
        ]
    )
    end_time = time.time()

    # 🖨️ 顯示回應
    print("\n✅ 模型回答：\n")
    print(response["choices"][0]["message"]["content"])
    print(f"⏱️ 回答時間：{end_time - start_time:.2f} 秒")