from typing import List
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# ✅ GGUF 模型用 llama-cpp 做 embedding（可選用）
from llama_cpp import Llama

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

# ✅ 可自由切換使用哪個 embedding 模型
def get_embedding_model(use_llama: bool = False, llama_model_path: str = "") -> Embeddings:
    if use_llama:
        if not llama_model_path:
            raise ValueError("請提供 GGUF 模型的路徑！")
        return LlamaCppEmbeddings(model_path=llama_model_path)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={"device": device}
        )

def save_faiss_index(documents: List[Document], embedding_model: Embeddings, save_path: str = "faiss_nomic_index"):
    print("📦 建立 FAISS 向量庫中...")
    vectorstore = FAISS.from_documents(documents, embedding=embedding_model)
    print(f"💾 儲存至：{save_path}")
    vectorstore.save_local(save_path)
    print("✅ 儲存完成！")

def load_faiss_index(save_path: str, embedding_model: Embeddings) -> FAISS:
    print(f"📂 載入向量庫：{save_path}")
    return FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)
