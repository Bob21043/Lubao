from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from transformers import pipeline
from typing import Dict, Any
import faiss
import torch
import time

start_time = time.time()  # 在執行前記錄開始時間
# 1️⃣ 加載 PDF 文件
loader = PyMuPDFLoader("MD-500_500S_Instruction-manual_CV8-F3_N3 0923_final.pdf")
documents = loader.load()

# 2️⃣ 文本分塊處理
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # 可根據需要調整
    chunk_overlap=200   # 可根據需要調整
)

texts = text_splitter.split_documents(documents)  # 使用 split_documents 方法

# 3️⃣ 設定嵌入模型（確保它使用 GPU）
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={"device": "cuda"}  # 5[ l讓 HuggingFace 在 GPU 運行
)

# 4️⃣ 先建立 CPU 版本的 FAISS Vector Store
vector_store = FAISS.from_documents(texts, embedding_model)

# 5️⃣ 取得 LangChain 內部的 CPU FAISS Index 並轉換為 GPU
cpu_index = vector_store.index
res = faiss.StandardGpuResources()  # 啟動 GPU 資源
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # 轉換到 GPU
vector_store.index = gpu_index  # 替換 LangChain 內部的 Index 為 GPU 版本

print("✅ FAISS 已成功轉換至 GPU 運行！🚀")

# 釋放 embedding 模型
del embedding_model
del documents
torch.cuda.empty_cache()
print("✅ 已釋放 embedding 模型與文件資源！")

# 6️⃣ 建立 LLM Pipeline（確保 LLM 也在 GPU 運行）
llm_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-3B",
    device=0,
    max_new_tokens=100,
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

short_answer_template = """
以下為使用者問題與檢索內容：

【使用者問題】
{question}

【檢索內容】
{context}

請根據上述資訊進行分析，並提供詳盡且具體的回答。
"""


# short_answer_template = """
# 你現在是齒雕機專業助手，請遵守下列6點原則，並搭配"輔助答案內容"來回答"用戶問題"：
# 1. 僅根據"輔助答案內容"回答用戶問題。
# 2. 找到與"用戶問題最相似的段落"，依此段落回答問題。
# 3. 如有步驟請詳細描述步驟內容。
# 4. 不要添加額外的內容，猜測或任何上下文中未提及的細節。
# 5. 請"完整地回答問題"，不要只回應到一半。
# 6. 請用繁體中文回答。

# 用戶問題: {question}

# 輔助答案內容: {context}

# 答案:
# """


SHORT_ANSWER_PROMPT = PromptTemplate(
    template=short_answer_template,
    input_variables=["context", "question"]
)

# 8️⃣ 建立 LLMChain
llm_chain = LLMChain(llm=llm, prompt=SHORT_ANSWER_PROMPT)

# 9️⃣ 自訂 QA Chain，結合 FAISS 向量庫與 LLM
class SimpleQAChain(Chain):
    vector_store: Any  # 或指定型別，如: FAISS
    llm_chain: LLMChain
    
    @property
    def input_keys(self):
        return ["question"]
    
    @property
    def output_keys(self):
        # ★ 這裡改成同時回傳 "retrieve" 及 "answer"
        return ["retrieve", "answer"]
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        question = inputs["question"]
        
        # 從向量資料庫檢索最相似的 3 條內容
        docs = self.vector_store.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # 將檢索到的內容（context）與問題一起餵給 LLM
        answer = self.llm_chain.run({"context": context, "question": question})
        
        # ★ 回傳兩項內容：一份是檢索到的文本 (retrieve)，一份是 LLM 的最終回應 (answer)
        return {
            "retrieve": context,
            "answer": answer
        }

# 🔟 建立 SimpleQAChain 實例（確保 FAISS GPU 化成功）
qa_chain = SimpleQAChain(vector_store=vector_store, llm_chain=llm_chain)

# 1️⃣1️⃣ 問答過程 - 測試
query = "如何進行PC與牙科加工機的連接？"
result = qa_chain({"question": query})
end_time = time.time()    # 在執行後記錄結束時間
elapsed = end_time - start_time

# 將兩項結果分別印出
print("=== Retrieve (檢索到的內容) ===")
print(result["retrieve"])
print("\n=== AI Answer (模型回答) ===")
print(result["answer"])
print(f"整體運作時間：{elapsed} 秒")
