from PDF_Chunk import split_pdf_by_visual_chapter, convert_to_documents, convert_to_chunk
from embedding_VectorDB import get_embedding_model, save_faiss_index, load_faiss_index
from Retriever_llm import ask_llm
import os

# 📄 PDF 路徑
pdf_path = r"C:\Users\medgreen\Desktop\LuBao\RAG\MD-500_500S_Instruction-manual_CV8-F3_N3 0923_final.pdf"

# 📘 PDF ➜ Documents ➜ Chunks
chapters = split_pdf_by_visual_chapter(pdf_path)
documents = convert_to_documents(chapters)
documents = convert_to_chunk(documents, chunk_size=500)
print(documents)

# 🔢 embedding + 儲存
embedding_model = get_embedding_model()
base_dir = os.path.dirname(__file__)
save_path = os.path.join(base_dir, "faiss_nomic_index")
save_faiss_index(documents, embedding_model, save_path)
print("📂 實際儲存路徑：", os.path.abspath(save_path))

# 📦 載入向量庫
vectorstore = load_faiss_index(save_path, embedding_model)

# 🔁 問答迴圈
while True:
    question = input("\n🧠 請輸入你的問題（輸入 q 結束）：\n")
    if question.lower() == "q":
        break
    ask_llm(vectorstore, question)
