
from langchain_community.document_loaders import PyMuPDFLoader
from llama_cpp import Llama
import time
import fitz

# initialize LLM model
def ask_llm_direct(model_path: str, question: str, reference_text: str):

    messages = [
        {"role": "system", "content": "你是的設備的AI專家助手，請用繁體中文回答問題。如果無法回答，請說'我不知道'。"},
        {"role": "user", "content": f"【參考內容】\n{reference_text}\n\n【問題】{question}"}
    ]
    
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=100,
        temperature=0,
        top_p=0.9,
        max_tokens=512,
        repeat_penalty=1.1,
        verbose=False
    )

    start_time = time.time()
    response = llm.create_chat_completion(messages=messages)
    end_time = time.time()

    print("\n✅ 模型直接回答：\n")
    print(response["choices"][0]["message"]["content"])
    print(f"⏱️ 回答時間：{end_time - start_time:.2f} 秒")

# Extract tables from PDF and return as list of lists
def extract_tables_from_pdf_as_arrays(pdf_path, pages=[3, 4]):
    doc = fitz.open(pdf_path)
    table_list = []
    page_list = range(len(doc)) if pages is None else pages

    for i in page_list:
        page = doc[i]
        if hasattr(page, "find_tables"):
            found_tables = page.find_tables()
            for ti, table in enumerate(found_tables):
                try:
                    arr = table.extract()  # ← 這裡直接是 list of lists
                    table_list.append({
                        "page": i+1,
                        "index": ti+1,
                        "array": arr,
                    })
                except Exception as e:
                    table_list.append({
                        "page": i+1,
                        "index": ti+1,
                        "array": [],
                        "error": str(e)
                    })
    return table_list


def tables_to_string(tables):
    result = []
    for table in tables:
        result.append(f"【第{table['page']}頁 表格{table['index']}】")
        for row in table["array"]:
            result.append(" , ".join([str(x) if x is not None else "" for x in row]))
        result.append("")  # 空行斷表格
    return "\n".join(result)




def main():
    pdf_path =r"C:\Users\medgreen\Desktop\LuBao\Lubao_LLM_test\LLM_RAG\AS_129765_SR-X_UM_B95TW_KW_TW_2044_6.pdf."
    # pdf_path = r"C:\Users\medgreen\Desktop\LuBao\MD-500_500S_Instruction-manual_CV8-F3_N3 0923_final.pdf"
    
    # model_path = r"C:\\Users\\medgreen\\.cache\\huggingface\\hub\\models--YorkieOH10--Meta-Llama-3.1-8B-Instruct-hf-Q4_K_M-GGUF\\snapshots\\a1b5ff51c81d0f8eacc89f2460a775a5b3013272\\meta-llama-3.1-8b-instruct-hf-q4_k_m.gguf"
    model_path =r"C:\Users\medgreen\.cache\huggingface\hub\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    
    
    # 擷取 PDF 中的表格
    tables =extract_tables_from_pdf_as_arrays(pdf_path)
    print(f"總共擷取到 {len(tables)} 張表格。")
    for table in tables:
        print(f"\n【第{table['page']}頁 表格{table['index']}】")
        for row in table["array"]:
            print(row)
    
    reference_text = tables_to_string(tables)
       
    print("啟動 LLM 問答模式。")
    while True:
        question = input("\n🧠 請輸入你的問題（輸入 q 結束）：\n")
        if question.lower() == "q":
            print("👋 已結束問答，歡迎隨時再來！")
            break
        ask_llm_direct(model_path,question, reference_text)


if __name__ == "__main__":
    main()
    