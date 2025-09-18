
from langchain_community.document_loaders import PyMuPDFLoader
from llama_cpp import Llama
import time

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

def main():
    pdf_path =r"C:\Users\medgreen\Desktop\LuBao\Lubao_LLM_test\LLM_RAG\AS_129765_SR-X_UM_B95TW_KW_TW_2044_6.pdf."
    # pdf_path = r"C:\Users\medgreen\Desktop\LuBao\MD-500_500S_Instruction-manual_CV8-F3_N3 0923_final.pdf"
    
    # model_path = r"C:\\Users\\medgreen\\.cache\\huggingface\\hub\\models--YorkieOH10--Meta-Llama-3.1-8B-Instruct-hf-Q4_K_M-GGUF\\snapshots\\a1b5ff51c81d0f8eacc89f2460a775a5b3013272\\meta-llama-3.1-8b-instruct-hf-q4_k_m.gguf"
    model_path =r"C:\Users\medgreen\.cache\huggingface\hub\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    # loader = PyMuPDFLoader(pdf_path,mode="page",extract_tables="markdown")
    # docs = loader.load()
    # print(docs[4].page_content)    
    # reference_text =docs[4].page_content

    reference_text ="""
    |錯誤NO. / 訊息|Col2|LED狀態 （SR&amp;#45;X300/X300W）|Col4|LED狀態 （SR&amp;#45;X100/X100W）|Col6|
|---|---|---|---|---|---|
|E0|FILE SYSTEM|黃色閃爍|■□|黃紅色閃爍|■■|
|E1|FACTPARAM|||黃紅色閃爍|■■|
|E2|CHECK SUM|||黃紅色閃爍|■■|
|E2|CONFIG VER|||黃紅色閃爍|■■|
|E3|PROFINET|||黃色閃爍|■□|
|E4|BUFFER OVER|||黃色閃爍|■□|
|E5|IP DUPLICATE|||黃色閃爍|■□|
|E6|FW UPDATE|||黃紅色閃爍|■■|
|E7|PLC LINK|||黃色閃爍|■□|
|E8|SCRIPT|||黃色閃爍|■□|
|E9|DSP PROG|||黃紅色閃爍|■■|
|E10|CMOS|||黃紅色閃爍|■■|
|E11|AUTO FOCUS|||黃紅色閃爍|■■|
|E12|HOST CONNECT|||黃色閃爍|■□|
|E13|MOTOR|||黃紅色閃爍|■■|
|E13|MOTOR|||黃紅色閃爍|■■|
|E15|REPLACE FILE|||黃色閃爍|■□|
|E99|MISC|||黃紅色閃爍|■■|
|E15|REPLACE FILE|||黃色閃爍|■□|
|E99|MISC|||黃紅色閃爍|■■|
    """
       
    print("啟動 LLM 問答模式。")
    while True:
        question = input("\n🧠 請輸入你的問題（輸入 q 結束）：\n")
        if question.lower() == "q":
            print("👋 已結束問答，歡迎隨時再來！")
            break
        ask_llm_direct(model_path,question, reference_text)


if __name__ == "__main__":
    main()
    