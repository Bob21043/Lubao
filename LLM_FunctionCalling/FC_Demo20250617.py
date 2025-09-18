from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from pydantic import BaseModel, Field ,ValidationError
import requests
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
import json
from langchain_core.runnables import Runnable, RunnableConfig
from opencc import OpenCC
import re


# === 建立轉換器（簡體轉繁體） === 

cc = OpenCC('s2t')  # s2t: Simplified to Traditional

def convert_to_traditional_chinese(text: str) -> str:
    return cc.convert(text)

# === 設定模型 ===
model = "qwen3:8b"
# model = "llama3-groq-tool-use:8b"
llm = ChatOllama(
    model=model,
    base_url="http://localhost:11434",
    temperature=0.1,
    verbose=False
)
# ===工具設定 ===
class CreateUserInput(BaseModel):
    
    account: str = Field(..., description="使用者帳號")
    password: str = Field(..., description="使用者密碼")
    user_name: str = Field(..., description="使用者名稱")

@tool("create_llm_user", args_schema=CreateUserInput)
def create_llm_user(account: str, password: str, user_name: str) -> str:
    """向 LLM 測試平台建立新使用者帳號"""
    try:
        req = {"account": account,"psw": password,"user_name": user_name}
        # print("🧪 JSON字串 : ", json.dumps(req, ensure_ascii=False))

        response = requests.post(
            "https://192.168.100.5/api/llmtesting/user/create",
            json=req,
            timeout=5,
            verify=False
        )
        data = response.json()
        if response.status_code == 200 and data.get("message") == "ok":
            return f"✅ 使用者建立成功,user_id: {data.get('user_id')}"
        else:
            return f"❌ 使用者建立失敗,訊息: {data}"
    except Exception as e:
        return f"🚨 Exception: 呼叫建立帳號 API 時發生錯誤 : {str(e)}"


class CheckUserInput(BaseModel):
    user_id: str = Field(..., description="使用者的 UUID",examples="例如123456-15618-15615-5s84fs18sf等等")


@tool("check_llm_user", args_schema=CheckUserInput)
def check_llm_user(user_id: str) -> str:
    """查詢是否有此使用者/帳戶"""
    try:
        req = {"user_id": user_id}
        response = requests.post(
            "https://192.168.100.5/api/llmtesting/user/check",
            json=req,
            timeout=5,
            verify=False
        )
        data = response.json()
        if response.status_code == 200:
            exists = data.get("exists", False)
            return f"🔎 查詢成功,使用者 {'存在' if exists else '不存在'},user_id: {data.get('user_id')}"
        else:
            return f"❌ 查詢失敗,訊息: {data}"
    except Exception as e:
        return f"🚨 Exception: 呼叫查詢帳號 API 時發生錯誤 : {str(e)}"
    

class ListMachineInput(BaseModel):
    #設定default=""，可以直接查詢所有機台
    keyword: str = Field(default="", description="查詢機台的關鍵字")

@tool("list_machines", args_schema=ListMachineInput)
def list_machines(keyword: str = "") -> str:
    """查詢機台清單，可依據關鍵字搜尋或返回全部"""
    try:
        req = {"keyword": keyword or None}
        print("🧪 JSON字串 : ", json.dumps(req, ensure_ascii=False))

        response = requests.post(
            "https://192.168.100.5/api/llmtesting/machine/list",
            json=req,
            timeout=5,
            verify=False
        )
        data = response.json()

        if response.status_code == 200 and isinstance(data, list):
            if not data:
                return "🔍 查無任何符合的機台資料。"
            # ✅ 直接回傳 JSON 字串格式給使用者看
            return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            return f"❌ 查詢失敗,回傳資料 : {data}"
    except Exception as e:
        return f"🚨 呼叫查詢機台列表 API 時發生錯誤 : {str(e)}"

# === 工具清單 ===
tools = [create_llm_user, check_llm_user, list_machines]
# ===Test topic 2 : Prompt設定 ===
prompt = ChatPromptTemplate.from_messages([

    SystemMessagePromptTemplate.from_template("""
        你是一位專業且理性的助理,請**只用繁體中文**回答使用者問題，即使資料內容是英文，也請以中文回答並解釋。
        請根據問題選擇正確工具,並執行 API 查詢,最後根據回傳結果給出清楚推論與建議。                                      
        ##工作流程（請逐步推理） : 
        1. **理解問題意圖** : 
        - 根據使用者的提問,判斷使用哪一個工具
        - 若無法判斷,請主動詢問使用者以釐清需求
        2. **確認關鍵參數是否足夠** :
        - 確認問題是否包含必要資訊                                  
        - 若資訊不足,主動告知使用者，使用工具需要的參數。
        3. **執行後的合理性驗證與推論** :
        - 檢查回傳結果是否合理（格式正確、數值合理、非空、無錯誤）
        - 若工具回傳已經是明確回覆（如清單、數值、文字說明），請直接輸出工具的內容
        - 如果結果不合理則回覆使用者，根據結果回覆使用者，可能導致錯誤的原因
        - 根據結果合理則進行推論與解釋，請以自然語言清楚說明即可，不需要額外區塊或標籤。
        4. **缺少必要參數時的處理方式** :
        - 所有工具呼叫前，必須確認所有必要參數已明確提供。禁止自行猜測或生成任何參數。
        - 若工具回傳「缺少必要參數」的提示
        - 收到此類提示,請回應使用者 : 
            - 明確指出缺少的參數
            - 說明為何這些參數必要
            - 協助使用者理解如何提供這些資訊（可以用問句方式幫助引導）
        ## 注意事項 :                                      
        禁止自己給參數回答,務必使用工具完成任務後再回覆
        請依據工具回傳的內容，完整提供相關資訊；若使用者有提出具體需求，則須針對該需求給出詳盡回覆。
        請務必以【繁體中文】完成最終回覆（包含工具內容、說明文字、結論）。                                                         
        """
        ),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("使用者的問題是 : {input}"),
    MessagesPlaceholder("agent_scratchpad")
])
# === 建立記憶 ===
memory = ConversationBufferWindowMemory(k=3 ,memory_key="chat_history", return_messages=True)

# === 建立 AgentExecutor ===
agent = create_tool_calling_agent(llm, tools, prompt)
search_agent_executor = AgentExecutor(agent=agent, tools=tools,memory= memory, verbose=False, handle_parsing_errors=True)


# === 回覆用LLM ===
llm_polisher = ChatOllama(
    model="qwen3:8b",  # 也可以用別的模型
    base_url="http://localhost:11434",
    temperature=0.3,
    verbose=False
)

def rephrase_with_llm(text: str) -> str:
    prompt = f"""
你是一位專業繁體中文助理，負責將來自系統的技術或工具輸出內容，轉換為自然、清楚、易懂的繁體中文回答。

請依據以下原始內容進行處理，注意以下目標：

1. 將句子轉換為**自然的台灣用語繁體中文**，避免出現簡體、翻譯腔或機器味。
2. 保留原始回應中的資訊重點（如成功/失敗、錯誤原因、ID 編號等），但調整語序與句法使其更貼近人類語言風格。
3. 若回應內容過於簡略、模糊，請幫忙補上清楚的語意與合理說明（例如：「請再提供相關資訊」提示使用者）。
4. 禁止直接逐字翻譯或僅改變字體，務必進行語句整理與本地化重構。

原始輸出：
{text}

請直接回覆繁體中文結果，不需額外說明。
"""
    response = llm_polisher.invoke(prompt)
    return response.content.strip()
# === Agent think去除 ===
def strip_think_block(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# === 釋放模型 ===
def stop_ollama_model(model_name="qwen3:8b", host="http://localhost:11434"):
    try:
        url = f"{host}/api/stop"
        payload = {"name": model_name}
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            print(f"✅ 模型已停止並釋放資源：{model_name}")
        else:
            print(f"⚠️ 停止失敗（{response.status_code}）: {response.text}")
    except Exception as e:
        print(f"❌ 停止模型時發生錯誤：{e}")


# ✅ 封裝成函式給外部 Agent 使用

def run_env_agent(question: str):
    response = search_agent_executor.invoke({"input": question})
    raw_output = response["output"]
    refined_output = rephrase_with_llm(raw_output)
    final_output  = convert_to_traditional_chinese(refined_output)
    response["output"] = strip_think_block(final_output)
    return response

# 自動測試函式
def run_auto_test():
    #連續問問題
    test_questions = [
        
    ]

    for i, question in enumerate(test_questions, start=1):
        print(f"\n🔍 測試問題 {i}: {question}")
        try:
            response = run_env_agent(question)
            print(f"🤖 回覆內容 : {response['output']}")
        except Exception as e:
            print(f"⚠️ 錯誤 : {str(e)}")

        print(f"🧠 記憶長度 : {len(memory.chat_memory.messages)}")






# 獨立執行測試
if __name__ == "__main__":
    print("🔁 選擇模式 : ")
    print("1️⃣ 手動對話模式")
    print("2️⃣ 自動測試模式")
    mode = input("請輸入模式編號（1 或 2） : ").strip()

    if mode == "1":
        print("\n🤖 助理已就緒！您可以開始提問。")
        try:
            while True:
                question = input("\n📝 請輸入你的問題（輸入 exit 離開） : ")
                if question.strip().lower() == "exit":
                    break

                try:
                    response = run_env_agent(question)
                    print("\n🤖 回覆內容 : ", response["output"])
                except Exception as e:
                    print(f"\n🛑 發生錯誤：{type(e).__name__}: {e}")
                    
                    # 詢問使用者是否需要 AI 幫忙解釋錯誤
                    choice = input("\n需要幫你分析這個錯誤並引導你修正嗎？（y/n）: ").strip().lower()
                    if choice == "y":
                        response = search_agent_executor.invoke({
                            "input": f"以下是錯誤訊息，請協助說明原因並引導修正：\n{str(e)}"
                        })
                        print("\n助理回應 : ", response["output"])
                    else:
                        print("已跳過 AI 分析。請檢查程式輸入或邏輯。")
        except KeyboardInterrupt:
            print("\n 使用者中斷程式。")    
        finally:
            print("\n 程式已結束。")


    elif mode == "2":
        run_auto_test()
        print("\n✅ 自動測試完成。")
    else:
        print("❌ 請輸入有效的選項（1 或 2）")

    
    
    
    
    
  
