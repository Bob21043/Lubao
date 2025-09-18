
from langchain_community.document_loaders import PyMuPDFLoader
from llama_cpp import Llama
import time
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain


def ask_llm_direct(model_path: str, question: str, api_data: str, memory=None):
    
    # Initial memory for conversation
    if memory is None:
        memory = ConversationBufferWindowMemory(k=3, return_messages=True)
    
    messages = [
        {"role": "system", "content": "你是的設備的AI專家助手，請用繁體中文回答問題。如果無法回答，請說'我不知道'。"},
    ]
    
    #add history messages to the conversation
    print("--- 記憶內容（更新前）---")
    print(memory.chat_memory.messages)
    for msg in memory.chat_memory.messages:
        role = "user" if msg.type == "human" else "assistant"
        messages.append({"role": role, "content": msg.content})
    
    # Append the current question and api_data to the messages
    prompt_with_context = f"【參考內容】\n{api_data}\n\n【問題】{question}\n\n請根據以上參考內容和之前的對話歷史，精準地回答這個問題。"
    messages.append({"role": "user", "content": prompt_with_context})
    
    #initial LLM model
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,  
        temperature=0.1,
        top_p=0.9,
        repeat_penalty=1.1,
        verbose=False
    )
    
    start_time = time.time()
    response = llm.create_chat_completion(messages=messages)
    end_time = time.time()
    
    # Add the current question and response to the memory
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(response["choices"][0]["message"]["content"])
    
    answer = response["choices"][0]["message"]["content"]
    # print the response
    print("\n✅ 模型直接回答：\n")
    print(answer)
    print(f"⏱️ 回答時間：{end_time - start_time:.2f} 秒")
    return answer, memory



def main():
    # model_path = r"C:\\Users\\medgreen\\.cache\\huggingface\\hub\\models--YorkieOH10--Meta-Llama-3.1-8B-Instruct-hf-Q4_K_M-GGUF\\snapshots\\a1b5ff51c81d0f8eacc89f2460a775a5b3013272\\meta-llama-3.1-8b-instruct-hf-q4_k_m.gguf"
    model_path =r"C:\Users\medgreen\.cache\huggingface\hub\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    api_data ="""
[
    {
        "營運服務據點": "台灣連達林口總公司",
        "作業者": "白哲瑋",
        "機種擔當": "DI",
        "人員tag": "(人員)台灣",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 108,
                "月交通時數": 0
            },
            {
                "月份": "2025/02",
                "月工作時數": 151.5,
                "月交通時數": 0
            },
            {
                "月份": "2025/03",
                "月工作時數": 216,
                "月交通時數": 0
            }
        ]
    },
    {
        "營運服務據點": "台灣連達林口總公司",
        "作業者": "龔聖駿",
        "機種擔當": "DI",
        "人員tag": "(人員)台灣",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 158.5,
                "月交通時數": 0
            },
            {
                "月份": "2025/02",
                "月工作時數": 134.5,
                "月交通時數": 0
            },
            {
                "月份": "2025/03",
                "月工作時數": 178,
                "月交通時數": 0
            }
        ]
    },
    {
        "營運服務據點": "台灣連達林口總公司",
        "作業者": "陳彥寧",
        "機種擔當": "DI",
        "人員tag": "(人員)台灣",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 119,
                "月交通時數": 0
            },
            {
                "月份": "2025/02",
                "月工作時數": 117,
                "月交通時數": 0
            },
            {
                "月份": "2025/03",
                "月工作時數": 175,
                "月交通時數": 0
            }
        ]
    },
    {
        "營運服務據點": "台灣連達林口總公司",
        "作業者": "余品陞",
        "機種擔當": "DI",
        "人員tag": "(人員)台灣",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 73.5,
                "月交通時數": 0
            },
            {
                "月份": "2025/02",
                "月工作時數": 140.5,
                "月交通時數": 0
            },
            {
                "月份": "2025/03",
                "月工作時數": 144,
                "月交通時數": 0
            }
        ]
    },
    {
        "營運服務據點": "台灣連達林口總公司",
        "作業者": "許文豪",
        "機種擔當": "DI",
        "人員tag": "(人員)台灣",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 122.5,
                "月交通時數": 0
            },
            {
                "月份": "2025/02",
                "月工作時數": 72,
                "月交通時數": 0
            },
            {
                "月份": "2025/03",
                "月工作時數": 134,
                "月交通時數": 0
            }
        ]
    },
    {
        "營運服務據點": "台灣連達林口總公司",
        "作業者": "陳薈羽",
        "機種擔當": "DI",
        "人員tag": "(人員)台灣",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 159.5,
                "月交通時數": 0
            },
            {
                "月份": "2025/02",
                "月工作時數": 100,
                "月交通時數": 0
            },
            {
                "月份": "2025/03",
                "月工作時數": 125.5,
                "月交通時數": 0
            }
        ]
    },
    {
        "營運服務據點": "台灣連達林口總公司",
        "作業者": "曾景彥",
        "機種擔當": "DI",
        "人員tag": "(人員)台灣",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 126.5,
                "月交通時數": 0
            },
            {
                "月份": "2025/02",
                "月工作時數": 122,
                "月交通時數": 0
            },
            {
                "月份": "2025/03",
                "月工作時數": 123,
                "月交通時數": 0
            }
        ]
    },
    {
        "營運服務據點": "深圳連群",
        "作業者": "刘玲",
        "機種擔當": "DI",
        "人員tag": "(人員)華南",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 23.5,
                "月交通時數": 0
            },
            {
                "月份": "2025/02",
                "月工作時數": 100.5,
                "月交通時數": 0
            },
            {
                "月份": "2025/03",
                "月工作時數": 146.5,
                "月交通時數": 26
            }
        ]
    },
    {
        "營運服務據點": "深圳連群",
        "作業者": "李康东",
        "機種擔當": "DI",
        "人員tag": "(人員)華南",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 49.5,
                "月交通時數": 0
            },
            {
                "月份": "2025/02",
                "月工作時數": 64.1,
                "月交通時數": 0
            },
            {
                "月份": "2025/03",
                "月工作時數": 145.8,
                "月交通時數": 0
            }
        ]
    },
    {
        "營運服務據點": "深圳連群",
        "作業者": "梁家铭",
        "機種擔當": "DI",
        "人員tag": "(人員)華南",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 100,
                "月交通時數": 0
            },
            {
                "月份": "2025/02",
                "月工作時數": 105,
                "月交通時數": 0
            },
            {
                "月份": "2025/03",
                "月工作時數": 138,
                "月交通時數": 7
            }
        ]
    },
    {
        "營運服務據點": "深圳連群",
        "作業者": "黄焕贤",
        "機種擔當": "DI",
        "人員tag": "(人員)華南",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 54,
                "月交通時數": 0
            },
            {
                "月份": "2025/02",
                "月工作時數": 118,
                "月交通時數": 0
            },
            {
                "月份": "2025/03",
                "月工作時數": 138,
                "月交通時數": 13
            }
        ]
    },
    {
        "營運服務據點": "深圳連群",
        "作業者": "宋濂东",
        "機種擔當": "DI",
        "人員tag": "(人員)華南",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 122.9,
                "月交通時數": 11
            },
            {
                "月份": "2025/02",
                "月工作時數": 110,
                "月交通時數": 16
            },
            {
                "月份": "2025/03",
                "月工作時數": 135.5,
                "月交通時數": 19
            }
        ]
    },
    {
        "營運服務據點": "深圳連群",
        "作業者": "马梓维",
        "機種擔當": "DI",
        "人員tag": "(人員)華南",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 36,
                "月交通時數": 0
            },
            {
                "月份": "2025/02",
                "月工作時數": 38,
                "月交通時數": 0
            },
            {
                "月份": "2025/03",
                "月工作時數": 128.3,
                "月交通時數": 0
            }
        ]
    },
    {
        "營運服務據點": "深圳連群",
        "作業者": "陈聪",
        "機種擔當": "DI",
        "人員tag": "(人員)華南",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 151.4,
                "月交通時數": 6
            },
            {
                "月份": "2025/02",
                "月工作時數": 132,
                "月交通時數": 7
            },
            {
                "月份": "2025/03",
                "月工作時數": 122,
                "月交通時數": 5
            }
        ]
    },
    {
        "營運服務據點": "深圳連群",
        "作業者": "吴俊兴",
        "機種擔當": "DI",
        "人員tag": "(人員)華南",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 68.9,
                "月交通時數": 6
            },
            {
                "月份": "2025/02",
                "月工作時數": 104,
                "月交通時數": 10
            },
            {
                "月份": "2025/03",
                "月工作時數": 114.5,
                "月交通時數": 14
            }
        ]
    },
    {
        "營運服務據點": "深圳連群",
        "作業者": "李涛",
        "機種擔當": "DI",
        "人員tag": "(人員)華南",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 53,
                "月交通時數": 0
            },
            {
                "月份": "2025/02",
                "月工作時數": 61,
                "月交通時數": 0
            },
            {
                "月份": "2025/03",
                "月工作時數": 110.5,
                "月交通時數": 1
            }
        ]
    },
    {
        "營運服務據點": "深圳連群",
        "作業者": "卢桂荣",
        "機種擔當": "DI",
        "人員tag": "(人員)華南",
        "作業時數": [
            {
                "月份": "2025/01",
                "月工作時數": 128.8,
                "月交通時數": 10
            },
            {
                "月份": "2025/02",
                "月工作時數": 70.5,
                "月交通時數": 5
            },
            {
                "月份": "2025/03",
                "月工作時數": 102.5,
                "月交通時數": 12
            }
        ]
    }
]
    """
    
    # Initialize memory for conversation
    memory = None
    
    # print("啟動 LLM 問答模式。")
    # while True:
    #     question = input("\n🧠 請輸入你的問題（輸入 q 結束）：\n")
    #     if question.lower() == "q":
    #         print("👋 已結束問答，歡迎隨時再來！")
    #         break
    #     ask_llm_direct(model_path,question, api_data,memory)
    print("啟動 LLM 問答模式。")
    print("💡 輸入 'memory' 可查看記憶內容，輸入 'clear' 可清除記憶")
    
    while True:
        question = input("\n🧠 請輸入你的問題（輸入 q 結束）：\n")
        if question.lower() == "q":
            print("👋 已結束問答，歡迎隨時再來！")
            break
        elif question.lower() == "memory":
            # 查看記憶內容
            if memory is None or len(memory.chat_memory.messages) == 0:
                print("📝 目前沒有記憶內容")
            else:
                print("📚 目前記憶內容：")
                for i, msg in enumerate(memory.chat_memory.messages):
                    role = "👤 用戶" if msg.type == "human" else "🤖 AI"
                    print(f"{i+1}. {role}: {msg.content[:100]}...")
        elif question.lower() == "clear":
            # 清除記憶
            if memory:
                memory.clear()
            memory = None
            print("🔄 記憶已清除")
        else:
            _, memory = ask_llm_direct(model_path, question, api_data, memory)

if __name__ == "__main__":
    main()
    
    
