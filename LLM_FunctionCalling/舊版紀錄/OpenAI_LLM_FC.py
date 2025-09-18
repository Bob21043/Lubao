from openai import OpenAI
import requests
import json

client = OpenAI(api_key ="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")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "取得指定 StationName 的目前溫度",
            "parameters": {
                "type": "object",
                "properties": {
                    "StationName": {
                        "type": "string",
                        "description": "城市及地區，例如：San Francisco, New York"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["°C", "°F"],
                        "description": "溫度單位，請選擇 °C 或 °F"
                    }
                },
                "required": ["StationName"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_appliance_voltage",
            "description": "取得指定電器（如吹風機、冰箱）的電壓資訊",
            "parameters": {
                "type": "object",
                "properties": {
                    "Appliance": {
                        "type": "string",
                        "description": "電器名稱，例如：吹風機、冰箱"
                    }
                },
                "required": ["Appliance"]
            }
        }
    }
]

# call 溫度API
def get_temperature_api(StationName, unit=None):
    try:
        params = {"StationName": StationName}
        if unit is not None:
            params["unit"] = unit
        response = requests.get("http://127.0.0.1:1880/api/temperature", params=params)
        response.raise_for_status()
        return response.json()  # 回傳 JSON 物件，如 {"temperature": 25}
    except Exception as e:
        return {"error": str(e)}
# call 電壓API
def get_appliance_voltage_api(Appliance):
    try:
        response = requests.get("http://127.0.0.1:1880/api/voltage", params={"Appliance": Appliance})
        response.raise_for_status()
        return response.json()  # 回傳 JSON 物件，如 {"voltage": "220V"}
    except Exception as e:
        return {"error": str(e)}
mapping = {
    'get_temperature': get_temperature_api,
    'get_appliance_voltage': get_appliance_voltage_api,
}
messages = [
    {"role": "user", "content": "請問收音機的電壓是多少?"}
]

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    tools=tools
)
# 假設返回的訊息中包含 tool_calls (根據官方文件，此格式可能有差異)
llm_response = completion.choices[0].message
print("第一階段LLM 返回的結構化工具呼叫結果:")
print(llm_response)

# 檢查是否有 tool_calls 欄位
if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
    # 這邊假設 tool_calls 是一個 JSON 字串，請根據實際返回格式做解析
    tool_call = llm_response.tool_calls[0]
    func_name = tool_call.function.name
    # 解析參數
    arguments = json.loads(tool_call.function.arguments)
    
    # 呼叫對應的 API 並取得結果
    api_result = mapping[func_name](**arguments)
    
    # 將 API 回應包裝成一則 tool 消息，並加入對話中
    messages.append(llm_response)  # 將原本 LLM 返回的工具呼叫訊息加入對話
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": func_name,
        "content": json.dumps(api_result)
    })
    
    # 第二階段：再次呼叫 OpenAI API，將工具執行結果帶回，生成最終回覆
    final_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools
    )
    
    final_response = final_completion.choices[0].message.content
    print("第二階段LLM回覆內容:")
    print(final_response)
else:
    print("沒有檢測到工具呼叫訊息。")