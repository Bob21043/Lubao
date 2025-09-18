from transformers import AutoModel, AutoTokenizer
from transformers import GenerationConfig
import torch
from mtkresearch.llm.prompt import MRPromptV3
import json
import requests
#選擇模型
model_id = 'MediaTek-Research/Llama-Breeze2-3B-Instruct-v0_1'
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map='auto',
    img_context_token_id=128212
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

#模型生成設定
generation_config = GenerationConfig(
  max_new_tokens=2048,
  do_sample=True,
  temperature=0.01,
  top_p=0.01,
  repetition_penalty=1.1,
  eos_token_id=128009
)
#組合問句與Function內容給模型的工具
prompt_engine = MRPromptV3()

sys_prompt = 'You are a helpful AI assistant built by MediaTek Research. The user you are helping speaks Traditional Chinese and comes from Taiwan.'

#轉token給模型來生成內容
def _inference(tokenizer, model, generation_config, prompt, pixel_values=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if pixel_values is None:
        output_tensors = model.generate(**inputs, generation_config=generation_config)
    else:
        output_tensors = model.generate(**inputs, generation_config=generation_config, pixel_values=pixel_values.to(model.device, dtype=model.dtype))
    output_str = tokenizer.decode(output_tensors[0])
    return output_str
import json

# 定義可用的工具 (地區溫度與電器電壓)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "取得指定 Location 的目前溫度",
            "parameters": {
                "type": "object",
                "properties": {
                    "Location": {
                        "type": "string",
                        "description": "城市及地區，例如：San Francisco, New York"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["°C", "°F"],
                        "description": "溫度單位，請選擇 °C 或 °F"
                    }
                },
                "required": ["Location"]
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

# call Node-RED 上的溫度 API
def get_temperature_api(Location, unit=None):
    try:

        response = requests.get("http://127.0.0.1:1880/api/temperature")
        response.raise_for_status()  
        return response.json()       # 回傳 JSON 物件
    except Exception as e:
        return {"error": str(e)}

# call Node-RED 上的電器電壓 API
def get_appliance_voltage_api(Appliance):
    try:

        response = requests.get("http://127.0.0.1:1880/api/voltage")
        response.raise_for_status()
        return response.json()      
    except Exception as e:
        return {"error": str(e)}
#映射對應API
mapping = {
    'get_temperature': get_temperature_api,
    'get_appliance_voltage': get_appliance_voltage_api,
}


# stage 1: query轉json
conversations = [
    {"role": "user", "content": "請問台北目前幾度？"},
]
functions = [tool["function"] for tool in tools if tool.get("type") == "function"]

prompt = prompt_engine.get_prompt(conversations, functions=functions)

output_str = _inference(tokenizer, model, generation_config, prompt)
result = prompt_engine.parse_generated_str(output_str)

print("第一階段 結構化內容與call api\n"+str(result))

conversations.append(result)

tool_call = result['tool_calls'][0]
func_name = tool_call['function']['name']
func = mapping[func_name]
arguments = json.loads(tool_call['function']['arguments'])
called_result = func(**arguments)

# stage 2: put executed results
conversations.append(
    {
        'role': 'tool',
        'tool_call_id': tool_call['id'],
        'name': func_name,
        'content': json.dumps(called_result)
    }
)

prompt = prompt_engine.get_prompt(conversations, functions=functions)

output_str2 = _inference(tokenizer, model, generation_config, prompt)
result2 = prompt_engine.parse_generated_str(output_str2)
print("第二階段 問句根據API的respond與問句等回覆內容\n"+str(result2))