import os
import io
import re
import requests
import PIL.Image
import uvicorn
from collections import defaultdict
from fastapi import FastAPI, Request, Header, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Google GenAI SDK
from google import genai
from google.genai import types

# Line Bot
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, ImageMessage,
)

# ======= 修正這裡：加入 init_chat_model 的引入 =======
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
# ===================================================

# ==========================
#  環境設定與工具函式
# ==========================

agnes_api_key = os.environ.get("AGNES_API_KEY")
google_api = os.environ.get("GOOGLE_API_KEY")

line_bot_api = LineBotApi(os.environ.get("CHANNEL_ACCESS_TOKEN"))
line_handler = WebhookHandler(os.environ.get("CHANNEL_SECRET"))

genai_client = genai.Client(api_key=google_api)
# thinking_config = genai.types.ThinkingConfig(thinking_budget=0) # thinking_budget = 0,  turn off thinking mode
generation_config = genai.types.GenerateContentConfig(max_output_tokens=800, temperature=0.2, top_p=0.5)

user_message_history = defaultdict(list)
app = FastAPI()

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_image_url_from_line(message_id):
    try:
        message_content = line_bot_api.get_message_content(message_id)
        file_path = f"static/{message_id}.png"
        with open(file_path, "wb") as f:
            for chunk in message_content.iter_content():
                f.write(chunk)
        return file_path
    except Exception as e:
        print(f"❌ 圖片取得失敗：{e}")
        return None

def store_user_message(user_id, message_type, message_content):
    user_message_history[user_id].append({"type": message_type, "content": message_content})

def get_previous_message(user_id):
    if user_id in user_message_history and len(user_message_history[user_id]) > 0:
        return user_message_history[user_id][-1]
    return {"type": "text", "content": "No message!"}

# ==========================
#  LangChain 工具定義
# ==========================

def generate_and_upload_image(prompt: str, api_key: str = None) -> str:
    """根據文字提示生成圖片並儲存至本地 static 資料夾。"""
    
    # 優先使用傳入的 api_key，若無則使用全域/環境變數設定
    key = agnes_api_key
    if not key or key == "YOUR_API_KEY":
        return "錯誤：未提供有效的 API 金鑰。"

    url = "https://apihub.agnes-ai.com/v1/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    
    # 依據官方文件規範設定 Payload
    data = {
        "model": "agnes-image-2.1-flash",
        "prompt": prompt,
        "size": "1K",         # 推薦檔位：1K, 2K, 3K, 4K
        "ratio": "1:1",       # 支援 1:1, 16:9, 9:16, 4:3, 3:4 等
        "extra_body": {
            "response_format": "url"  # 必須放在 extra_body 內
        }
    }
    
    try:
        # 1. 發送請求至 Agnes AI API (設定 timeout=120)
        response = requests.post(url, json=data, headers=headers, timeout=120)
        
        # 檢查 HTTP 狀態碼
        if response.status_code != 200:
            return f"API 請求失敗 [HTTP {response.status_code}]: {response.text}"
        
        resp_data = response.json()
        
        # 2. 解析 JSON 回應 (路徑: data[0].url)
        data_list = resp_data.get("data", [])
        if not data_list or not data_list[0].get("url"):
            return f"圖片生成失敗：回應中未包含圖片 URL ({resp_data})"
            
        image_url = data_list[0]["url"]
        
        # 3. 下載生成的圖片
        image_download_response = requests.get(image_url, timeout=30)
        if image_download_response.status_code != 200:
            return f"從 URL 下載圖片失敗 [HTTP {image_download_response.status_code}]"
            
        # 4. 確保 static 資料夾存在，並直接儲存二進位圖片檔
        os.makedirs("static", exist_ok=True)
        file_path = f"static/{os.urandom(8).hex()}.png"
        
        with open(file_path, "wb") as f:
            f.write(image_download_response.content)
        
        # 5. 回傳圖片 URL
        base_url = os.getenv("HF_SPACE", "http://localhost:7860").rstrip("/")
        return f"{base_url}/{file_path}"

    except requests.exceptions.Timeout:
        return "錯誤：請求逾時，圖片生成時間過長。"
    except requests.exceptions.RequestException as e:
        return f"網路請求異常: {e}"
    except Exception as e:
        return f"程式執行出錯: {e}"

def analyze_image_with_text(image_path: str, user_text: str) -> str:
    """根據圖片路徑和文字提問進行分析。"""
    try:
        if not os.path.exists(image_path):
            return "錯誤：找不到該圖片檔案。"
        img_user = PIL.Image.open(image_path)

        response = genai_client.models.generate_content(
                        model="gemini-3.6-flash", # gemini-3.6-flash
                        contents=[img_user, user_text],
                        config=generation_config
        )
        
        return response.text if response.text else "Gemini 沒答案！"
    except Exception as e:
        return f"分析出錯: {e}"

# ==========================
#  LangChain 代理人設定
# ==========================
tools = [generate_and_upload_image, analyze_image_with_text]

system_prompt = """
你是一個強大的影像生成及影像分析代理人。如果生成了圖片，請直接給出 imageUrl。
【重要指示】：如果使用者只是單純打招呼、閒聊或沒有提供明確的圖片需求，請直接用文字友善回覆，絕對不要呼叫任何工具。
"""

# 明確指定使用 google_genai，徹底封鎖 Render 誤判成 Vertex AI 的可能
llm = init_chat_model("gemini-3.6-flash", model_provider="google_genai")

agent_executor = create_agent(
    model=llm, # 將實例化後的模型傳入
    tools=tools,
    system_prompt=system_prompt,
)

# ==========================
#  FastAPI 路由
# ==========================
@app.get("/")
def root():
    return {"status": "running"}

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks, x_line_signature=Header(None)):
    body = await request.body()
    try:
        background_tasks.add_task(line_handler.handle, body.decode("utf-8"), x_line_signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400)
    return "ok"

@line_handler.add(MessageEvent, message=(ImageMessage, TextMessage))
def handle_message(event):
    user_id = event.source.user_id
    
    # 1. 處理圖片訊息
    if isinstance(event.message, ImageMessage):
        image_path = get_image_url_from_line(event.message.id)
        if image_path:
            store_user_message(user_id, "image", image_path)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="收到圖片了！請問你想對這張圖做什麼分析？"))
    # 2. 處理文字訊息 (修正縮進，確保它與上面的 if 對齊)
    elif isinstance(event.message, TextMessage):
        user_text = event.message.text
        previous_message = get_previous_message(user_id)
        if previous_message["type"] == "image":
            image_path = previous_message["content"]
            user_text = f"請分析這張圖片 {image_path}，問題是：{user_text}"
            user_message_history[user_id].pop()

        agent_input = {"messages": [{"role": "user", "content": user_text}]} 
        try:
            response = agent_executor.invoke(agent_input)
            messages = response.get("messages", [])
            
            out = ""
            image_url = None

            # 1. 由後往前找，優先找出包含 imageUrl 的訊息
            for msg in reversed(messages):
                content = msg.content
                
                # 如果內容是字典 (工具回傳的格式)
                if isinstance(content, dict) and "imageUrl" in content:
                    image_url = content["imageUrl"]
                    break
                
                # 如果內容是字串，嘗試從中提取網址
                elif isinstance(content, str) and content.strip():
                    # 這裡用我們先前的 Regex 找找看
                    found_urls = re.findall(r'https?://[^\s<>"\)]+|www\.[^\s<>"\)]+', content)
                    img = next((u for u in found_urls if any(ext in u.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif'])), None)
                    if img:
                        image_url = img.split(')')[0]
                        out = content # 保留文字內容
                        break
                    elif not out: # 如果還沒找到圖片，先暫存文字內容
                        out = content

                # 如果內容是列表 (Gemini 2.5 有時會回傳 block list)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            # 找圖片網址
                            text = block.get("text", "")
                            found_urls = re.findall(r'https?://[^\s<>"\)]+|www\.[^\s<>"\)]+', text)
                            img = next((u for u in found_urls if any(ext in u.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif'])), None)
                            if img:
                                image_url = img.split(')')[0]
                            if text:
                                out += text

            # 2. 最終回覆邏輯
            if image_url:
                line_bot_api.reply_message(
                    event.reply_token,
                    [
                        TextSendMessage(text="這是我為你生成的圖片："),
                        ImageSendMessage(original_content_url=image_url, preview_image_url=image_url)
                    ]
                )
            elif out.strip():
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=out.strip()))
            else:
                # 萬一真的什麼都沒有，回報一個預設訊息，避免 LINE 噴 400 錯誤
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="圖片已生成，但我暫時無法取得連結，請稍後再試。"))

        except Exception as e:
            print(f"Agent Error: {e}")
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="抱歉，我現在無法處理這個請求。"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
