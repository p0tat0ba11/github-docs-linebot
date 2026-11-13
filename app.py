import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from rag_engine import initialize_rag_pipeline

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

if not LINE_CHANNEL_ACCESS_TOKEN:
    raise ValueError("Missing required environment variable: LINE_CHANNEL_ACCESS_TOKEN")
if not LINE_CHANNEL_SECRET:
    raise ValueError("Missing required environment variable: LINE_CHANNEL_SECRET")

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    rag_chain = initialize_rag_pipeline()
    yield rag_chain

app = FastAPI(lifespan=lifespan)

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()
    try:
        handler.handle(body.decode("utf-8"), signature)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return "OK"

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    user_question = event.message.text
    try:
        answer = rag_chain.invoke(user_question)
    except Exception as e:
        answer = f"Error processing your question: {str(e)}"

    if len(answer) > 5000:
        answer = answer[:4997] + "..."

    with ApiClient(configuration) as client:
        line_bot_api = MessagingApi(client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=answer)]
            )
        )