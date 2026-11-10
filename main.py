import os
from dotenv import load_dotenv

from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    PushMessageRequest,
    TextMessage,
)

# Load environment variables
load_dotenv()

LINE_CHANNEL_ACCESS_TOKEN = os.getenv(
    "LINE_CHANNEL_ACCESS_TOKEN"
)
LINE_USER_ID = os.getenv("LINE_USER_ID")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
    raise ValueError("Missing required environment variables.")

# Configure LINE Messaging API
configuration = Configuration(
    access_token=LINE_CHANNEL_ACCESS_TOKEN
)

# Send a test message
with ApiClient(configuration) as api_client:
    line_bot_api = MessagingApi(api_client)

    message = TextMessage(
        text="Hello, this is a test message from the LINE Bot!"
    )

    request = PushMessageRequest(
        to=LINE_USER_ID,
        messages=[message]
    )

    line_bot_api.push_message(request)
    