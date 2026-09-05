import requests

BOT_TOKEN = ""
CHAT_ID = ""

message = (
    "🚨 TEST ALERT 🚨\n\n"
    "CrowdGuard Alert Bot\n\n"
    "✅ Telegram notification system is working!\n\n"
    "This is a test message from the "
    "AI-Based Stampede Early Warning System."
)

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

data = {
    "chat_id": CHAT_ID,
    "text": message
}

print("Sending test message...")

try:

    response = requests.post(
        url,
        data=data,
        timeout=30
    )

    print("HTTP Status:", response.status_code)
    print("Telegram Response:")
    print(response.text)

except Exception as e:

    print("ERROR:")
    print(e)
