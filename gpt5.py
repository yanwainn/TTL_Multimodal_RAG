import os
from openai import AzureOpenAI

endpoint = "https://customer-ps-ttlpilot-t93a-aoai-temp.openai.azure.com/"
model_name = "gpt-5"
deployment = "gpt-5"

subscription_key = "<your-api-key>"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_completion_tokens=16384,
    formatting_reenabled=true,
    model=deployment
)

print(response.choices[0].message.content)