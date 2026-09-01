import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def generate_answer(question, context):

    prompt = f"""
You are an AI Resume Assistant.

Answer the user's question using ONLY the information
provided in the resume context.

Resume Context:
{context}

Question:
{question}

Instructions:
- Give a clear and simple answer.
- Do not make up information.
- If the answer is not available in the resume context,
  say: "The information is not available in the resume."
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text