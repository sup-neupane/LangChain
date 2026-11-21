from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

result = model.invoke("How many Harry Potter Books are there?")

print(result.content[0]["text"])