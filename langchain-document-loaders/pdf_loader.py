from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader('questions.pdf')

docs = loader.load()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

prompt = PromptTemplate(
    template="Give me answer to all theser questions  - \n {question}",
    input_variables=['question']
)

parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({'question':docs[0].page_content}))