from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os

api_key = os.environ["GOOGLE_API_KEY"] # get this free api key from https://makersuite.google.com/

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key, temperature=0.1)

poem = llm("Write a 4 line poem of my love for samosa")
print(poem)


loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")

# Store the loaded data in the 'data' variable
data = loader.load()

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

e = instructor_embeddings.embed_query("What is your refund policy?")

# Create a FAISS instance for vector database from 'data'
vectordb = FAISS.from_documents(documents=data,
                                 embedding=instructor_embeddings)

# Create a retriever for querying the vector database
retriever = vectordb.as_retriever(score_threshold = 0.7)

prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}


from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True,
                            chain_type_kwargs=chain_type_kwargs)

#print(chain('Do you provide job assistance and also do you provide job gurantee?'))