from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
import os

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Create Google Palm LLM model
#llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    # Create an instance of the PromptTemplate class
    # The 'template' argument is assigned the prompt_template string
    # The 'input_variables' argument specifies the variables that will be replaced in the template
    PROMPT = PromptTemplate(
        template=prompt_template,               # Template with placeholders for context and question
        input_variables=["context", "question"] # List of variables that will be replaced in the template
    )

    # Create a RetrievalQA chain using a specific chain type
    chain = RetrievalQA.from_chain_type(llm=llm,                              # The language model to be used for generating answers
                                        chain_type="stuff",                   # The type of chain, "stuff" is a method for handling retrieved documents
                                        retriever=retriever,                  # The retriever object responsible for fetching relevant documents
                                        input_key="query",                    # The key in the input dictionary that contains the query
                                        return_source_documents=True,         # A flag to indicate whether to return the source documents along with the answer
                                        chain_type_kwargs={"prompt": PROMPT}) # Additional keyword arguments for the chain type, including the custom prompt

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))