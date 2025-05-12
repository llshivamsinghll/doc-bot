from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging


from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.prompt import system_prompt

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


try:
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

    if not PINECONE_API_KEY or not GROQ_API_KEY:
        raise ValueError("Missing required API keys")

    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
except Exception as env_error:
    logger.error(f"Environment configuration error: {env_error}")
    raise


try:
    embeddings = download_hugging_face_embeddings()
except Exception as embed_error:
    logger.error(f"Embedding download failed: {embed_error}")
    raise


index_name = "doc-bot"
try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
except Exception as vector_store_error:
    logger.error(f"Vector store initialization failed: {vector_store_error}")
    raise

# Create retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3} 
)


try:
    llm = ChatGroq(
        api_key="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",
        model_name="llama3-70b-8192",
        temperature=0.4,
        max_tokens=500,
    )
except Exception as llm_error:
    logger.error(f"LLM initialization failed: {llm_error}")
    raise

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


class ChatRequest(BaseModel):
    msg: str


@app.post("/get")
async def chat(msg: str = Form(...)):
    try:
        input = msg
        logger.info(f"Received input: {input}")
        
        
        response = rag_chain.invoke({"input": input})
        
       
        logger.info(f"Generated response: {response['answer']}")
        return response["answer"]
    
    except Exception as e:
        
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=0000)
