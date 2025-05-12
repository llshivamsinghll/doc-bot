from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.prompt import system_prompt

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

# Load environment variables
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

# Initialize embeddings
try:
    embeddings = download_hugging_face_embeddings()
except Exception as embed_error:
    logger.error(f"Embedding download failed: {embed_error}")
    raise

# Initialize vector store
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

# Initialize LLM
try:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,  # Using env var instead of hardcoded key
        model_name="llama3-70b-8192",
        temperature=0.4,
        max_tokens=500,
    )
except Exception as llm_error:
    logger.error(f"LLM initialization failed: {llm_error}")
    raise

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

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

@app.get("/")
async def root():
    return {"message": "RAG API is running. Use /get endpoint for queries."}

if __name__ == "__main__":
    import uvicorn
    # For deployment, use 0.0.0.0 to make the server accessible externally
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)