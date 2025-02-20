from fastapi import FastAPI, HTTPException
import os
import time
import logging
from datetime import datetime
from langchain_community.llms import CTransformers
from langchain.embeddings import SentenceTransformerEmbeddings
from transformers import AutoTokenizer

from constants import model_file_path
from models import QuestionRequest, QuestionResponse
from utils import detect_ethnic_in_question, get_ethnic_db

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_timing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ethnic QA API")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


@app.on_event("startup")
async def startup_event():
    global llm, tokenizer, embedding_function

    start_time = time.time()
    logger.info("Starting model initialization...")

    llm = CTransformers(
        model=model_file_path,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.01
    )
    llm_time = time.time() - start_time
    logger.info(f"LLM initialization completed in {llm_time:.2f} seconds")

    tokenizer_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("vilm/vinallama-7b-chat")
    tokenizer_time = time.time() - tokenizer_start
    logger.info(f"Tokenizer initialization completed in {tokenizer_time:.2f} seconds")

    embedding_start = time.time()
    embedding_function = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_time = time.time() - embedding_start
    logger.info(f"Embedding model initialization completed in {embedding_time:.2f} seconds")

    total_time = time.time() - start_time
    logger.info(f"Total initialization completed in {total_time:.2f} seconds")


# API endpoint
@app.post("/answer", response_model=QuestionResponse)
async def get_answer(request: QuestionRequest):
    try:
        request_start = time.time()
        logger.info(f"Processing question: {request.question}")

        # Ethnic detection timing
        ethnic_start = time.time()
        ethnic, fixed_question = detect_ethnic_in_question(request.question)
        ethnic_time = time.time() - ethnic_start
        logger.info(f"Ethnic detection completed in {ethnic_time:.2f} seconds. Detected: {ethnic}")

        if not ethnic:
            raise HTTPException(status_code=400, detail="No ethnic group detected in the question")

        # Database retrieval timing
        db_start = time.time()
        ethnic_db = get_ethnic_db(ethnic, embedding_function)
        db_time = time.time() - db_start
        logger.info(f"Database initialization completed in {db_time:.2f} seconds")

        template = """
        <|im_start|>system
        Sử dụng thông sau đây để tạo câu trả lời cho câu hỏi bên dưới và hãy thay đổi format cho đẹp hơn. 
        Nếu có thì hãy tạo câu trả lời, nếu không thì đừng tạo câu trả lời.
        Thông tin: {context}
        <|im_end|>
        <|im_start|>user
        {question}<|im_end|>
        <|im_start|>assistant
        """

        # Context search timing
        search_start = time.time()
        results = ethnic_db.similarity_search_with_relevance_scores(fixed_question, k=1)
        merged_context = "\n\n".join([doc.page_content for doc, _ in results])
        search_time = time.time() - search_start
        logger.info(f"Context search completed in {search_time:.2f} seconds")

        # LLM inference timing
        formatted_prompt = template.format(context=merged_context, question=request.question)

        inference_start = time.time()
        answer = llm(formatted_prompt)
        inference_time = time.time() - inference_start
        logger.info(f"LLM inference completed in {inference_time:.2f} seconds")

        total_time = time.time() - request_start
        logger.info(f"Total request processing completed in {total_time:.2f} seconds")

        # Log timing summary
        timing_summary = {
            "timestamp": datetime.now().isoformat(),
            "question": request.question,
            "ethnic": ethnic,
            "ethnic_detection_time": ethnic_time,
            "database_init_time": db_time,
            "context_search_time": search_time,
            "llm_inference_time": inference_time,
            "total_processing_time": total_time
        }
        logger.info(f"Timing summary: {timing_summary}")

        return QuestionResponse(
            answer=answer.strip(),
            ethnic=ethnic,
            fixed_question=fixed_question
        )

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))