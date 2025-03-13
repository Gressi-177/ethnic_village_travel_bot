from fastapi import FastAPI, HTTPException
import os
import time
import logging
import google.generativeai as genai
from datetime import datetime
from langchain_community.llms import CTransformers
from langchain.embeddings import SentenceTransformerEmbeddings
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer
import concurrent.futures

from constants import model_file_path, threshold, min_score, max_score, API_KEY
from models import QuestionRequest, QuestionResponse, DataUpdateRequest, DataUpdateResponse
from utils import get_ethnic_db, detect_ethnic_in_question, format_tour_info, get_database_schema, generate_sql_query, execute_query
from chroma_data_store import generate_data_store

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
executor = concurrent.futures.ThreadPoolExecutor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

@app.on_event("startup")
async def startup_event():
    global llm, tokenizer, embedding_function, germini_model

    start_time = time.time()
    
    logger.info("Starting model initialization...")

    llm = CTransformers(
        model=model_file_path,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    llm_time = time.time() - start_time
    logger.info(f"LLM initialization completed in {llm_time:.2f} seconds")

    tokenizer_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("vilm/vinallama-7b-chat")
    tokenizer_time = time.time() - tokenizer_start
    logger.info(f"Tokenizer initialization completed in {tokenizer_time:.2f} seconds")
    
    genai.configure(api_key=API_KEY)
    germini_model = genai.GenerativeModel("gemini-1.5-pro")

    embedding_start = time.time()
    embedding_function = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_time = time.time() - embedding_start
    logger.info(f"Embedding model initialization completed in {embedding_time:.2f} seconds")

    total_time = time.time() - start_time
    logger.info(f"Total initialization completed in {total_time:.2f} seconds")
    
@app.get("/get_sql", response_model=QuestionResponse)
async def get_sql(question: str):
    model_req = generate_sql_query(germini_model, "/prompt.txt", question)
    
    print(model_req)
    
    if "SELECT" in model_req.upper():
        data = execute_query(model_req)
    else:
        data = model_req
    
    return QuestionResponse(
        answer=model_req,
        ethnic = model_req
    )

def get_res_by_data(data):
    # if not data:
    #     return QuestionResponse(answer="Không có dữ liệu", ethnic="Không có dữ liệu")
    return QuestionResponse(
        answer=format_tour_info(data),
        ethnic=format_tour_info(data),
    )
            
def get_res_by_question(fixed_question, request_start):
    ethnic = detect_ethnic_in_question(fixed_question)

    if not ethnic:
        return QuestionResponse(
            answer="Tôi không biết bạn đang muốn tìm hiểu về dân tộc nào. Hãy nêu đầy đủ câu hỏi với tên dân tộc.",
            ethnic=ethnic,
        )

    # Database retrieval timing
    db_start = time.time()
    ethnic_db = get_ethnic_db(ethnic, embedding_function)
    db_time = time.time() - db_start
    logger.info(f"Database initialization completed in {db_time:.2f} seconds")

    template = """
    <|im_start|>system
    Bạn là một trợ lí AI hữu ích. Hãy sử dụng thông tin dưới đây để lấy ra câu trả lời ngắn gọn cho câu hỏi bên dưới mà không thêm bất kỳ kí tự nào.
    Nếu không tìm thấy câu trả lời trong thông tin mà dạng câu so sánh thì bạn có thể tự trả lời.
    Nếu vẫn không tìm thấy câu trả lời thì trả lời: Không tìm thấy câu trả lời.
    Thông tin: {context}
    <|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    """
    
    # Context search timing
    search_start = time.time()
    results = ethnic_db.similarity_search_with_relevance_scores(fixed_question, k=2)
    print(fixed_question, results)
    
    # or normalize_score(results[0][1], min_score, max_score) < threshold
    # if(len(results) == 0):
    #     return QuestionResponse(
    #         answer="Không có câu trả lời cho câu hỏi của bạn!",
    #         ethnic=ethnic,
    #         fixed_question=fixed_question
    #     )
    
    # merged_context = "\n\n".join([doc.page_content for doc, _ in results])
    # search_time = time.time() - search_start
    # logger.info(f"Context search completed in {search_time:.2f} seconds")
    

    # # LLM inference timing
    # formatted_prompt = template.format(context=merged_context, question=fixed_question)

    # inference_start = time.time()
    # answer = llm(formatted_prompt)
    # inference_time = time.time() - inference_start
    # logger.info(f"LLM inference completed in {inference_time:.2f} seconds")

    # total_time = time.time() - request_start
    # logger.info(f"Total request processing completed in {total_time:.2f} seconds")

    # # Log timing summary
    # timing_summary = {
    #     "timestamp": datetime.now().isoformat(),
    #     "question": fixed_question,
    #     "ethnic": ethnic,
    #     "database_init_time": db_time,
    #     "context_search_time": search_time,
    #     "llm_inference_time": inference_time,
    #     "total_processing_time": total_time
    # }
    # logger.info(f"Timing summary: {timing_summary}")
    
    # return QuestionResponse(
    #     answer=answer.strip().split("<|im_end|>")[0].split("<|im_start|>")[0],
    #     ethnic=ethnic,
    # )
    return QuestionResponse(
        answer="",
        ethnic=ethnic,
    )

@app.post("/answer", response_model=QuestionResponse)
async def get_answer(request: QuestionRequest):
    try:
        request_start = time.time()
        logger.info(f"Processing question: {request.question}")

        # Ethnic detection timing
        # fixed_question = fix_question(germini_model, request.question).strip()
        model_res = generate_sql_query(germini_model, "/prompt.txt", request.question).strip()
        
        if "SELECT" in model_res.upper():
            data = execute_query(model_res)
            return get_res_by_data(data)
        else:
            fixed_question = model_res
            return get_res_by_question(fixed_question, request_start)
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_data", response_model=DataUpdateResponse)
async def update_data(request: DataUpdateRequest):
    try:
        executor.submit(generate_data_store, request.file_name, request.data)
        
        return DataUpdateResponse(
            status=True
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))