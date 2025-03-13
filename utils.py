import os
import re
import unicodedata
from datetime import datetime
from langchain_community.vectorstores import Chroma
from constants import data_path, db_path, COLLECTION_PREFIX
from chroma_data_store import get_latest_version
import mysql.connector

ethnic_groups = [
    "Kinh", "Tày", "Thái", "Mường", "Nùng", "HMông", "Dao", "Gia Rai", "Ê Đê", "Ba Na",
    "Xơ Đăng", "Sán Chay", "Cơ Ho", "Chăm", "Sán Dìu", "Hrê", "Ra Glai", "M’Nông", "X’Tiêng",
    "Bru - Vân Kiều", "Thổ", "Khơ Mú", "Cơ Tu", "Giáy", "Lào", "La Chí", "La Ha", "Pà Thẻn"
]

def get_database_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="bantourdulich"
    )

def get_database_schema():
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES;")
    tables = [row[0] for row in cursor.fetchall()]
    
    schema = {}
    for table in tables:
        cursor.execute(f"DESCRIBE {table};")
        schema[table] = [col[0] for col in cursor.fetchall()]
    
    cursor.close()
    conn.close()
    return schema

def execute_query(query):
    print(query)
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    columns = [col[0] for col in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    cursor.close()
    conn.close()
    return rows
    
def fix_question(model, text):
    ethnic_list_str = ", ".join(ethnic_groups)
    
    prompt = f"""
    Tôi có danh sách các dân tộc sau: {ethnic_list_str}. 
    Nếu câu hỏi dưới đây chứa tên dân tộc bị viết sai, hãy sửa lại chính xác theo danh sách trên.
    Và chỉ cần đưa ra lại câu hỏi đúng mà không thêm bất kì kí tự nào.
    Câu hỏi:
    "{text}"
    """
    response = model.generate_content(prompt)
    return response.text

def remove_accents(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()

def detect_ethnic_in_question(question):
    for key in ethnic_groups:
        if key in question:
            return remove_accents(key).replace(" ", "_")
    return None

def get_ethnic_db(ethnic_name, base_embedding_function):
    ethnic_persist_dir = f"./chroma_db_new/{ethnic_name}"
    if os.path.exists(ethnic_persist_dir):
        return Chroma(
            persist_directory=ethnic_persist_dir,
            embedding_function=base_embedding_function
        )

    source = f"/content/dantoc_new/{ethnic_name}.md"
    base_db = Chroma(persist_directory=db_path, embedding_function=base_embedding_function)
    existing_collections = base_db._client.list_collections()
    latest_version = get_latest_version(existing_collections)
    collection_name = f"{COLLECTION_PREFIX}{latest_version}"
    latest_collection = base_db._client.get_or_create_collection(collection_name)
    
    results = latest_collection.get(where={"source": source})

    from langchain.schema import Document
    documents = [
        Document(page_content=content, metadata={"source": source})
        for content in results["documents"]
    ]

    temp_ethnic_db = Chroma.from_documents(
        documents=documents,
        ids=[f"{ethnic_name}_{i}" for i in range(len(documents))],
        embedding=base_embedding_function,
        persist_directory=ethnic_persist_dir
    )

    return temp_ethnic_db

def normalize_score(score, min_score, max_score):
    return (score - min_score) / (max_score - min_score)

def read_prompt_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def generate_sql_query(model, prompt_file, question):
    cur_date = datetime.now()
    day = cur_date.day
    month = cur_date.month
    year = cur_date.year
    
    prompt = read_prompt_from_file(data_path + prompt_file).format(
        question=question,
        day=day,
        month=month,
        year=year,
        ethnic_list_str=", ".join(ethnic_groups)
    )
    
    response = model.generate_content(prompt)
    
    sql_query = re.search(r"```sql\n(.*?)\n```", response.text, re.DOTALL)
    return sql_query.group(1).strip() if sql_query else response.text.strip()

def format_tour_info(data_list):
    messages = []
    for idx, data in enumerate(data_list, start=1):
        tour_info = f"""
        <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 20px; border-radius: 5px; background-color: #f9f9f9;">
            <h2>📢 Thông tin tour 🏝️</h2>
            <p><strong>📌 Tiêu đề:</strong> {data.get('t_title', 'N/A')}</p>
            <p><strong>🗙️ Hành trình:</strong> {data.get('t_journeys', 'N/A')}</p>
            <p><strong>🗓️ Lịch trình:</strong> {data.get('t_schedule', 'N/A')}</p>
            <p><strong>🚖 Phương tiện di chuyển:</strong> {data.get('t_move_method', 'N/A')}</p>
            <p><strong>🚪 Cổng xuất phát:</strong> {data.get('t_starting_gate', 'N/A')}</p>
            <p><strong>📆 Ngày bắt đầu:</strong> {data.get('t_start_date', 'N/A')}</p>
            <p><strong>📆 Ngày kết thúc:</strong> {data.get('t_end_date', 'N/A')}</p>
            <p><strong>👥 Số khách:</strong> {data.get('t_number_guests', 'N/A')}</p>
            <p><strong>💰 Giá người lớn:</strong> {data.get('t_price_adults', 'N/A')} VND</p>
            <p><strong>👶 Giá trẻ em:</strong> {data.get('t_price_children', 'N/A')} VND</p>
            <p><strong>🔥 Khuyến mãi:</strong> {data.get('t_sale', 'N/A')}%</p>
            <p><strong>👀 Lượt xem:</strong> {data.get('t_view', 'N/A')}</p>
            <p><strong>📝 Mô tả:</strong> {data.get('t_description', 'N/A')}</p>
            <p><strong>📚 Nội dung:</strong> {data.get('t_content', 'N/A')}</p>
            <p><strong>🖼️ Album ảnh:</strong> {data.get('t_anbum_image', 'N/A')}</p>
            <p><strong>📸 Hình ảnh chính:</strong> <img src="{data.get('t_image', '#')}" alt="Tour Image" style="max-width:100%; height:auto;"></p>
            <p><strong>📍 Địa điểm:</strong> {data.get('t_location_id', 'N/A')}</p>
            <p><strong>👤 Người tạo:</strong> {data.get('t_user_id', 'N/A')}</p>
            <p><strong>📊 Số người đăng ký:</strong> {data.get('t_number_registered', 'N/A')}</p>
            <p><strong>❤️ Lượt theo dõi:</strong> {data.get('t_follow', 'N/A')}</p>
            <p><strong>📌 Trạng thái:</strong> {data.get('t_status', 'N/A')}</p>
        </div>
        """
        messages.append(tour_info)
    
    return "".join(messages)
