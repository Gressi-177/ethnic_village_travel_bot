import os
import unicodedata
from langchain_community.vectorstores import Chroma
from constants import db_path, ETHNIC_MAP
from pyvi import ViTokenizer
from symspellpy import SymSpell, Verbosity

ethnic_groups = [
    "Kinh", "Tày", "Thái", "Mường", "Nùng", "HMông", "Dao", "Gia Rai", "Ê Đê", "Ba Na",
    "Xơ Đăng", "Sán Chay", "Cơ Ho", "Chăm", "Sán Dìu", "Hrê", "Ra Glai", "M’Nông", "X’Tiêng",
    "Bru - Vân Kiều", "Thổ", "Khơ Mú", "Cơ Tu", "Giáy", "Lào", "La Chí", "La Ha", "Pà Thẻn"
]

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

for ethnic in ethnic_groups:
    sym_spell.create_dictionary_entry(ethnic, 1)
    
def fix_question(model, text):
    ethnic_list_str = ", ".join(ethnic_groups)
    
    prompt = f"""
    Tôi có danh sách các dân tộc sau: {ethnic_list_str}. 
    Nếu câu hỏi dưới đây chứa tên dân tộc bị viết sai, hãy sửa lại chính xác theo danh sách trên.
    Và chỉ cần đưa ra lại câu hỏi đúng mà không thêm bất kì từ nào.
    Câu hỏi:
    "{text}"
    """
    response = model.generate_content(prompt)
    return response.text

def correct_text(text):
    words = ViTokenizer.tokenize(text).split()
    corrected_words = []
    
    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)
    
    return " ".join(corrected_words)

def remove_accents(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()

# def detect_ethnic_in_question(question):
#     normalized_question = remove_accents(question.lower())

#     for key, variations in ETHNIC_MAP.items():
#         for variant in variations:
#             if variant in normalized_question:
#                 return remove_accents(key).replace(" ", "_")
#     return None

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
    results = base_db.get(where={"source": source})

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