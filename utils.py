import os
import unicodedata
from langchain_community.vectorstores import Chroma

from constants import db_path, ETHNIC_MAP


def remove_accents(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()


def detect_ethnic_in_question(question):
    normalized_question = remove_accents(question.lower())

    for key, variations in ETHNIC_MAP.items():
        for variant in variations:
            if variant in normalized_question:
                question = question.replace(variant, key)
                return remove_accents(key).replace(" ", "_"), question
    return None, question

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
