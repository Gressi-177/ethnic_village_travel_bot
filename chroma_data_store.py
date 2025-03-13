import os
import re
import shutil

from constants import COLLECTION_PREFIX, CHROMA_PATH, DATA_PATH, CHROMA_PATH_VER

# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

def generate_data_store(file_name: str, data: str):
    if file_name:
        documents = [Document(page_content=data, metadata={"source": f"/content/dantoc_new/{file_name}"})]
        chunks = split_text(documents)
        create_new_versioned_collection(file_name, chunks)
    else:
        documents = load_documents(None)
        chunks = split_text(documents)
        save_to_chroma(chunks)



def load_documents(file_name: str):
    if file_name:
        file_path = os.path.join(DATA_PATH, file_name)
        
        if not os.path.exists(file_path):
            return
        
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        loader = DirectoryLoader(DATA_PATH, glob="*.md")
        
    documents = loader.load()
    print(documents)
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]    
    print(document.page_content)
    print(document.metadata)
    
    for chunk in chunks:
        chunk.metadata["source"] = chunk.metadata["source"].replace("data/dantoc_new\\", "/content/dantoc_new/")


    return chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedding_function = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def get_latest_version(collection_names):
    """Tìm số phiên bản mới nhất từ danh sách collection."""
    versions = []
    for name in collection_names:
        match = re.search(rf"{COLLECTION_PREFIX}(\d+)", name)
        if match:
            versions.append(int(match.group(1)))

    return max(versions) if versions else 0

def create_new_versioned_collection(file_name: str, chunks: list[Document]):
    embedding_function = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Kiểm tra nếu database đã tồn tại
    if os.path.exists(CHROMA_PATH):
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Lấy danh sách collections hiện có
        existing_collections = db._client.list_collections()

        # Xác định version mới nhất và tăng lên
        latest_version = get_latest_version(existing_collections)
        new_version = latest_version + 1
        new_collection_name = f"{COLLECTION_PREFIX}{new_version}"

        print(f"Creating new collection: {new_collection_name}")

        # Tạo collection mới
        new_db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function,
            collection_name=new_collection_name
        )

        if latest_version > 0:
            last_collection_name = f"{COLLECTION_PREFIX}{latest_version}"
            old_collection = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embedding_function,
                collection_name=last_collection_name
            )
            old_docs = old_collection.get(include=["documents", "metadatas"])

            if old_docs["documents"]:
                new_db.add_texts(
                    texts=old_docs["documents"],
                    metadatas=old_docs["metadatas"]
                )
                print(f"Copied {len(old_docs['documents'])} documents from '{last_collection_name}'.")
        
        # Xóa các chunks cũ theo file_name
        source_path = f"/content/dantoc_new/{file_name}"
        ids_to_delete = []
        existing_docs = new_db.get(include=["documents", "metadatas"])
        
        for i, meta in enumerate(existing_docs["metadatas"]):
            if meta.get("source") == source_path:
                ids_to_delete.append(existing_docs["ids"][i])

        if ids_to_delete:
            new_db.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} old chunks from '{file_name}'.")

        # Thêm dữ liệu mới vào collection mới
        new_db.add_documents(chunks)
        print(f"Added {len(chunks)} new chunks to collection '{new_collection_name}'.")

    else:
        print("No existing Chroma database found. Creating the first version.")
        new_collection_name = f"{COLLECTION_PREFIX}1"
        new_db = Chroma.from_documents(
            chunks, embedding_function, persist_directory=CHROMA_PATH, collection_name=new_collection_name
        )

    new_db.persist()
    print(f"Updated Chroma with collection '{new_collection_name}'.")