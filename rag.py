# ==========================================
# RAG SYSTEM - PERSISTENT CHROMADB
# ==========================================

import os
import hashlib

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from langchain_chroma import Chroma


# ==========================================
# 1. CONFIGURATION
# ==========================================

CHROMA_DIR = "./chroma_db"

COLLECTION_NAME = "agentic_rag_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # <-- ee string ga ne undali

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

TOP_K = 4


# ==========================================
# 2. CREATE EMBEDDING MODEL
# ==========================================

embeddings = FastEmbedEmbeddings(
    model_name=EMBEDDING_MODEL
)

# ==========================================
# 3. CONNECT TO PERSISTENT CHROMADB
# ==========================================

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)


# ==========================================
# 4. CREATE RETRIEVER
# ==========================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": TOP_K
    }
)


# ==========================================
# 5. TEXT SPLITTER
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


# ==========================================
# 6. GENERATE FILE HASH
# ==========================================

def generate_file_hash(file_path):

    hasher = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(8192)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


# ==========================================
# 7. LOAD DOCUMENT
# ==========================================

def load_document(file_path):

    extension = os.path.splitext(file_path)[1].lower()


    if extension == ".pdf":

        loader = PyPDFLoader(file_path)


    elif extension == ".docx":

        loader = Docx2txtLoader(file_path)


    elif extension == ".txt":

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )


    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )


    documents = loader.load()

    return documents


# ==========================================
# 8. SPLIT DOCUMENT
# ==========================================

def split_documents(documents):

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks


# ==========================================
# 9. CHECK WHETHER FILE ALREADY EXISTS
# ==========================================

def document_already_exists(file_hash):

    results = vectorstore.get(
        where={
            "file_hash": file_hash
        }
    )

    return len(results["ids"]) > 0


# ==========================================
# 10. ADD DOCUMENT TO CHROMADB
# ==========================================

def add_document(file_path):

    # --------------------------------------
    # Generate unique hash for file
    # --------------------------------------

    file_hash = generate_file_hash(
        file_path
    )


    # --------------------------------------
    # Check duplicate
    # --------------------------------------

    if document_already_exists(file_hash):

        return {
            "status": "already_exists",
            "message": "Document already exists in ChromaDB.",
            "file": os.path.basename(file_path)
        }


    # --------------------------------------
    # Load document
    # --------------------------------------

    documents = load_document(
        file_path
    )


    # --------------------------------------
    # Split into chunks
    # --------------------------------------

    chunks = split_documents(
        documents
    )


    # --------------------------------------
    # Add metadata
    # --------------------------------------

    file_name = os.path.basename(
        file_path
    )


    for chunk_index, chunk in enumerate(chunks):

        chunk.metadata["file_name"] = file_name

        chunk.metadata["file_hash"] = file_hash

        chunk.metadata["chunk_index"] = chunk_index


    # --------------------------------------
    # Generate unique chunk IDs
    # --------------------------------------

    ids = []

    for chunk_index in range(len(chunks)):

        chunk_id = (
            f"{file_hash}_{chunk_index}"
        )

        ids.append(chunk_id)


    # --------------------------------------
    # Add chunks to existing ChromaDB
    # --------------------------------------

    vectorstore.add_documents(
        documents=chunks,
        ids=ids
    )


    return {
        "status": "added",
        "message": "Document successfully added to ChromaDB.",
        "file": file_name,
        "chunks": len(chunks)
    }


# ==========================================
# 11. ADD MULTIPLE DOCUMENTS
# ==========================================

def add_documents(file_paths):

    results = []


    for file_path in file_paths:

        result = add_document(
            file_path
        )

        results.append(result)


    return results


# ==========================================
# 12. GET RETRIEVER
# ==========================================

def get_retriever():

    return retriever


# ==========================================
# 13. SEARCH DOCUMENTS
# ==========================================

def search_documents(query):

    documents = retriever.invoke(
        query
    )

    return documents


# ==========================================
# 14. GET ALL STORED DOCUMENTS
# ==========================================

def get_stored_documents():

    data = vectorstore.get()

    return data


# ==========================================
# 15. GET NUMBER OF STORED CHUNKS
# ==========================================

def get_document_count():

    data = vectorstore.get()

    return len(data["ids"])