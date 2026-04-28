import chromadb
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_pdf(file_name: str):
    logger.info(f"Processing {file_name} now")
    doc = fitz.open(file_name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    documents = []
    metadatas = []
    ids = []

    chunk_counter = 0

    for page_num, page in enumerate(doc.pages()):
        page_text = page.get_text()

        page_chunks = text_splitter.split_text(page_text)

        for chunk in page_chunks:
            documents.append(chunk)
            metadatas.append({"source": file_name, "page": page_num + 1})
            ids.append(f"id_{page_num}_{chunk_counter}")
            chunk_counter += 1

    return documents, metadatas, ids


docs, metas, doc_ids = process_pdf("GDPR_PDF.pdf")

client = chromadb.PersistentClient()
collection = client.get_or_create_collection(name="gdpr_knowledge_base")

collection.add(documents=docs, metadatas=metas, ids=doc_ids)
logger.info(f"Successfully added {len(docs)} chunks to Chromadb")
print("success")
