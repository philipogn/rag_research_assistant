import re
import pymupdf4llm
from pathlib import Path
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
import config

from embed import embed_texts, get_collection

input_dir = Path("data/input_pdf")

def all_to_md():
    files = []
    for file in input_dir.glob("*.pdf"):
        # page_chunks for metadeta
        md_text = pymupdf4llm.to_markdown(file, page_chunks=True)

        # Path(f"data/output_md/{file.stem}.md").write_bytes(md_text.encode())
        files.append({
            "paper_id": file.stem,
            "content": md_text
        })
    return files


# STRIPPING REFERENCES AFTER CONVERSION, CHANGE
def strip_references(md_text):
    pattern = re.compile(
        r"^#{1,6}\s*\**(references)\s*\**\s*$", 
        re.IGNORECASE | re.MULTILINE
    )
    match = pattern.search(md_text)
    if match:
        return md_text[:match.start()].rstrip()
    return md_text


def text_splitting(documents):
    """
    func needs unique id (filename_int), documents=chunk of splitted text,
    embedding, which will be generated per chunk
    optional metadata
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    # splitter = MarkdownTextSplitter(chunk_size=40, chunk_overlap=0)

    collection = get_collection()
    ids, texts, metadata = [], [] ,[]

    for md_file in documents:
        paper_id = md_file["paper_id"]
        paper_idx = 0

        for page in md_file["content"]:
            # page_text = strip_references(page["text"])
            chunks = splitter.create_documents([page["text"]])
            page_number = page.get("metadata", {}).get("page_number", 0)

            for idx, chunk in enumerate(chunks):
                chunk_ids = f"{paper_id}_{paper_idx}"
                ids.append(chunk_ids)
                texts.append(chunk.page_content)
                metadata.append({"paper": paper_id, "page": page_number})
                paper_idx += 1

    embeddings = [embed_texts(text) for text in texts]
    # print(len(ids), len(texts), len(metadata))

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadata)
    return ids, texts, metadata


if __name__ == "__main__":
    md_text = all_to_md()
    text_splitting(md_text)