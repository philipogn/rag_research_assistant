import pymupdf4llm
from pathlib import Path
import re
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter


input_dir = Path("data/input_pdf")
output_dir = Path("data/output_md")

def all_to_md():
    files = []
    for file in input_dir.glob("*.pdf"):
        # page_chunks for metadeta
        md_text = pymupdf4llm.to_markdown(file, page_chunks=True)

        # Path(f"data/output_md/{file.stem}.md").write_bytes(md_text.encode())
        # # print(md_text)
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
        md_text[:match.start()].rstrip()
    return md_text

def text_splitting(documents):
    """
    func needs unique id (filename_int), documents=chunk of splitted text,
    embedding, which will be generated per chunk
    optional metadata
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # splitter = MarkdownTextSplitter(chunk_size=40, chunk_overlap=0)
    ids, texts, metadata = [], [] ,[]

    id_idx = 0

    for md_file in documents:
        paper_id = md_file["paper_id"]
        for page in md_file["content"]:
            page_text = strip_references(page["text"])
            chunks = splitter.create_documents([page_text])
            for chunk in chunks:
                chunk_ids = f"{paper_id}_{id_idx}"
                ids.append(chunk_ids)
                texts.append(chunk.page_content)
                print(page.get("metadata", {}).get("page_number", 0))
                id_idx += 1
    print(len(texts))
    print(len(ids))


if __name__ == "__main__":
    md_text = all_to_md()
    # docs = strip_references(md_text)
    text_splitting(md_text)