import pymupdf4llm
from pathlib import Path
import re
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter


input_dir = Path("data/input_pdf")
output_dir = Path("data/output_md")

def all_to_md():
    files = []
    for file in input_dir.glob("*.pdf"):
        md_text = pymupdf4llm.to_markdown(file)

        Path(f"data/output_md/{file.stem}.md").write_bytes(md_text.encode())
        # print(md_text)
        files.append(md_text)
    return files

# STRIPPING REFERENCES AFTER CONVERSION, CHANGE
def strip_references(md_text):
    files = []
    pattern = re.compile(
        r"^#{1,6}\s*\**(references)\s*\**\s*$", 
        re.IGNORECASE | re.MULTILINE
    )
    for pdf in md_text:
        match = pattern.search(pdf)
        if match:
            files.append(pdf[:match.start()].rstrip())
    
    return files

def text_splitting(documents):
    for md_file in documents:
        # splitter = MarkdownTextSplitter(chunk_size=40, chunk_overlap=0)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        print(splitter.create_documents([md_file]))
        

if __name__ == "__main__":
    md_text = all_to_md()
    docs = strip_references(md_text)
    text_splitting(docs)