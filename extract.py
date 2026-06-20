import pymupdf4llm
from pathlib import Path
import glob
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter


dir = Path('data/input_pdf')

def all_to_md():
    count = 1
    for file in glob.glob(f'{dir}/*.pdf'):
        md_text = pymupdf4llm.to_markdown(file)

        Path(f"data/output_md/output{count}.md").write_bytes(md_text.encode())
        count += 1

        splitter = MarkdownTextSplitter(chunk_size=40, chunk_overlap=0)

        splitter.create_documents([md_text])
        # print(md_text)

if __name__ == "__main__":
    all_to_md()