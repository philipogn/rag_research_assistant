from pathlib import Path
import re

from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownTextSplitter
import config

from embed import embed_texts, get_collection

input_dir = Path("data")
output_dir = Path("data/output_md")

def docling_to_md(file: Path, converter: DocumentConverter):
    result = converter.convert(str(file))
    doc = result.document

    doc.save_as_markdown(output_dir / f"test_{file.stem}.md")
    # print(doc.export_to_markdown())

def files_to_docling():
    converter = DocumentConverter()
    papers = []
    for file in input_dir.glob("*.pdf"):
        content = docling_to_md(file, converter)
        papers.append({
            "paper_id": file.stem,
            "content": content
        })
        break


# TODO: (currently unused) pdf converter breaks, fix
def strip_references(md_text):
    pattern = re.compile(
        r"^#{1,6}\s*\**(references)\s*\**\s*$", 
        re.IGNORECASE | re.MULTILINE
    )
    match = pattern.search(md_text)
    if match:
        return md_text[:match.start()].rstrip()
    return md_text

def text_splitting(documents: list[dict]):
    """
    func needs unique id (filename_int), documents=chunk of splitted text,
    embedding, which will be generated per chunk
    optional metadata
    """
    splitter = MarkdownTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    collection = get_collection()
    ids, texts, metadata = [], [] ,[]

    # Each doc is dict of {paper_id, md_content, tables}
    for md_file in documents:
        paper_id = md_file["paper_id"]
        # paper_idx = 0
        collection.delete(where={"paper": paper_id})

        chunks = chunking_function_here(md_file["content"], splitter)

        for chunk in chunks:
            chunk_ids = f"{paper_id}_{paper_idx}"
            ids.append(chunk_ids)
            texts.append(chunk)
            metadata.append({"paper": paper_id, "page": page_number})
            paper_idx += 1

    embeddings = [embed_texts(text) for text in texts]
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadata)
    return ids, texts, metadata


if __name__ == "__main__":
    files = files_to_docling()
    print(f"Chunking {len(files)} files")
    # ids, texts, metadata = text_splitting(files)
    # print(f"Embedded {len(ids)} chunks")