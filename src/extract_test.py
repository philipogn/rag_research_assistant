from pathlib import Path
import re

from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownTextSplitter
import config

from embed import embed_texts, get_collection

input_dir = Path("data")
output_dir = Path("data/output_md")

def _extract_paper_content(file: Path, converter: DocumentConverter) -> list[dict]:
    """Extracts page number, label, and text attributes"""
    result = converter.convert(str(file))
    doc = result.document
    # doc.save_as_json(output_dir / f"test_{file.stem}.json")
    # print(doc.export_to_markdown())
    
    items = []
    for item, _ in doc.iterate_items():
        if not getattr(item, "prov", None):
            continue
        page_num = item.prov[0].page_no
        label = getattr(item, "label", "")

        if label == "picture": # skip pics, not useful
            continue
        elif label == "table":
            items.append((page_num, "table", item.export_to_markdown(doc)))
        else:
            text = getattr(item, "text", "")
            if text and text.strip():
                items.append((page_num, label, text))
    print(items)
    return items


def files_to_docling():
    converter = DocumentConverter()
    papers = []
    for file in input_dir.glob("*.pdf"):
        content = _extract_paper_content(file, converter)
        papers.append({
            "paper_id": file.stem,
            "content": content
        })
        break # TEMPORARY, REMOVE AFTER TESTING
    return papers


# TODO: could use docling label attribute to exclude references instead of regex
# def strip_references(md_text):
#     pattern = re.compile(
#         r"^#{1,6}\s*\**(references)\s*\**\s*$", 
#         re.IGNORECASE | re.MULTILINE
#     )
#     match = pattern.search(md_text)
#     if match:
#         return md_text[:match.start()].rstrip()
#     return md_text

def build_chunks():
    pass


def text_splitting(documents: list[dict]):
    """
    Chunk and embed the contents
    Each paper are keys of {paper_id, content(page_num, label, text)}
    """
    splitter = MarkdownTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    collection = get_collection()
    ids, texts, metadata = [], [] ,[]

    for paper in documents:
        paper_id = paper["paper_id"]
        collection.delete(where={"paper": paper_id})

        chunks = build_chunks(paper["content"], splitter)

        for idx, chunk in enumerate(chunks):
            chunk_ids = f"{paper_id}_{idx}"
            ids.append(chunk_ids)
            texts.append(chunk)
            # metadata.append({"paper": paper_id, "page": page_number})

    embeddings = [embed_texts(text) for text in texts]
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadata)
    return ids, texts, metadata


if __name__ == "__main__":
    files = files_to_docling()
    print(f"Chunking {len(files)} files")
    # ids, texts, metadata = text_splitting(files)
    # print(f"Embedded {len(ids)} chunks")