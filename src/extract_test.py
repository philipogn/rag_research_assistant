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
    # print(items)
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

def _preprocess_table(table_markdown: str) -> str:
    """
    Docling to markdown introduces whitespace for alignment, 
    cleaning up to reduce chunk size 
    """
    cleaned = []
    for line in table_markdown.split("\n"):
        # matches at 4 dashes, minimise seperator to 3 dashes for readability/debugging and avoids removing dash for n/a data
        collapsed = re.sub(r"-{4,}", "---", line) 
        cleaned.append("|".join(part.strip() for part in collapsed.split("|")))
    return "\n".join(cleaned)

def build_chunks(paper: list[tuple], splitter: MarkdownTextSplitter):
    chunks = [] # build to return
    heading = "" # keep track of latest heading for section aware splitting
    current_page = None
    pending_parts = []

    def build_pending_parts():
        if not pending_parts: return
        combined = "\n\n".join(pending_parts)
        for piece in splitter.split_text(combined):
            sectioned_chunk = f"[Section: {heading}]\n{piece}"
            chunks.append((current_page, sectioned_chunk))

    for page_num, label, text in paper:
        if label == "section_header":
            # function to build all previous text under prev header
            build_pending_parts()
            # then update header
            heading = text.strip()
            continue

        # TODO: if table, save somewhere and continue building text before creating chunk to avoid between table text splitting 
        if label == "table":
            # same as above to avoid table seperating texts (keeping tables atomic/poor retrieval on splitted tables)
            build_pending_parts()
            clean_table = _preprocess_table(text)
            tagged_table = f"[Section: {heading}]\n{clean_table}"
            chunks.append((page_num, tagged_table))
            # print(chunks[-1])
            continue

        if current_page is not None and page_num != current_page: # if page mismatch, build prev text, keeps metadata intact
            build_pending_parts()
        current_page = page_num
        pending_parts.append(text)

    build_pending_parts() # final run for any leftover text
    return chunks


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

        for idx, (page_number, chunk) in enumerate(chunks):
            chunk_ids = f"{paper_id}_{idx}"
            ids.append(chunk_ids)
            texts.append(chunk)
            metadata.append({"paper": paper_id, "page": page_number})

    embeddings = [embed_texts(text) for text in texts]
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadata)
    return ids, texts, metadata


if __name__ == "__main__":
    files = files_to_docling()
    print(f"Chunking {len(files)} files")
    ids, texts, metadata = text_splitting(files)
    print(f"Embedded {len(ids)} chunks")