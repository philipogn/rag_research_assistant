import json
import re
import pymupdf4llm
from pathlib import Path
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
import config

from embed import embed_texts, get_collection

input_dir = Path("data")

def _tables_by_page(file: Path) -> dict[int, list[dict]]:
    """Returns dict of page_number: list of Table object(s), if table in pages"""
    doc = json.loads(pymupdf4llm.to_json(file))
    tables = {}
    for page in doc["pages"]:
        page_tables = [box["table"] for box in page["boxes"] if box["boxclass"] == "table"]
        if page_tables:
            tables[page["page_number"]] = page_tables
    return tables


def all_to_md() -> list[dict]:
    """
    For each doc:
    paper_id: UUID for each paper
    content: full doc content as markdown text
    tables_by_pages: {page_number: Table object} //metadata and content in array/md text in object
    """
    output_dir = input_dir / "output_md"
    output_dir.mkdir(exist_ok=True)

    files = []
    for file in input_dir.glob("*.pdf"):
        md_text = pymupdf4llm.to_markdown(file, page_chunks=True)   # page_chunks for metadeta

        # full_text = "\n\n".join(page["text"] for page in md_text)
        # (output_dir / f"{file.stem}.md").write_text(full_text, encoding="utf-8")

        files.append({
            "paper_id": file.stem,
            "content": md_text,
            "tables_by_page": _tables_by_page(file),
        })
    return files


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

def _chunk_table_markdown(table_markdown: str, chunk_size: int) -> list[str]:
    """Split an oversized table on row boundaries, repeating the header+separator in each piece."""
    if len(table_markdown) <= chunk_size:
        return [table_markdown]

    lines = table_markdown.split("\n")
    header = lines[:2]  # header row + separator row
    chunks = []
    current = list(header)
    current_len = len("\n".join(current))
    for row in lines[2:]:
        if current_len + len(row) + 1 > chunk_size and len(current) > 2:
            chunks.append("\n".join(current))
            current = list(header)
            current_len = len("\n".join(current))
        current.append(row)
        current_len += len(row) + 1
    if len(current) > 2:
        chunks.append("\n".join(current))
    return chunks


def _split_page_text(text: str, tables: list[dict], splitter: MarkdownTextSplitter) -> list[str]:
    """Split page markdown into chunks, keeping detected tables intact"""
    non_table_parts = []
    table_chunks = []
    remaining = text
    for table in tables:
        idx = remaining.find(table["markdown"]) # find start idx of table
        if idx == -1: # if no tables found
            continue
        before, remaining = remaining[:idx], remaining[idx + len(table["markdown"]):]
        non_table_parts.append(before)
        table_chunks.extend(_chunk_table_markdown(table["markdown"], config.CHUNK_SIZE))
    non_table_parts.append(remaining)

    chunks = []
    combined_text = "\n\n".join(part for part in non_table_parts if part.strip())
    if combined_text.strip():
        chunks.extend(splitter.split_text(combined_text))
    chunks.extend(table_chunks)
    return chunks


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
        paper_idx = 0
        tables_by_page = md_file.get("tables_by_page", {})

        collection.delete(where={"paper": paper_id})

        for page in md_file["content"]:
            # page_text = strip_references(page["text"])
            page_number = page.get("metadata", {}).get("page_number", 0)
            chunks = _split_page_text(page["text"], tables_by_page.get(page_number, []), splitter)

            for idx, chunk in enumerate(chunks):
                chunk_ids = f"{paper_id}_{paper_idx}"
                ids.append(chunk_ids)
                texts.append(chunk)
                metadata.append({"paper": paper_id, "page": page_number})
                paper_idx += 1

    embeddings = [embed_texts(text) for text in texts]
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadata)
    return ids, texts, metadata


if __name__ == "__main__":
    md_text = all_to_md()
    print(f"Chunking {len(md_text)} files")
    ids, texts, metadata = text_splitting(md_text)
    print(f"Embedded {len(ids)} chunks")