from embed import embed_texts, get_collection
'''embed query then query vector db'''

def retrieve_relevant_chunks(query, n_results=6):
    collection = get_collection()
    query_embedding = embed_texts(query)
    # retrieves relevant results based on similarity ranking
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=n_results,
        # include text from doc, metadata in results
        include=["documents", "metadatas"]
    )
    # TODO: join texts with source
    relevant_chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        relevant_chunks.append({
            "text": doc,
            "paper": meta.get("paper", 0),
            "page": meta.get("page", 0)
        })
    return relevant_chunks

def generate_response(query, context):
    context_chunks = []
    # TODO: provide context in full as prompt
    for chunk in context:
        context_chunks.append(f"{chunk['text']} (paper:{chunk['paper']}, page:{chunk['page']})")
    full_context = "\n\n".join(context_chunks)
    prompt = (
        "system prompt here "
        "and context"
    )
    return full_context

if __name__ == "__main__":
    query = "what are the results of paper Political Leaning and Politicalness Classification of Texts"
    context = retrieve_relevant_chunks(query)
    # print(context)
    response = generate_response(query, context)
    print(response)