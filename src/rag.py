from embed import embed_texts, get_collection
'''embed query then query vector db'''

def embed_query(query, n_results=6):
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

    print(type(results))
    return [k for k in results["documents"][0]]


if __name__ == "__main__":
    print(embed_query("what are the results of paper Political Leaning and Politicalness Classification of Texts"))