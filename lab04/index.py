import json
from pathlib import Path
import chromadb
import ollama
import sys


def normalize_collection_name(collection_name: str) -> str:
    invalid_chars = [":", "."]
    for char in invalid_chars:
        collection_name = collection_name.replace(char, "_")
    return collection_name


def get_or_create_collection(input_path: Path = Path("data/pandas_help_corpus.json"), model: str = "qwen3-embedding:8b") -> chromadb.Collection:
    
    client = chromadb.PersistentClient(path="data/chroma")
    name = input_path.stem
    collection_name = f"{name}_with_{model}"
    collection_name = normalize_collection_name(collection_name)

    if collection_name in [collection.name for collection in client.list_collections()]:
        print(f"Collection found")
        return client.get_collection(name=collection_name)
    
    # create
    print(f"Creating Collection...")
    collection = client.create_collection(name=collection_name)

    print(f"Loading JSON...")
    with open(input_path, "r") as f:
        chunks = json.load(f)
    print(f"JSON loaded")

    for i, chunk in enumerate(chunks):
        
        response = ollama.embed(model=model, input=str(chunk))
        embedding = response["embeddings"][0]
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk["doc"]],
            metadatas=[{
                "symbol": chunk["symbol"],
                "signature": chunk["signature"],
                "embedding_model": model
            }]
        )
        print(f"Processed {i+1}/{len(chunks)} chunks", end="\r")

    print(f"Collection created")

    return collection


def get_embedding_model(collection, id: int = 0):
    return collection.get(ids=[str(id)])['metadatas'][0]['embedding_model']
    

def query_collection(collection, query: str, n_results: int = 5) -> list[dict]:
    embedding_model = get_embedding_model(collection)
    query_embedding = ollama.embed(model=embedding_model, input=query)["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    # return clean_results(results)
    return results

def clean_results(results):
    ids = results["ids"][0]
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # flattens results
    outputs = []
    for i in range(len(ids)):
        output = {
            "id": ids[i],
            "symbol": metadatas[i]["symbol"],
            "signature": metadatas[i]["signature"],
            "metadatas": metadatas[i],
            "doc": docs[i],
            "distance": distances[i]
        }
        outputs.append(output)

    return outputs


def main():
    current_model = "qwen3-embedding:8b"
    if len(sys.argv) > 1:
        current_model = sys.argv[1]

    input_path = Path("data/pandas_help_corpus.json")
    collection = get_or_create_collection(input_path, current_model)

    query = f"waht is a dataframe"
    results = query_collection(collection, query)
    # results = clean_results(results)
    print(type(results), len(results))
    
    for key, value in results.items():
        print(f"{key}: {value}")



if __name__ == "__main__":
    main()