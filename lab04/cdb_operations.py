import chromadb

def main():
    client = chromadb.PersistentClient(path="data/chroma")
    collection_names = [collection.name for collection in client.list_collections()]
    print(collection_names)

    # client.delete_collection(name="pandas_help_corpus")
    

if __name__ == "__main__":
    main()