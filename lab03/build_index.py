import ollama
import numpy as np
import json
import chunker
import time
from pathlib import Path

def main():

    # index_dir = Path(f"index_{time.strftime('%Y%m%d_%H%M%S')}")
    folder_index = 1
    while Path(f"index_{folder_index}").exists(): # increment folder index
        folder_index += 1
    index_dir = Path(f"index_{folder_index}")
    index_dir = Path("index_5")
    index_dir.mkdir(parents=True, exist_ok=True)

    models = ["mxbai-embed-large", "nomic-embed-text"]
    current_model = models[1] 

    chunker_params = {
        "target_chars": 1400,
        "max_chars": 1600,
        "min_words": 80,
        "min_chars": 1200,
        "overlap_sentences": 3}
    
    chunks = chunker.main(
        target_chars = chunker_params["target_chars"],
        max_chars = chunker_params["max_chars"],
        min_words = chunker_params["min_words"],
        min_chars = chunker_params["min_chars"],
        overlap_sentences = chunker_params["overlap_sentences"]
    )

    #save chunks
    chunks_path = index_dir / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk, f)  # Serialize the dictionary
            f.write("\n")        # Add a newline after each JSON object
    print(f"Saved {len(chunks)} chunks to {chunks_path}")

    vectors = []

    # Process with progress display
    for index, chunk in enumerate(chunks, start=1):
        response = ollama.embed(model=current_model, input=chunk["text"])
        vector = response["embeddings"][0]
        vectors.append(vector)
        print(f"Processed {index}/{len(chunks)} chunks", end="\r")

    # Save numpy array
    embeddings_path = index_dir / "embeddings.npy"
    matrix = np.array(vectors, dtype=np.float32)
    np.save(embeddings_path, matrix)
    print(f"Saved {len(vectors)} embeddings to {embeddings_path}")

    meta = {
        "model": current_model,
        "dim": len(vectors[0]),
        "normalize": False,
        "chunker_params": chunker_params
    }
    meta_path = index_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"Saved index metadata to {meta_path}")

if __name__ == "__main__":
    main()
