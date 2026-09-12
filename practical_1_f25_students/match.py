import numpy as np

# === STUDENT INSTRUCTIONS =======================================================
# match.py: 1-Nearest Neighbor (1-NN) Search with Text Embeddings
#
# You will implement nearest_neighbor(q, embeddings) to find, for each query
# vector, the SINGLE most similar NumPy function using cosine similarity
# (which equals the dot product here since all vectors are unit length).
#
# We provide:
#   - A fixed list of 10 function symbols
#   - A 10×10 matrix of function embeddings
#   - A  5×10 matrix of query embeddings
#   - An "answers" list for grading (used by main())
# 
# Note: You do NOT need to import ollama or call an LLM for this problem.
# 
# You may use plain Python loops or NumPy. You may break ties any way you like.
# Feel free to add any helper functions you need, but don't modify the symbols,
# query texts, embeddEMBEDDINGSings, or answers.
# ================================================================================

# The 10 symbols (order matters: answers refer to these indices!)
symbols = [
    "np.array",        # 0
    "np.arange",       # 1
    "np.linspace",     # 2
    "np.reshape",      # 3
    "np.mean",         # 4
    "np.std",          # 5
    "np.dot",          # 6
    "np.concatenate",  # 7
    "np.sum",          # 8
    "np.transpose",    # 9
]

# QUERY TEXTS (for reference only, not used in your computations)
query_texts = [
    "reshape a 2-D array into one dimension",    # -> np.reshape (3)
    "join two arrays together",                  # -> np.concatenate (7)
    "evenly spaced numbers between 0 and 10",    # -> np.arange (1)
    "dot product of two vectors",                # -> np.dot (6)
    "make a numpy array from a list",            # -> np.array (0)
]

# FUNCTION EMBEDDINGS for `symbols` (10×10, already normalized)
embeddings = [
  [-0.142, -0.303, -0.227, -0.231, -0.166, -0.513,  0.168,  0.325, -0.582, -0.142],  # 0. np.array
  [-0.257, -0.061,  0.044, -0.463, -0.170, -0.619, -0.142, -0.008, -0.524, -0.078],  # 1. np.arange
   [0.098, -0.196,  0.079, -0.554, -0.367, -0.430, -0.288,  0.197, -0.436, -0.083],  # 2. np.linspace
  [-0.303, -0.054, -0.365,  0.100, -0.085, -0.566,  0.284,  0.136, -0.554, -0.170],  # 3. np.reshape
   [0.042,  0.261, -0.079, -0.088, -0.161,  0.131, -0.294,  0.076, -0.336, -0.817],  # 4. np.mean
  [-0.141,  0.045, -0.032,  0.273, -0.442, -0.094, -0.379,  0.103, -0.273, -0.685],  # 5. np.std
  [-0.008,  0.275, -0.155, -0.225, -0.386, -0.209,  0.410,  0.168, -0.591, -0.333],  # 6. np.dot
   [0.245, -0.301, -0.161, -0.052, -0.381, -0.434, -0.148,  0.094, -0.395, -0.548],  # 7. np.concatenate
   [0.008,  0.510, -0.239,  0.130,  0.019, -0.446,  0.546,  0.054, -0.116,  0.390],  # 8. np.sum
  [-0.196, -0.160, -0.088, -0.197,  0.003, -0.419, -0.060,  0.307, -0.598, -0.509],  # 9. np.transpose
]

# QUERY EMBEDDINGS for `query_texts` (5×10, already normalized)
queries = [
  [-0.051, -0.085, -0.175,  0.030, -0.162, -0.676,  0.072,  0.178, -0.653, -0.108],  # "reshape a 2-D array ..."
  [-0.054,  0.006, -0.161, -0.217, -0.396, -0.654,  0.131,  0.073, -0.271, -0.493],  # "join two arrays ..."
  [-0.368,  0.257, -0.515, -0.504,  0.150, -0.007, -0.103,  0.152, -0.417,  0.221],  # "evenly spaced numbers ..."
   [0.182,  0.083, -0.222, -0.133, -0.340, -0.477,  0.593,  0.041, -0.432, -0.101],  # "dot product of two ..."
  [-0.169, -0.114, -0.018, -0.122, -0.228, -0.558,  0.072,  0.225, -0.672, -0.267],  # make a numpy array ..."
]

# Correct INDEX for each query ce o(maps into `symbols`)
answers = [3, 7, 1, 6, 0]

# ===== STUDENT TODO ======================================================
def nearest_neighbor(q, embeddings) -> int:
    """
    Return the index (0–9) of the most similar embedding to query vector q.
    """
    query_vector = np.array(q,dtype=np.float32)
    embeddings_vectors = np.array(embeddings,dtype=np.float32)
    scores = embeddings_vectors @ query_vector
    top_indexes = np.argsort(-scores)
    return top_indexes[0]
    
# =========================================================================


def main():
    # This is mostly to test your code, but you are free to modify it for debugging purposes
    correct = 0
    for i, q in enumerate(queries):
        predicted_index = nearest_neighbor(q, embeddings)
        true_index = answers[i]
        predicted_symbol = symbols[predicted_index]
        true_symbol = symbols[true_index]
        print(f"Query {i}: Pred = {predicted_symbol}, True = {true_symbol}", end=" -> ")
        if predicted_index == true_index:
            print("Correct")
            correct += 1
        else:
            print("Incorrect")

    hit1 = correct / len(queries)
    print(f"\nHit@1: {hit1:.2f}")

    
if __name__ == "__main__":
    main()
