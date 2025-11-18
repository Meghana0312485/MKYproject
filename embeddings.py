from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SemanticSearchEngine:
    def __init__(self):
        # Load embedding model once
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

    def create_index(self, chunks):
        """
        Creates a FAISS index from text chunks.
        """
        if not chunks or len(chunks) == 0:
            print("No chunks provided to create index.")
            return
        
        self.chunks = chunks

        # Generate embeddings
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        embeddings = embeddings.astype("float32").copy()

        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query, top_k=3):
        """
        Searches the FAISS index for the best matching chunks.
        """
        if self.index is None:
            print("Index is not created. Upload PDFs first.")
            return []

        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype("float32")

        # Search FAISS
        distances, indices = self.index.search(query_embedding, top_k)

        # Return matching text chunks
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
