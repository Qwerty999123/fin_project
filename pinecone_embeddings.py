from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder # For BM25 sparse vectors
import numpy as np
import re
import hashlib
from typing import List, Dict
import os
import time

from sentence_transformers import SentenceTransformer

# Initialize Pinecone at the module level
# It's better to initialize it once at the start of your application


def get_index(pinecone: Pinecone, index_name: str):
    # 1. Delete existing index if it exists
    if index_name in pinecone.list_indexes().names():
        print(f"Deleting existing index: {index_name}")
        pinecone.delete_index(index_name)
        print(f"Index {index_name} deleted.")
    else:
        print(f"Index {index_name} does not exist, no deletion necessary.")

    # 2. Create fresh index using create_index_for_model for integrated embedding
    print(f"Creating new index: {index_name} with integrated 'llama-text-embed-v2'")
    pinecone.create_index( # Corrected from create_index_for_model
        name=index_name,
        metric="cosine", # llama-text-embed-v2 uses cosine or dotproduct
        dimension=768, # default dimension for llama-text-embed-v2
        # embed parameter should be at the top-level of create_index for integrated models
        # and 'field_map' is not used directly in create_index for embedded models
        # Instead, it's inferred from the text being passed in the 'upsert' method
        # We will specify the embedding model when upserting.
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index {index_name} created.")

    # 3. Wait for the index to be ready
    while not pinecone.describe_index(index_name).status['ready']:
        print("Waiting for index to be ready...")
        time.sleep(5)
    print(f"Index {index_name} is now ready.")

    return pinecone.Index(index_name)

# class EmbeddingEngine:
#     def __init__(self, model_name: str = 'BAAI/bge-base-en-v1.5'):
#         # For now, we are relying on Pinecone's integrated embedding for dense vectors.
#         # This class might be used for other purposes or for local embedding generation if needed later.
#         self.model = SentenceTransformer(model_name)
#         self.model.max_seq_length = 512

#     def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
#         prefixed_texts = [
#             f"query: {text}" if "query:" not in text.lower() else text
#             for text in texts
#         ]
#         return self.model.encode(prefixed_texts, batch_size=batch_size, convert_to_numpy=True)

class PineconeVectorStore:
    def __init__(self, index_name: str, pinecone: Pinecone, dimension: int = 1024): # default dimension for llama-text-embed-v2
        self.index_name = index_name
        self.dimension = dimension
        
        self.index = get_index(pinecone, index_name)
        
        # Initialize BM25 encoder for sparse vectors
        self.bm25_encoder = BM25Encoder()
        
        # Fit BM25 encoder on a representative corpus of your data.
        # This is crucial for BM25's effectiveness.
        # For this example, we'll fit on a small sample. In a real scenario,
        # you'd fit it on a larger corpus of your document chunks.
        print("Fitting BM25Encoder...")
        sample_corpus = ["This is a document about machine learning.", "Another document discussing natural language processing.", "A third document focused on artificial intelligence applications."]
        self.bm25_encoder.fit(sample_corpus) 
        print("BM25Encoder fitted.")

    def overwrite_vectors(self, document_chunks, pdf_filename: str, pinecone: Pinecone):
        """
        Completely replaces all vectors in the index with new data from a PDF.
        Leverages Pinecone's integrated embedding for dense vectors and BM25 for sparse.
        """
        # Ensure the index is recreated before processing each new PDF
        # self.index = get_index(pinecone, self.index_name) 

        inputs = [f"query: {text['text']}" for text in document_chunks]

        # embeddings = pinecone.inference.embed(
        #     model = 'llama-text-embed-v2',
        #     inputs = inputs,
        #     parameters={
        #         "input_type": "passage",
        #         "truncate": "END"
        #     }
        # )

        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        embeddings = model.encode(inputs, batch_size=32, convert_to_numpy=True).tolist()

        records_to_upsert = []
        for i, chunk_text in enumerate(document_chunks):
            # Ensure chunk_text is always a string before encoding

            doc_id = hashlib.md5(f"{pdf_filename}-{chunk_text['text']}".encode('utf-8')).hexdigest()
            sparse_vector = self.bm25_encoder.encode_documents([chunk_text["text"]])

            records_to_upsert.append({
                "id": doc_id,
                "values": embeddings[i],
                # "sparse_values": sparse_vector,
                "metadata": {"text": chunk_text['text'], "header": chunk_text['header'], "page": chunk_text['page'], "type": chunk_text['type']}
            })

        batch_size = 100
        for i in range(0, len(records_to_upsert), batch_size):
            batch = records_to_upsert[i:i + batch_size]
            self.index.upsert(
                vectors=batch,
                batch_size=batch_size
            )
        print(f"Successfully uploaded {len(records_to_upsert)} chunks from {pdf_filename} to Pinecone.")

    def retrieve_chunks(self, query_text: str, pinecone: Pinecone, top_k: int = 5):
        """
        Retrieves top-k chunks based on the query using hybrid search.
        """
        # Generate sparse vector for the query using BM25Encoder
        sparse_query_vector = self.bm25_encoder.encode_queries([query_text])

        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        embeddings = model.encode(f"query: {query_text}", batch_size=32, convert_to_numpy=True).tolist()

        query_results = self.index.query(
            vector=embeddings,
            # sparse_vector=sparse_query_vector, # Include the sparse vector for hybrid search
            top_k=top_k,
            include_metadata=True,
            include_values=False # No need to return the vectors themselves for RAG
        )

        retrieved_chunks = []
        for match in query_results['matches']:
            retrieved_chunks.append({
                "id": match['id'],
                "score": match['score'],
                "text": match['metadata']['text'],
                "header": match['metadata']['header'], 
                "page": match['metadata']['page'], 
                "type": match['metadata']['type']
            })
        return retrieved_chunks