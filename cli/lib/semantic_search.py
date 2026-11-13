import numpy as np
import json, re
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[2]
MOVIES_PATH = ROOT_DIR/"data"/"movies.json"
CACHE_DIR = ROOT_DIR/"cache"

def load_movie_data(path:Path):
    with path.open("r",encoding="utf-8") as f:
        movie_data = json.load(f)
    return movie_data

def verify_model():
    semantic_search = SemanticSearch()
    MODEL = semantic_search.model.tokenizer.name_or_path
    MAX_LENGTH = semantic_search.model.max_seq_length
    print(f"Model loaded: {MODEL}")
    print(f"Max sequence length: {MAX_LENGTH}")

def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    movie_data = load_movie_data(MOVIES_PATH)
    documents = movie_data["movies"]
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_search(query, limit=5):
    semantic_search = SemanticSearch()
    movie_data = load_movie_data(MOVIES_PATH)
    documents = movie_data['movies']
    semantic_search.load_or_create_embeddings(documents)
    search_result = semantic_search.search(query, limit)
    for i, result in enumerate(search_result, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"  {result['description']}\n")

def chunking(text, size=200, overlap=0):
    split_docs = text.split()
    chunks = []
    for i in range(0, len(split_docs), size-overlap):
        if (len(split_docs)) - i <= overlap:
            break
        chunk = ' '.join(split_docs[i:i+size])
        chunks.append(chunk)
    total_chars = len(text)
    print(f"Chunking {total_chars} characters")

    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")

def semantic_chunk(text:str, max_size=4, overlap=0):
    stripped_text = text.strip()
    if not stripped_text:
        return []
    split_docs = re.split(r"(?<=[.!?])\s+", stripped_text)
    if len(split_docs) == 1 and not split_docs[0].endswith((".","!","?")):
        return [stripped_text]
    split_docs = [s.strip() for s in split_docs if s.strip()]

    chunks = []
    for i in range(0, len(split_docs), max_size-overlap):
        if (len(split_docs)) - i <= overlap:
            break
        chunk = ' '.join(split_docs[i:i+max_size])
        chunks.append(chunk)

    #for i, chunk in enumerate(chunks, 1):
    #    print(f"{i}. {chunk}")

    return chunks

def embed_chunks():
    movie_data = load_movie_data(MOVIES_PATH)
    documents = movie_data['movies']
    semantic_search = ChunkedSemanticSearch()
    embeddings = semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunks_command(query, limit=5):
    movie_data = load_movie_data(MOVIES_PATH)
    documents = movie_data['movies']
    semantic_search = ChunkedSemanticSearch()
    semantic_search.load_or_create_chunk_embeddings(documents)
    search_results = semantic_search.search_chunks(query, limit)
    for idx, result in enumerate(search_results,1):
        print(f"\n{idx}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")

class SemanticSearch():
    def __init__(self, model_name: str="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if text == "":
            raise ValueError("Text is empty or only contains" \
            "whitespaces")
        sentences = [text]
        encode_result = self.model.encode(sentences)
        return encode_result[0]

    def build_embeddings(self, documents):
        self.documents = documents
        movie_list = []
        for doc in self.documents:
            self.document_map[doc.get("id")] = doc
            string_representation = f"{doc['title']}: {doc['description']}"
            movie_list.append(string_representation)
        self.embeddings = self.model.encode(movie_list, show_progress_bar=True)
        np.save(str(CACHE_DIR/"movie_embeddings.npy"), self.embeddings)
        return self.embeddings

    def search(self, query, limit):
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call 'load_or_create_embeddings' first.")
        query_embedding = self.generate_embedding(query)
        results = []
        for i, embedding in enumerate(self.embeddings):
            doc = self.documents[i]
            similarity_score = cosine_similarity(query_embedding, embedding)
            results.append((similarity_score, doc))
        results.sort(key=lambda x: x[0], reverse=True)
        return [{
            "score": score,
            "title": doc["title"],
            "description": doc["description"]
        } for score, doc in results[:limit]]

    def load_or_create_embeddings(self, documents):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        embeddings_file_path = CACHE_DIR/"movie_embeddings.npy"
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc.get("id")] = doc
        if embeddings_file_path.exists():
            self.embeddings = np.load(str(embeddings_file_path))
        if ((self.embeddings is not None) and (len(self.embeddings) == len(documents))):
            return self.embeddings
        else:
            return self.build_embeddings(documents)

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.documents = documents
        all_chunks = []
        chunk_metadata = []

        for doc_idx, doc in enumerate(documents):
            self.document_map[doc.get("id")] = doc
            description = doc.get('description','') 
            if not description.strip():
                continue
            doc_chunks = semantic_chunk(description, max_size=4, overlap=1)
            if doc_chunks is not None:
                for chunk_idx, chunk in enumerate(doc_chunks):
                    all_chunks.append(chunk)
                    chunk_metadata.append({
                        'movie_idx': doc_idx,
                        'chunk_idx': chunk_idx,
                        'total_chunks': len(doc_chunks)})
                    
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        np.save(str(CACHE_DIR/"chunk_embeddings.npy"), self.chunk_embeddings)
        with open(CACHE_DIR/"chunk_metadata.json","w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings


    def search_chunks(self, query:str, limit: int=10):
        chunk_scores = []
        query_embedding = self.generate_embedding(query)
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            return []
        for idx, chunk_embedding in enumerate(self.chunk_embeddings):
            cos_sim = cosine_similarity(chunk_embedding, query_embedding)
            chunk_scores.append({
                'chunk_idx': self.chunk_metadata[idx]['chunk_idx'],
                'movie_idx': self.chunk_metadata[idx]['movie_idx'],
                'score': cos_sim
            })

        movie_scores = {}
        for chunk in chunk_scores:
            movie_idx = chunk['movie_idx']
            score = chunk['score']
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score

        sorted_movie_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        results = []
        for movie_idx, score in sorted_movie_scores:
            doc = self.documents[movie_idx]
            results.append({
                'id': doc['id'],
                'title': doc['title'],
                'document': doc['description'][:100],
                'score': round(score, 3)
            })

        return results
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        embeddings_file_path = CACHE_DIR/"chunk_embeddings.npy"
        metadata_file_path  = CACHE_DIR/"chunk_metadata.json"
        self.documents = documents

        for doc in documents:
            self.document_map[doc.get("id")] = doc

        if (embeddings_file_path.exists() and metadata_file_path.exists()):
            self.chunk_embeddings = np.load(str(embeddings_file_path))
            with open(metadata_file_path, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data['chunks']
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)