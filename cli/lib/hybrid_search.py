import os, json
from pathlib import Path
from .keyword_search import InvertedIndex, MOVIES_PATH, load_stop_words, STOPWORD_PATH
from .semantic_search import ChunkedSemanticSearch

ROOT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT_DIR/"cache"
MOVIES_PATH = ROOT_DIR/"data"/"movies.json"

def load_documents():
    with MOVIES_PATH.open("r",encoding="utf-8") as f:
        movie_data = json.load(f)
    return movie_data['movies']

def normalize_score(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex(stopwords=load_stop_words(STOPWORD_PATH))
        if not os.path.exists(self.idx.index_path):
            self.idx.build(MOVIES_PATH)
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        results = self.idx.bm25_search(query, limit)
        formatted = []
        for rank, (doc_id, score) in enumerate(results, 1):
            doc = self.idx.docmap[doc_id]
            formatted.append({
                'rank': rank,
                'id': doc_id,
                'title': doc['title'],
                'score': score
            })
        return formatted

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, 500*limit)
        semantic_results = self.semantic_search.search_chunks(query, 500*limit)
        norm_bm25_results = normalize_score([doc['score'] for doc in bm25_results])
        norm_semantic_results = normalize_score([doc['score'] for doc in semantic_results])

        doc_map = {}
        
        for i, doc in enumerate(bm25_results):
            doc_map[doc['id']] = {
                'doc': doc,
                'bm25_score': norm_bm25_results[i],
                'semantic_score': 0.0
            }

        for i, doc in enumerate(semantic_results):
            doc_id = doc['id']
            if doc_id in doc_map:
                doc_map[doc_id]['semantic_score'] = norm_semantic_results[i]
            else:
                doc_map[doc_id] = {
                    'doc': doc,
                    'bm25_score': 0.0,
                    'semantic_score': norm_semantic_results[i]
                }
        

        results = []
        for data in doc_map.values():
            hybrid = hybrid_score(data['bm25_score'], data['semantic_score'], alpha)
            results.append({
                **data['doc'],
                'bm25_normalized': data['bm25_score'],
                'semantic_normalized': data['semantic_score'],
                'hybrid_score': hybrid
            })

        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, 500 * limit)
        semantic_results = self.semantic_search.search_chunks(query, 500 * limit)

        doc_map = {}

        for doc in bm25_results:
            doc_id = doc['id']
            doc_map[doc_id] = {
                'doc': doc,
                'bm25_rrf': rrf_score(doc['rank'], k),
                'semantic_rrf': 0
            }

        for i, doc in enumerate(semantic_results, 1):
            doc_id = doc['id']
            if doc_id in doc_map:
                doc_map[doc_id]['semantic_rrf'] = rrf_score(i, k)
            else:
                doc_map[doc_id] = {
                    'doc': doc,
                    'bm25_rrf': 0,
                    'semantic_rrf': rrf_score(i, k)
                }

        results = []
        for data in doc_map.values():
            total_rrf = data['bm25_rrf'] + data['semantic_rrf']
            results.append({
                **data['doc'],
                'bm25_rrf': data['bm25_rrf'],
                'semantic_rrf': data['semantic_rrf'],
                'total_rrf': total_rrf
            })

        results.sort(key=lambda x: x['total_rrf'], reverse=True)
        return results[:limit]