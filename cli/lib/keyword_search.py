import string, json, pickle, math, os
from nltk.stem import PorterStemmer
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT_DIR/"cache"
MOVIES_PATH = ROOT_DIR/"data"/"movies.json"
STOPWORD_PATH = ROOT_DIR/"data"/"stopwords.txt"

BM25_K1 = 1.5
BM25_B = 0.75


stemmer = PorterStemmer()

def load_movie_data(path:Path):
    with path.open("r",encoding="utf-8") as f:
        movie_data = json.load(f)
    return movie_data

def normalize_and_tokenize(item:str) -> list[str]:
    lowered_item = item.lower()
    stamp = str.maketrans("","",string.punctuation)
    stamped = lowered_item.translate(stamp)
    tokenized = stamped.split()
    stemmed_tokens = []
    for token in tokenized:
        stemmed_token = stemmer.stem(token)
        stemmed_tokens.append(stemmed_token)
    return stemmed_tokens

def is_in_query(query:str, title:str, stopwords:list[str]) -> bool:
    # Legacy
    stemmer = PorterStemmer()
    query_tokens = normalize_and_tokenize(query)
    title_tokens = normalize_and_tokenize(title)

    query_tokens = [
        token for token in query_tokens
        if token not in stopwords
    ]
    for query_token in query_tokens:
        for title_token in title_tokens:
            if stemmer.stem(query_token) == stemmer.stem(title_token):
                return True
    return False

def load_stop_words(stopword_path:Path) -> list[str]:
    with stopword_path.open("r", encoding="utf-8") as f:
        stopwords = f.read()
    norm_stopwords_list = normalize_and_tokenize(stopwords)
    return norm_stopwords_list

def tokenize_single_term(term:str) -> str:
    tokens = normalize_and_tokenize(term)
    if len(tokens) != 1:
        raise Exception ('Exception (tokenize_single_term): Not a single token')
    return tokens[0]

def build_command() -> None:
    inverted_index = InvertedIndex(stopwords=load_stop_words(STOPWORD_PATH))
    inverted_index.build(MOVIES_PATH)
    inverted_index.save()

def bm25_idf_command(term:str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    tokenized_term = tokenize_single_term(term)
    bm25_idf = inverted_index.get_bm25_idf(tokenized_term)
    return bm25_idf

def bm25_tf_command(doc_id, term, k1=None, b=None) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    tokenized_term = tokenize_single_term(term)
    bm25_tf = inverted_index.get_bm25_tf(doc_id, tokenized_term)
    return bm25_tf

def bm25_command(query, limit=5):
    inverted_index = InvertedIndex()
    inverted_index.load()
    results = inverted_index.bm25_search(query, limit)
    formatted_results = []
    for rank, (doc_id, score) in enumerate(results, start=1):
        title = inverted_index.docmap[doc_id]['title']
        formatted_results.append((rank, doc_id, title, score))
    return formatted_results
####

class InvertedIndex:
    def __init__(self, stopwords=None) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = Counter()
        self.stopwords = set(stopwords) if stopwords else set()
        self.doc_lengths = {}
        self.doc_lengths_path = CACHE_DIR/"doc_lengths.pkl"

    def __add_document(self, doc_id:int, text:str) -> None:
        tokenized_text = normalize_and_tokenize(text)
        self.doc_lengths[doc_id] = 0
        for token in tokenized_text:
            if token in self.stopwords:
                continue
            self.doc_lengths[doc_id] += 1
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[token] += 1

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        else:
            total_tokens = sum(self.doc_lengths.values())
            return total_tokens/len(self.doc_lengths)

    def get_documents(self, term:str|int) -> list[str] | set:
        return sorted(self.index.get(term,set()))

    def get_tf(self, doc_id, term):
        movie = self.docmap[doc_id]
        doc_text = f"{movie['title']} {movie['description']}"
        tokens = normalize_and_tokenize(doc_text)
        return tokens.count(term)

    def get_bm25_idf(self, term:str) -> float:
        N = len(self.docmap)
        df = len(self.index.get(term,set()))
        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length == 0:
            length_norm = 1
        else:
            length_norm = 1 - b + b * (doc_length/avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id, term) -> float:
        tf_score = self.get_bm25_tf(doc_id, term)
        idf_score = self.get_bm25_idf(term)
        return tf_score * idf_score

    def bm25_search(self, query, limit):
        query_tokens = normalize_and_tokenize(query)
        query_tokens = [t for t in query_tokens if t not in self.stopwords]
        scores = {doc_id: 0.0 for doc_id in self.docmap}
        for token in query_tokens:
            doc_ids = self.get_documents(token)
            for doc_id in doc_ids:
                scores[doc_id] += self.bm25(doc_id, token)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:limit]

    def build(self, path:Path) -> None:
        movie_data = load_movie_data(path)
        for m in movie_data["movies"]:
            self.docmap[m["id"]] = m
            self.__add_document(m['id'], 
                                f"{m['title']} {m['description']}")

    def load(self) -> None:
        with (CACHE_DIR/"index.pkl").open("rb") as f:
            self.index = pickle.load(f)
        with(CACHE_DIR/"docmap.pkl").open("rb") as f:
            self.docmap = pickle.load(f)
        with(CACHE_DIR/"term_frequencies.pkl").open("rb") as f:
            self.term_frequencies = pickle.load(f)
        with(Path(self.doc_lengths_path)).open("rb") as f:
            self.doc_lengths = pickle.load(f)


    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with (CACHE_DIR/"index.pkl").open("wb") as f:
            pickle.dump(self.index, f)
        with (CACHE_DIR/"docmap.pkl").open("wb") as f:
            pickle.dump(self.docmap, f)
        with(CACHE_DIR/"term_frequencies.pkl").open("wb") as f:
            pickle.dump(self.term_frequencies, f)
        with(Path(self.doc_lengths_path)).open("wb") as f:
            pickle.dump(self.doc_lengths, f)
