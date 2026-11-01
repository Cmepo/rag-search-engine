import string, json, pickle
from nltk.stem import PorterStemmer
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT_DIR/"cache"
DATA_DIR = ROOT_DIR/"data"/"movies.json"

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


def build_command():
    inverted_index = InvertedIndex()
    inverted_index.build(DATA_DIR)
    inverted_index.save()

####

class InvertedIndex:
    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = Counter()

    def __add_document(self, doc_id, text):
        tokenized_text = normalize_and_tokenize(text)
        for token in tokenized_text:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[token] += 1

    def get_documents(self, term):
        return sorted(self.index.get(term,set()))

    def get_tf(self, doc_id, term):
        pass 
    
    def build(self, path):
        movie_data = load_movie_data(path)
        for m in movie_data["movies"]:
            self.docmap[m["id"]] = m
            self.__add_document(m["id"], 
                                f"{m['title']} {m['description']}")

    def load(self):
        with (CACHE_DIR/"index.pkl").open("rb") as f:
            self.index = pickle.load(f)
        with(CACHE_DIR/"docmap.pkl").open("rb") as f:
            self.docmap = pickle.load(f)
        with(CACHE_DIR/"term_frequencies.pkl").open("rb") as f:
            self.term_frequencies = pickle.load(f)


    def save(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with (CACHE_DIR/"index.pkl").open("wb") as f:
            pickle.dump(self.index, f)
        with (CACHE_DIR/"docmap.pkl").open("wb") as f:
            pickle.dump(self.docmap, f)
        with(CACHE_DIR/"term_frequencies.pkl").open("wb") as f:
            pickle.dump(self.term_frequencies, f)
