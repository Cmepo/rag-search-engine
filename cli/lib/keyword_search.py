import string, json, pickle
from nltk.stem import PorterStemmer
from pathlib import Path

def load_movie_data(path:Path):
    with path.open("r",encoding="utf-8") as f:
        movie_data = json.load(f)
    return movie_data

def normalize_and_tokenize(item:str) -> list[str]:
    lowered_item = item.lower()
    stamp = str.maketrans("","",string.punctuation)
    stamped = lowered_item.translate(stamp)
    tokenized = stamped.split()
    return tokenized

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
    pass

####

class InvertedIndex:
    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        tokenized_text = normalize_and_tokenize(text)
        for token in tokenized_text:
            self.index[token] = doc_id

    def get_documents(self, term):
        return sorted(self.index[term], reverse=True)

    def build(self,path):
        movie_data = load_movie_data(path)
        for movie in movie_data:
            self.docmap["doc_id"] = movie["id"]
            self.__add_document(movie["id"], 
                                f"{movie['title']} {movie['description']}")
            
    def save(self):
        pass
