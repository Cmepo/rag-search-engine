import argparse, json, string

# from config import DATA_DIR
# Doesn't work with bootdev

from pathlib import Path
from nltk.stem import PorterStemmer

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

json_path = DATA_DIR/"movies.json"
stopword_path = DATA_DIR/"stopwords.txt"

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    stopwords = load_stop_words(stopword_path)

    match args.command:
        case "search":
            results = []
            i=0
            print(f"Searching for: {args.query}")
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data["movies"]:
                if is_in_query(args.query, entry["title"],stopwords):
                    results.append(entry)
                    i+=1
                    if i < 6:
                        print(f"{i}. {entry["title"]}")
        case _:
            parser.print_help()

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
            if stemmer.stem(query_token) in stemmer.stem(title_token):
                return True
    return False

def load_stop_words(stopword_path:Path) -> list[str]:
    with stopword_path.open("r", encoding="utf-8") as f:
        stopwords = f.read()
    norm_stopwords_list = normalize_and_tokenize(stopwords)
    return norm_stopwords_list


if __name__ == "__main__":
    main()