import argparse, json
from pathlib import Path
from cli.lib.keyword_search import *

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
            movie_data = load_movie_data(json_path)
            for entry in movie_data["movies"]:
                if is_in_query(args.query, entry["title"],stopwords):
                    results.append(entry)
                    i+=1
                    if i < 6:
                        print(f"{i}. {entry["title"]}")
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()