import argparse, json
from pathlib import Path
from lib.keyword_search import *

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

json_path = DATA_DIR/"movies.json"
stopword_path = DATA_DIR/"stopwords.txt"

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    args = parser.parse_args()

    stopwords = load_stop_words(stopword_path)

    match args.command:
        case "search":
            inverted_index = InvertedIndex()
            try:
                inverted_index.load()
            except FileNotFoundError:
                parser.exit(
                    status=1,
                    message="Error: Index not found. Run 'build' first.\n"
                )

            query_tokens = normalize_and_tokenize(args.query)   
            results = set()

            print(f"Searching for: {args.query}")   

            for token in query_tokens:
                doc_ids = inverted_index.get_documents(token)

                for doc_id in doc_ids:
                    if doc_id not in results:
                        results.add(doc_id)
                        movie = inverted_index.docmap[doc_id]
                        print(f"{len(results)}. {movie["title"]} (ID: {doc_id})")
                        if len(results) >= 5:
                            break
                if len(results) >= 5:
                    break

        case "build":
            build_command()
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()