# Mistral Vibe is working on this file - adding comment per user request
import argparse, math
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
    
    tf_parser = subparsers.add_parser("tf", help="Obtain term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to count")

    idf_parser = subparsers.add_parser("idf", help="Inverse Document Frequency")
    idf_parser.add_argument("term", type=str, help="Term to check")

    tf_idf_parser = subparsers.add_parser("tfidf", help="TF-IDF")
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term to check")

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

            #query_tokens = normalize_and_tokenize(args.query)
            query_tokens = [t for t in normalize_and_tokenize(args.query)
                            if t not in stopwords]   
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

        case "tf":
            inverted_index = InvertedIndex()
            try:
                inverted_index.load()
            except FileNotFoundError:
                parser.exit(
                    status=1,
                    message="Error: Index not found. Run 'build' first.\n"
                )
            term = tokenize_single_term(args.term)
            tf = inverted_index.get_tf(args.doc_id, term)
            print(tf)
        
        case "idf":
            inverted_index = InvertedIndex()
            try:
                inverted_index.load()
            except FileNotFoundError:
                parser.exit(
                    status=1,
                    message="Error: Index not found. Run 'build' first.\n"
                )
            term = tokenize_single_term(args.term)
            total_doc_count = len(inverted_index.docmap)
            term_match_doc_count = len(inverted_index.index[term])
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            inverted_index = InvertedIndex()
            try:
                inverted_index.load()
            except FileNotFoundError:
                parser.exit(
                    status=1,
                    message="Error: Index not found. Run 'build' first.\n"
                )
            term = tokenize_single_term(args.term)
            tf = inverted_index.get_tf(args.doc_id, term)
            idf = math.log((len(inverted_index.docmap)+1)/
                           (len(inverted_index.index[term])+1))
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()