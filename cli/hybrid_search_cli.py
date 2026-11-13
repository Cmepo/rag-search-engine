import argparse
from lib.hybrid_search import *
from lib.query_enhancement import query_enhancement

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("score", nargs="*", type=float, help="Scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighted search")
    weighted_search_parser.add_argument("query", type=str, help="Query of the search")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Alpha of hybrid search")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Limit of searches to display")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Reciprocal Rank Fusion seach")
    rrf_search_parser.add_argument("query", type=str, help="Query of the search")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="Weight parameter")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Limit of searches to display")
    rrf_search_parser.add_argument("--enhance", type=str, default='spell', help="Enhance query with the help of LLM")


    args = parser.parse_args()

    match args.command:
        case "normalize":
            if not args.score:
                return
            min_score = min(args.score)
            max_score = max(args.score)
            if min_score == max_score:
                print ('1.0')
                return
            for score in args.score:
                norm_score = (score - min_score) / (max_score - min_score)
                print(f"* {norm_score:.4f}") 

        case "weighted-search":
            documents = load_documents()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.weighted_search(args.query, alpha=args.alpha, limit=args.limit)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"  BM25: {result['bm25_normalized']:.3f}, Semantic: {result['semantic_normalized']:.3f}")
                # Print first sentence of content
                content = result.get('content') or result.get('overview', '')
                if content:
                    first_sentence = content.split('.')[0] + '.' if '.' in content else content[:200]
                    print(f"  {first_sentence.strip()}")
                print()

        case "rrf-search":
            documents = load_documents()
            hybrid_search = HybridSearch(documents)

            if args.enhance:
                enhanced_query = query_enhancement(args.query, method=args.enhance)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
                results = hybrid_search.rrf_search(enhanced_query, k=args.k, limit=args.limit)
            else:
                results = hybrid_search.rrf_search(args.query, k=args.k, limit=args.limit)

            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"  RRF Score: {result['total_rrf']:.3f}")
                print(f"  BM25 Rank: {result['bm25_rrf']:.3f}, Semantic Rank: {result['semantic_rrf']:.3f}")
                # Print first sentence of content
                content = result.get('content') or result.get('overview', '')
                if content:
                    first_sentence = content.split('.')[0] + '.' if '.' in content else content[:200]
                    print(f"  {first_sentence.strip()}")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()