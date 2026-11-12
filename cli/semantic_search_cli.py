import argparse
from lib.semantic_search import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the model")
    subparsers.add_parser("verify_embeddings", help="Verify the embeddings")
    subparsers.add_parser("embed_chunks", help="Load or build the chunk embeddings")
    
    embed_parser = subparsers.add_parser("embed_text", help="Embeds a sample text")
    embed_parser.add_argument("text", type=str, help="Text sample to embed")
    
    embed_query_parser = subparsers.add_parser("embed_query", help="Embeds a query")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search movies by semantic meaning")
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument("--limit", default=5, type=int, help="Limit of searches")

    chunking_parser = subparsers.add_parser("chunk", help="Fixed-sized chunking of document")
    chunking_parser.add_argument("text", type=str, help="Text to chunk")
    chunking_parser.add_argument("--chunk-size", default=200, type=int, help="Size of chunk")
    chunking_parser.add_argument("--overlap", default=0, type=int, help="Overlap of chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantic chunking of document")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", default=4, type=int, help="Max size of semantic chunk")
    semantic_chunk_parser.add_argument("--overlap", default=0, type=int, help="Overlap of semantic chunks")

    search_chunks_parser = subparsers.add_parser("search_chunked", help="Search for the highest scores of chunks")
    search_chunks_parser.add_argument("query", type=str, help="Query to search for")
    search_chunks_parser.add_argument("--limit", type=int, default=5, help="Limit output of search results")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, limit=args.limit)
        case "chunk":
            chunking(args.text, size=args.chunk_size, overlap=args.overlap)
        case "semantic_chunk":
            semantic_chunk(args.text, max_size=args.max_chunk_size, overlap=args.overlap)
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunks_command(args.query, limit=args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()