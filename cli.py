#!/usr/bin/env python3
"""
Single-command entry point.

Usage:
    python cli.py setup                    # generate corpus + build index
    python cli.py ask "your question"       # ask the assistant a question
    python cli.py eval                       # run RAGAS eval over the golden set
    python cli.py compare-models             # run the two-model comparison
"""
import sys
import subprocess


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "setup":
        subprocess.run([sys.executable, "scripts/generate_corpus.py"], check=True)
        from src.ingestion import build_index
        build_index()

    elif command == "ask":
        question = " ".join(sys.argv[2:])
        if not question:
            print("Usage: python cli.py ask \"your question\"")
            sys.exit(1)
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        result = pipeline.answer(question)
        print(result.model_dump_json(indent=2))

    elif command == "eval":
        subprocess.run([sys.executable, "eval/run_eval.py"], check=True)

    elif command == "compare-models":
        subprocess.run([sys.executable, "eval/model_comparison.py"], check=True)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
