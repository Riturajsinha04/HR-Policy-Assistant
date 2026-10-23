"""
CLI interface for the HR Policy Assistant.
Run: python -m src.cli
"""

import logging
import sys

from src.rag_chain import ask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Interactive terminal Q&A loop."""
    print("=" * 60)
    print("  Enso Group HR Policy Assistant (CLI)")
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    print()

    while True:
        try:
            question = input("❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("🔍 Searching the HR Manual...")
        try:
            result = ask(question)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        print()
        if result["found"]:
            print(f"✅ {result['answer']}")
        else:
            print(f"⚠️  {result['answer']}")

        print(f"⏱️  ({result['latency_s']}s)")
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
