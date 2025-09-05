#!/usr/bin/env python3
"""
Script to programmatically add questions to Langfuse dataset.
This allows batch uploading of evaluation questions without using the UI.

Usage:
    uv run python evaluation/add_questions_to_langfuse.py
"""

import json
import os
from typing import Any, Dict, List, Optional

from langfuse import Langfuse


def add_questions_to_dataset(
    dataset_name: str = "nova_clean_dataset_v1",
    questions: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Add questions to a Langfuse dataset programmatically.

    Args:
        dataset_name: Name of the dataset to update
        questions: List of question dictionaries with input, expected_output, and metadata
    """
    # Initialize Langfuse client (uses environment variables for auth)
    langfuse = Langfuse()

    # Default example questions if none provided
    if questions is None:
        questions = [
            {
                "input": {"question": "Hoe kan ik mijn factuur downloaden?"},
                "expected_output": "Log in op Mijn Odido en ga naar factuursectie",
                "metadata": {
                    "intent": "factuur uitleg",
                    "knowledge_base": "thuis_sf"
                }
            },
            {
                "input": {"question": "Wat kost roamen in Europa?"},
                "expected_output": "EU roaming is gratis binnen je bundel",
                "metadata": {
                    "intent": "mobiel",
                    "knowledge_base": "mobiel_community"
                }
            }
        ]

    print(f"Connecting to Langfuse and accessing dataset: {dataset_name}")

    # Get or create dataset
    try:
        dataset = langfuse.get_dataset(dataset_name)
        print(f"Found existing dataset: {dataset_name}")
    except Exception:
        # Create new dataset if it doesn't exist
        dataset = langfuse.create_dataset(
            name=dataset_name,
            description="Nova evaluation questions for testing AI assistant performance"
        )
        print(f"Created new dataset: {dataset_name}")

    # Add items to dataset
    success_count = 0
    for idx, question_data in enumerate(questions):
        try:
            # Create unique item ID based on question content
            item_id = f"question_{hash(question_data['input']['question']) % 10000}"

            dataset.upsert(
                item_id=item_id,
                input=question_data["input"],
                expected_output=question_data["expected_output"],
                metadata=question_data.get("metadata", {})
            )
            success_count += 1
            print(f"  [{idx + 1}/{len(questions)}] Added: {question_data['input']['question'][:50]}...")

        except Exception as e:
            print(f"  [{idx + 1}/{len(questions)}] Failed to add question: {e}")

    print(f"\nSummary: Successfully added {success_count}/{len(questions)} questions to dataset '{dataset_name}'")
    print(f"View dataset at: https://langfuse.dev01.datascience-tmnl.nl/datasets/{dataset_name}")


def load_questions_from_json(filename: str) -> List[Dict[str, Any]]:
    """Load questions from a JSON file."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to expected format if needed
    formatted_questions = []
    for item in data:
        formatted_question = {
            "input": {"question": item["question"]},
            "expected_output": item.get("expected_answer", ""),
            "metadata": {
                "intent": item.get("intent", ""),
                "knowledge_base": item.get("knowledge_base", "")
            }
        }
        formatted_questions.append(formatted_question)

    return formatted_questions


def main() -> None:
    """Main function to run the script."""
    import argparse

    parser = argparse.ArgumentParser(description="Add questions to Langfuse dataset")
    parser.add_argument(
        "--dataset",
        default="nova_clean_dataset_v1",
        help="Dataset name (default: nova_clean_dataset_v1)"
    )
    parser.add_argument(
        "--json-file",
        help="JSON file with questions to import"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Use example questions (default if no JSON file provided)"
    )

    args = parser.parse_args()

    # Load questions
    if args.json_file:
        print(f"Loading questions from: {args.json_file}")
        questions = load_questions_from_json(args.json_file)
    else:
        print("Using example questions (use --json-file to load from file)")
        questions = None  # Will use default examples

    # Add to Langfuse
    add_questions_to_dataset(
        dataset_name=args.dataset,
        questions=questions
    )


if __name__ == "__main__":
    main()
