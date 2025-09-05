#!/usr/bin/env python3
"""
Simple script to create a clean dataset in Langfuse
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

from eval_lib import get_nova_secrets
from langfuse import Langfuse

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_langfuse_client() -> Langfuse:
    """Get Langfuse client using AWS secrets"""
    secret_data = get_nova_secrets()
    return Langfuse(
        host=secret_data["langfuse_host"],
        public_key=secret_data["langfuse_public_key"],
        secret_key=secret_data["langfuse_secret_key"],
    )


def main() -> None:
    dataset_name = "nova_clean_dataset_v1"
    description = "Clean dataset with 42 high-quality test cases (no follow-ups, all have expected outputs)"

    logger.info("Creating Clean Dataset")
    logger.info("=" * 60)

    # Load cleaned test cases
    test_cases_path = Path(__file__).parent.parent / "tests" / "data" / "test_cases_cleaned.json"

    if not test_cases_path.exists():
        logger.error(f"Cleaned test cases not found at {test_cases_path}")
        logger.info("Please run clean_dataset.py first")
        return

    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    logger.info(f"Loaded {len(test_cases)} cleaned test cases")

    # Connect to Langfuse
    try:
        langfuse = get_langfuse_client()
        logger.info("Connected to Langfuse")
    except Exception as e:
        logger.error(f"Failed to connect to Langfuse: {e}")
        return

    # Create dataset
    try:
        langfuse.create_dataset(
            name=dataset_name,
            description=description,
            metadata={
                "created_at": datetime.now().isoformat(),
                "source": "test_cases_cleaned.json",
                "total_items": len(test_cases),
            },
        )
        logger.info(f"Created dataset: {dataset_name}")
    except Exception as e:
        logger.warning(f"Dataset might already exist: {e}")

    # Add items to dataset
    success_count = 0
    for i, test_case in enumerate(test_cases):
        try:
            metadata = {
                "intent": test_case.get("intent", "unknown"),
                "knowledge_base": test_case.get("knowledge_base", "general"),
                "expected_sources": test_case.get("expected_sources", []),
                "test_id": test_case.get("id", f"test_{i:03d}"),
            }

            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                input=test_case["question"],
                expected_output=test_case.get("expected_answer", ""),
                metadata=metadata,
            )
            success_count += 1

            if (i + 1) % 10 == 0:
                logger.info(f"   Added {i + 1}/{len(test_cases)} items...")

        except Exception as e:
            logger.error(f"Failed to add test case {i}: {e}")

    logger.info(f"\nSuccessfully created dataset '{dataset_name}' with {success_count} items")
    logger.info(f"View in Langfuse: https://langfuse.dev01.datascience-tmnl.nl")

    # Show statistics
    intents: Dict[str, int] = {}
    for case in test_cases:
        intent = case.get("intent", "unknown")
        intents[intent] = intents.get(intent, 0) + 1

    logger.info("\nDataset statistics:")
    for intent, count in sorted(intents.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"   {intent}: {count} cases ({count / len(test_cases) * 100:.1f}%)")


if __name__ == "__main__":
    main()
