#!/usr/bin/env python3
"""
Nova Experiment Runner with Multiple Data Source Options
Similar to the flexible pattern with chatbot/benchmark/csv options
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from langfuse import Langfuse

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Optional import - only needed for CSV functionality
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available - CSV functionality disabled")

import yaml

# Add evaluation folder to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from eval_lib import EvalTemplates, get_nova_secrets, run_complete_nova_evaluation_experiment


def get_langfuse_client() -> Langfuse:
    """Hybrid credential approach - works in both test and production environments"""

    # Try default credential chain first (for production/GitHub Actions/ECS)
    try:
        client = boto3.client('secretsmanager', region_name='eu-central-1')

        # Test if we can actually access secrets
        client.describe_secret(SecretId='ds-secret-dev-nova-application-credentials')

        # If we get here, credentials work - proceed with simple approach
        response = client.get_secret_value(SecretId='ds-secret-dev-nova-application-credentials')
        secret_data = json.loads(response['SecretString'])

        logger.info("Using default AWS credential chain for Langfuse client")
        return Langfuse(
            public_key=secret_data["langfuse_public_key"],
            secret_key=secret_data["langfuse_secret_key"],
            host=secret_data["langfuse_host"],
        )

    except Exception as e:
        logger.debug(f"Default credential chain failed: {e}")
        logger.info("Falling back to manual credential setup")

        # Fallback to original approach for local development
        secret_data = get_nova_secrets()
        return Langfuse(
            public_key=secret_data["langfuse_public_key"],
            secret_key=secret_data["langfuse_secret_key"],
            host=secret_data["langfuse_host"],
        )


def create_dataset_from_csv(csv_file: str, dataset_name: str, langfuse: Langfuse) -> Optional[str]:
    """Create Langfuse dataset from CSV file"""
    if not PANDAS_AVAILABLE:
        print("Error: pandas not available - cannot process CSV files")
        print("Install with: pip install pandas")
        return None

    logger.info(f"Loading CSV: {csv_file}")

    # Read CSV
    df = pd.read_csv(csv_file, delimiter=",")
    logger.info(f"Found {len(df)} rows in CSV")

    # Convert to Nova dataset format
    dataset_questions = []
    for _, row in df.iterrows():
        item = {
            "question": row.get("question", row.get("Question", "")),
            "expected_answer": row.get("expected_answer", row.get("Expected Answer", "")),
            "intent": row.get("intent", row.get("Intent", "onduidelijk")),
            "knowledge_base": row.get("knowledge_base", row.get("Knowledge Base", "general")),
            "answer_quality": row.get("answer_quality", row.get("Answer Quality", "good")),
            "expected_sources": str(row.get("expected_sources", row.get("Expected Sources", ""))).split("|")
            if row.get("expected_sources") or row.get("Expected Sources")
            else [],
            "follow_up": bool(row.get("follow_up", row.get("Follow Up", False))),
            "id": row.get("id", row.get("ID", f"csv_item_{_}")),
        }
        dataset_questions.append(item)

    # Create dataset in Langfuse
    try:
        langfuse.create_dataset(
            name=dataset_name,
            description=f"Dataset created from {csv_file} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            metadata={
                "source": csv_file,
                "created_at": datetime.now().isoformat(),
                "total_items": len(dataset_questions),
            },
        )
        print(f"Created dataset: {dataset_name}")
    except Exception as e:
        print(f"Warning: Dataset might already exist: {e}")

    # Add items to dataset
    success_count = 0
    for i, item in enumerate(dataset_questions):
        try:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                input=item["question"],
                expected_output=item["expected_answer"],
                metadata={
                    "intent": item["intent"],
                    "knowledge_base": item["knowledge_base"],
                    "answer_quality": item["answer_quality"],
                    "expected_sources": item["expected_sources"],
                    "follow_up": item["follow_up"],
                    "test_id": item["id"],
                },
            )
            success_count += 1

            if (i + 1) % 10 == 0:
                print(f"   Added {i + 1}/{len(dataset_questions)} items...")

        except Exception as e:
            print(f"Error: Failed to add item {i}: {e}")

    print(f"Successfully created dataset '{dataset_name}' with {success_count} items")
    return dataset_name


def create_dataset_from_json(json_file: str, dataset_name: str, langfuse: Langfuse) -> str:
    """Create Langfuse dataset from JSON file"""
    logger.info(f"Loading JSON: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        dataset_questions = json.load(f)

    logger.info(f"Found {len(dataset_questions)} items in JSON")

    # Create dataset in Langfuse
    try:
        langfuse.create_dataset(
            name=dataset_name,
            description=f"Dataset created from {json_file} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            metadata={
                "source": json_file,
                "created_at": datetime.now().isoformat(),
                "total_items": len(dataset_questions),
            },
        )
        print(f"Created dataset: {dataset_name}")
    except Exception as e:
        print(f"Warning: Dataset might already exist: {e}")

    # Add items to dataset
    success_count = 0
    for i, item in enumerate(dataset_questions):
        try:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                input=item["question"],
                expected_output=item.get("expected_answer", ""),
                metadata={
                    "intent": item.get("intent", "onduidelijk"),
                    "knowledge_base": item.get("knowledge_base", "general"),
                    "answer_quality": item.get("answer_quality", "good"),
                    "expected_sources": item.get("expected_sources", []),
                    "follow_up": item.get("follow_up", False),
                    "test_id": item.get("id", f"json_item_{i}"),
                },
            )
            success_count += 1

            if (i + 1) % 10 == 0:
                print(f"   Added {i + 1}/{len(dataset_questions)} items...")

        except Exception as e:
            print(f"Error: Failed to add item {i}: {e}")

    print(f"Successfully created dataset '{dataset_name}' with {success_count} items")
    return dataset_name


def create_benchmark_dataset(questions_list: List[Dict[str, Any]], dataset_name: str, langfuse: Langfuse) -> str:
    """Create benchmark dataset from a list of questions"""
    logger.info(f"Creating benchmark dataset with {len(questions_list)} questions")

    # Create dataset in Langfuse
    try:
        langfuse.create_dataset(
            name=dataset_name,
            description=f"Benchmark dataset created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            metadata={
                "source": "benchmark_creation",
                "created_at": datetime.now().isoformat(),
                "total_items": len(questions_list),
            },
        )
        print(f"Created benchmark dataset: {dataset_name}")
    except Exception as e:
        print(f"Warning: Dataset might already exist: {e}")

    # Add items to dataset
    success_count = 0
    for i, item in enumerate(questions_list):
        try:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                input=item["question"],
                expected_output=item.get("expected_answer", ""),
                metadata={
                    "intent": item.get("intent", "onduidelijk"),
                    "knowledge_base": item.get("knowledge_base", "general"),
                    "benchmark_type": item.get("benchmark_type", "standard"),
                    "difficulty": item.get("difficulty", "medium"),
                    "test_id": item.get("id", f"benchmark_{i}"),
                },
            )
            success_count += 1

        except Exception as e:
            print(f"Error: Failed to add benchmark item {i}: {e}")

    print(f"Successfully created benchmark dataset '{dataset_name}' with {success_count} items")
    return dataset_name


def main() -> None:
    """Main function with multiple options like the reference pattern"""
    logger.info("NOVA EXPERIMENT RUNNER - MULTIPLE OPTIONS")
    logger.info("=" * 60)

    # =====================================================================
    # CONFIGURATION PATHS (plug and play - change these as needed)
    # =====================================================================
    experiment_config_path = "configuration/experiment.yaml"
    eval_templates_config_path = "configuration/evaluator_prompts.yaml"

    logger.info(f"Experiment config: {experiment_config_path}")
    logger.info(f"Eval templates config: {eval_templates_config_path}")

    # Set up connection (Nova pattern with AWS Secrets)
    langfuse = get_langfuse_client()
    logger.debug("Connected to Langfuse")

    # Load evaluation templates
    eval_templates = EvalTemplates(config_path=eval_templates_config_path)
    logger.debug("Loaded evaluation templates")

    # Load experiment configuration for dataset name
    with open(experiment_config_path, "r") as f:
        experiment_config = yaml.safe_load(f)
    default_dataset_name = experiment_config["experiment"].get("dataset_name", "nova_clean_dataset_v1")
    logger.info(f"Default dataset: {default_dataset_name}")

    # =====================================================================
    # DATA SOURCE OPTIONS (uncomment the one you want to use)
    # =====================================================================

    # Option 1: Create dataset from CSV file (requires pandas)
    # Option 1: CSV dataset creation (commented out)
    if PANDAS_AVAILABLE:
        logger.debug("pandas available - CSV functionality enabled")
        # csv_file = "nova_questions.csv"  # Your CSV file
        # dataset_name = f"nova_csv_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # create_dataset_from_csv(csv_file, dataset_name, langfuse)
    else:
        logger.warning("pandas not available - skipping CSV option")

    # Option 2: Create dataset from JSON file
    # Option 2: JSON dataset creation (commented out)
    # json_file = "tests/data/test_cases_cleaned.json"  # Your JSON file
    # dataset_name = f"nova_json_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # create_dataset_from_json(json_file, dataset_name, langfuse)

    # Option 3: Create benchmark dataset from Python list (COMMENTED OUT)
    # Option 3: Benchmark dataset creation (disabled)
    # benchmark_questions = [
    #     {
    #         "question": "Hoe kan ik mijn mobiele abonnement opzeggen?",
    #         "expected_answer": "Je kunt je abonnement opzeggen via de Odido app of website.",
    #         "intent": "mobiel",
    #         "knowledge_base": "mobiel_sf",
    #         "benchmark_type": "cancellation",
    #         "difficulty": "easy",
    #         "id": "benchmark_001"
    #     },
    #     {
    #         "question": "Waarom werkt mijn internet thuis zo langzaam?",
    #         "expected_answer": "Langzaam internet kan verschillende oorzaken hebben zoals netwerkcongestie.",
    #         "intent": "thuis",
    #         "knowledge_base": "thuis_sf",
    #         "benchmark_type": "technical_issue",
    #         "difficulty": "medium",
    #         "id": "benchmark_002"
    #     },
    #     {
    #         "question": "Wat kost het om naar het buitenland te bellen?",
    #         "expected_answer": "De kosten voor internationale gesprekken variëren per land.",
    #         "intent": "mobiel",
    #         "knowledge_base": "mobiel_sf",
    #         "benchmark_type": "pricing",
    #         "difficulty": "easy",
    #         "id": "benchmark_003"
    #     }
    # ]
    #
    # dataset_name = f"nova_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # create_benchmark_dataset(benchmark_questions, dataset_name, langfuse)

    # =====================================================================
    # EXPERIMENT EXECUTION OPTIONS
    # =====================================================================

    logger.info("\nStarting experiment execution...")

    # Option A: Use Nova WebSocket (current working approach)
    # Running Nova WebSocket Experiment
    run_complete_nova_evaluation_experiment(
        experiment_config_path=experiment_config_path,
        eval_templates_config_path=eval_templates_config_path,
        dataset_source="nova_websocket",
    )

    # Option B: Use JSON file directly (uncomment to use)
    # Option B: JSON File Experiment (commented out)
    # run_complete_nova_evaluation_experiment(
    #     experiment_config_path=experiment_config_path,
    #     eval_templates_config_path=eval_templates_config_path,
    #     dataset_source="json",
    #     json_filename="tests/data/test_cases_cleaned.json"
    # )

    # Option C: Use custom dataset name (uncomment to use)
    # Option C: Custom Dataset Experiment (commented out)
    # # First update experiment.yaml to use your dataset_name, then:
    # run_complete_nova_evaluation_experiment(
    #     experiment_config_path=experiment_config_path,
    #     eval_templates_config_path=eval_templates_config_path,
    #     dataset_source="nova_websocket"
    # )

    logger.info("\nNova experiment completed!")
    logger.info("View results: https://langfuse.dev01.datascience-tmnl.nl")


if __name__ == "__main__":
    main()
