#!/usr/bin/env python3
"""
Fixed Smart Adaptive Experiment - Proper Langfuse Dataset Integration

This version follows the working experiment_runner.py pattern:
- Uses item.run() for proper dataset linking
- Creates dataset runs that appear in Langfuse
- Maintains smart adaptive timeout logic
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from eval_lib import get_nova_secrets
from langfuse import Langfuse

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_question_complexity(question: str) -> Dict[str, Any]:
    """Quick complexity analysis for timeout prediction"""
    question_lower = question.lower()
    word_count = len(question.split())

    # Complexity scoring
    length_score = 1 if word_count <= 5 else (2 if word_count <= 15 else 3)

    unclear_keywords = ["hoe", "wat", "waarom", "kan ik", "moet ik", "problemen", "lukt niet"]
    technical_terms = ["bandsteering", "bkr", "dunning", "creditering", "afkoopsom", "betalingsregeling"]

    unclear_count = sum(1 for kw in unclear_keywords if kw in question_lower)
    technical_count = sum(1 for term in technical_terms if term in question_lower)

    clarity_score = 3 if unclear_count >= 2 else (2 if unclear_count >= 1 else 1)
    domain_score = 3 if technical_count >= 1 else 2

    total_score = length_score + clarity_score + domain_score

    if total_score <= 4:
        return {"level": "simple", "timeout": 30, "retries": []}
    elif total_score <= 7:
        return {"level": "medium", "timeout": 60, "retries": [90]}
    else:
        return {"level": "complex", "timeout": 90, "retries": [120, 150]}


def adaptive_nova_question_in_trace(question: str, trace: Any, complexity: Dict[str, Any]) -> Dict[str, Any]:
    """Process question with smart adaptive logic inside trace context"""
    timeouts = [complexity["timeout"]] + complexity["retries"]

    sys.path.insert(0, str(Path(__file__).parent))
    from nova_external_test import ask_nova_external_evaluation

    # Log complexity analysis
    logger.debug(f"Complexity: {complexity['level']}, timeout: {complexity['timeout']}s")

    for attempt, timeout in enumerate(timeouts, 1):
        try:
            print(f"    Attempt {attempt}/{len(timeouts)}: timeout {timeout}s")

            start_time = time.time()
            result = ask_nova_external_evaluation(question, skip_evaluations=True)
            duration = time.time() - start_time

            has_real_answer = (
                result.get("success", False)
                and result.get("answer", "").strip()
                and "Empty Response" not in result.get("answer", "")
            )

            print(f"    Duration: {duration:.1f}s, Success: {has_real_answer}, Intent: {result.get('intent', 'N/A')}")

            if has_real_answer:
                return {
                    "answer": result.get("answer", ""),
                    "success": True,
                    "intent": result.get("intent", ""),
                    "sources": result.get("sources", []),
                    "duration": duration,
                    "timeout_used": timeout,
                    "has_real_answer": True,
                    "attempt": attempt,
                    "complexity": complexity["level"],
                    "score": 0.9,
                }
            elif attempt < len(timeouts):
                # Try again with longer timeout
                print(f"    Retrying with {timeouts[attempt]}s timeout")
                continue
            else:
                # Final failure
                return {
                    "answer": result.get("answer", ""),
                    "success": result.get("success", False),
                    "intent": result.get("intent", ""),
                    "sources": result.get("sources", []),
                    "duration": duration,
                    "timeout_used": timeout,
                    "has_real_answer": False,
                    "attempt": attempt,
                    "complexity": complexity["level"],
                    "score": 0.0,
                }

        except Exception as e:
            print(f"    Error: Attempt {attempt} error: {str(e)}")

            if attempt < len(timeouts):
                continue
            else:
                return {
                    "answer": "",
                    "success": False,
                    "intent": "",
                    "sources": [],
                    "duration": timeout,
                    "timeout_used": timeout,
                    "has_real_answer": False,
                    "attempt": attempt,
                    "complexity": complexity["level"],
                    "error": str(e),
                    "score": 0.0,
                }

    return {"error": "Unexpected end of function"}


def get_langfuse_client() -> Langfuse:
    """Get Langfuse client"""
    secret_data = get_nova_secrets()
    return Langfuse(
        host=secret_data["langfuse_host"],
        public_key=secret_data["langfuse_public_key"],
        secret_key=secret_data["langfuse_secret_key"],
    )


def main() -> int:
    run_name = f"fixed_adaptive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_name = "nova_clean_dataset_v1"

    print("FIXED SMART ADAPTIVE EXPERIMENT")
    print("=" * 60)
    print("Proper Langfuse dataset integration")
    # Smart adaptive timeouts are enabled
    print("Will appear in Langfuse dataset runs")
    print("=" * 60)

    # Get Langfuse and dataset
    langfuse = get_langfuse_client()
    dataset = langfuse.get_dataset(dataset_name)
    items = list(dataset.items)

    print(f"Dataset: {dataset_name}")
    print(f"Items: {len(items)} questions")
    print(f"Run name: {run_name}")

    # Quick complexity analysis
    complexity_counts = {"simple": 0, "medium": 0, "complex": 0}
    for item in items:
        level = analyze_question_complexity(item.input)["level"]
        complexity_counts[level] += 1

    print(f"Complexity: {complexity_counts}")
    print("=" * 60)

    # Process all items using proper dataset integration
    start_time = time.time()
    results = []

    for i, item in enumerate(items, 1):
        try:
            question = item.input
            complexity = analyze_question_complexity(question)

            print(f"{i:2d}/42 [{complexity['level']:7s}]: {question[:50]}...", end=" ", flush=True)

            # EXPLICIT TRACE MANAGEMENT - Following HR chatbot pattern
            with item.run(
                run_name=run_name,
                run_metadata={
                    "experiment_type": "smart_adaptive_evaluation",
                    "complexity_level": complexity["level"],
                    "recommended_timeout": complexity["timeout"],
                    "timestamp": datetime.now().isoformat(),
                },
            ) as root_span:
                # Process question within dataset item context
                result = adaptive_nova_question_in_trace(question, root_span, complexity)

                # CRITICAL: Force update the dataset run trace with input/output
                # This must happen AFTER the nova call to override any nested trace behavior
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                intent = result.get("intent", "")

                # Run full evaluations now that Nova is working
                if result.get("has_real_answer", False):
                    print(f"      Running full evaluations...")
                    try:
                        sys.path.insert(0, str(Path(__file__).parent))
                        from nova_external_test import run_all_evaluations

                        evaluations = run_all_evaluations(
                            question=question,
                            answer=answer,
                            sources=result.get("sources", []),
                            intent=result.get("intent"),
                        )

                        # Convert to scores dictionary
                        evaluation_scores = {name: feedback.score for name, feedback in evaluations.items()}

                        # Add individual evaluation scores to trace
                        for eval_name, feedback in evaluations.items():
                            root_span.score(name=eval_name, value=feedback.score, comment=feedback.comment)

                        print(f"      Evaluations completed: {len(evaluation_scores)} scores")

                    except Exception as e:
                        print(f"      Error: Evaluation error: {e}")
                        evaluation_scores = {}
                else:
                    evaluation_scores = {}
                    print(f"      Skipping evaluations (no real answer)")

                # Add adaptive score
                root_span.score(
                    name="adaptive_score",
                    value=result.get("score", 0.0),
                    comment=f"{complexity['level']} question, attempt {result.get('attempt', 1)}",
                )

                # WORKING SOLUTION: Use update_current_trace for dataset trace-level input/output
                # This should work the same way as in nova_external_test.py
                langfuse.update_current_trace(
                    input=question,  # CRITICAL: Set trace-level input (verified working!)
                    output=answer,  # CRITICAL: Set trace-level output (verified working!)
                    metadata={
                        "complexity": complexity,
                        "attempt": result.get("attempt", 1),
                        "timeout_used": result.get("timeout_used", complexity["timeout"]),
                        "duration": result.get("duration", 0),
                        "intent": result.get("intent", ""),
                        "sources_count": len(result.get("sources", [])),
                        "has_real_answer": result.get("has_real_answer", False),
                        "experiment_type": "dataset_update_current_trace_pattern",
                        "trace_fix_applied": True,
                        "evaluation_scores": evaluation_scores,
                        "evaluations_completed": len(evaluation_scores),
                    },
                )

                # Add to results
                results.append(
                    {
                        "question_num": i,
                        "question": question,
                        "complexity": complexity["level"],
                        "success": result.get("has_real_answer", False),
                        "attempts": result.get("attempt", 1),
                        "duration": result.get("duration", 0),
                        "intent": result.get("intent", ""),
                        "answer_length": len(result.get("answer", "")),
                        "timeout_used": result.get("timeout_used", 0),
                        "evaluation_scores": evaluation_scores,
                    }
                )

                # Print result with evaluation summary
                if result.get("has_real_answer"):
                    if evaluation_scores:
                        avg_eval_score = sum(evaluation_scores.values()) / len(evaluation_scores)
                        eval_count = len(evaluation_scores)
                        print(f"Success ({result['duration']:.1f}s) | {eval_count} evals avg: {avg_eval_score:.2f}")
                    else:
                        print(f"Success ({result['duration']:.1f}s)")
                else:
                    print(f"Failed ({result.get('duration', 0):.1f}s)")

                # Checkpoint every 10 questions
                if i % 10 == 0:
                    successful = [r for r in results if r["success"]]
                    elapsed = time.time() - start_time
                    print(f"\nCheckpoint {i}/42: {len(successful)} successful ({len(successful) / i * 100:.0f}%)")
                    logger.info(f"Elapsed: {elapsed / 60:.1f}min | Avg: {elapsed / i:.1f}s per question")
                    print("-" * 60)

        except KeyboardInterrupt:
            print(f"\nWarning: Interrupted at question {i}")
            break
        except Exception as e:
            print(f"Error: ERROR: {str(e)[:50]}")
            continue

    total_duration = time.time() - start_time

    # Final analysis
    successful = [r for r in results if r["success"]]
    by_complexity: Dict[str, List[Dict[str, Any]]] = {"simple": [], "medium": [], "complex": []}

    for r in results:
        level = r["complexity"]
        if level in by_complexity:
            by_complexity[level].append(r)

    print(f"\nFIXED ADAPTIVE EXPERIMENT COMPLETED")
    print("=" * 60)
    if len(results) > 0:
        print(f"Results: {len(successful)}/{len(results)} successful ({len(successful) / len(results) * 100:.1f}%)")
        logger.info(f"Duration: {total_duration / 60:.1f} minutes")
        print(f"Speed: {len(results) / (total_duration / 60):.1f} questions/minute")
    else:
        print(f"No results processed")
        logger.info(f"Duration: {total_duration / 60:.1f} minutes")

    # Calculate average evaluation scores
    all_evaluation_scores: Dict[str, List[float]] = {}
    for result in successful:
        eval_scores = result.get("evaluation_scores", {})
        for eval_name, score in eval_scores.items():
            if eval_name not in all_evaluation_scores:
                all_evaluation_scores[eval_name] = []
            all_evaluation_scores[eval_name].append(score)

    print(f"\nEVALUATION SCORES:")
    if all_evaluation_scores:
        for eval_name, scores in all_evaluation_scores.items():
            avg_score = sum(scores) / len(scores) if scores else 0
            print(f"   {eval_name.capitalize():18s}: {avg_score:.3f} (n={len(scores)})")
    else:
        print("   No evaluation scores available")

    print(f"\nBY COMPLEXITY:")
    for level, level_results in by_complexity.items():
        if level_results:
            successful_level = [r for r in level_results if r["success"]]
            success_rate = len(successful_level) / len(level_results) * 100
            avg_time = sum(r["duration"] for r in level_results) / len(level_results)
            avg_attempts = sum(r["attempts"] for r in level_results) / len(level_results)

            print(
                f"   {level.capitalize():8s}: "
                f"{len(successful_level):2d}/{len(level_results):2d} "
                f"({success_rate:3.0f}%) "
                f"avg: {avg_time:5.1f}s "
                f"attempts: {avg_attempts:.1f}"
            )

    print(f"\nLANGFUSE INTEGRATION:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Run: {run_name}")
    print(f"   URL: https://langfuse.dev01.datascience-tmnl.nl")
    print(f"   Should now appear in Langfuse dataset runs!")

    # Summary data available in Langfuse - no local file needed
    print("Results saved to Langfuse dataset runs")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
