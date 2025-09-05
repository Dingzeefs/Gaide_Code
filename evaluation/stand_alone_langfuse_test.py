#!/usr/bin/env python3
"""
Nova External Evaluation - Self-Contained Pattern

Following the external pattern from simple_nova_with_langfuse.py:
- No Nova module imports
- Self-contained evaluation logic
- Direct Azure OpenAI calls
- Works anywhere without Nova's venv

Based on: /home/work/work_liam/simple_nova_with_langfuse.py
Enhanced: Adds evaluation scoring using external pattern
"""

import json
import os
import ssl
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from langfuse import Langfuse
from openai import AzureOpenAI
from pydantic import BaseModel, Field

import websocket

# Import yaml for configuration loading
try:
    import yaml
except ModuleNotFoundError:
    yaml = None

# Same configuration as working test
AWS_REGION = "eu-central-1"
AWS_URL = "nova_test.dev01.datascience-tmnl.nl"
AWS_PEM_SECRET_ARN = (
    "arn:aws:secretsmanager:eu-central-1:116660942516:secret:ds-ecs-dev-nova-milvus-server-pem-certficate-vPwHDB"
)

USER_ID = "liam.van.vliet@odido.nl"
PASSWORD = "test123"


# Self-contained evaluation system (no Nova imports)
class EvaluationFeedback(BaseModel):
    """Evaluation feedback structure"""

    comment: str = Field(description="Concise feedback explaining the score")
    score: float = Field(description="Score between 0.0 and 1.0")


def load_evaluation_templates():
    """Load evaluation templates from YAML config"""

    # Load from configuration file
    try:
        import yaml
        config_path = Path(__file__).parent / "configuration" / "evaluator_prompts.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Extract evaluators from config
        evaluators = config.get("experiment", {}).get("evaluators", {})
        if not evaluators:
            raise ValueError("No evaluators found in configuration")
            
        print(f"Loaded {len(evaluators)} evaluation templates from {config_path}")
        return evaluators
        
    except ImportError:
        raise ImportError("YAML parser not available - install PyYAML")
    except Exception as e:
        raise Exception(f"Failed to load evaluation templates: {e}")


# Load evaluation templates (from config or inline)
EVALUATION_TEMPLATES = load_evaluation_templates()


def get_credentials():
    """Get credentials using hybrid approach - default chain first, then fallback"""
    
    # Try default credential chain first (for production/GitHub Actions/ECS)
    try:
        client = boto3.client("secretsmanager", region_name=AWS_REGION)
        
        # Test if we can actually access secrets
        client.describe_secret(SecretId="ds-secret-dev-nova-application-credentials")
        
        # If we get here, credentials work - proceed with simple approach
        print("Using default AWS credential chain")
        langfuse_response = client.get_secret_value(SecretId="ds-secret-dev-nova-application-credentials")
        langfuse_data = json.loads(langfuse_response["SecretString"])

        # Get PEM certificate
        pem_response = client.get_secret_value(SecretId=AWS_PEM_SECRET_ARN)
        pem_content = pem_response["SecretString"]

        return langfuse_data, pem_content
        
    except Exception as e:
        print(f"Default credential chain failed: {e}")
        print("Falling back to manual credential setup")
        
        # Fallback to manual credential setup for local development
        session = boto3.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            region_name=AWS_REGION,
        )
        client = session.client("secretsmanager")

        # Get Langfuse credentials
        langfuse_response = client.get_secret_value(SecretId="ds-secret-dev-nova-application-credentials")
        langfuse_data = json.loads(langfuse_response["SecretString"])

        # Get PEM certificate
        pem_response = client.get_secret_value(SecretId=AWS_PEM_SECRET_ARN)
        pem_content = pem_response["SecretString"]

        return langfuse_data, pem_content


def write_temp_pem(content: str) -> str:
    """Same PEM handling as working test"""
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp:
            tmp.write(content)
        os.chmod(path, 0o600)
    except Exception as e:
        os.remove(path)
        raise e
    return path


def cleanup_temp_pem(path: str) -> None:
    """Clean up temporary PEM file"""
    try:
        os.remove(path)
    except Exception as e:
        print(f"Warning: Could not cleanup temp file {path}: {e}")


def get_nova_token() -> str:
    """Get WebSocket auth token from secrets using hybrid approach"""
    # Try default credential chain first
    try:
        client = boto3.client("secretsmanager", region_name=AWS_REGION)
        client.describe_secret(SecretId="ds-secret-dev-nova-application-credentials")
        
        response = client.get_secret_value(SecretId="ds-secret-dev-nova-application-credentials")
        secret_data = json.loads(response["SecretString"])
        return secret_data["ws_auth_token"]
        
    except Exception:
        # Fallback to manual credentials
        session = boto3.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            region_name=AWS_REGION,
        )
        client = session.client("secretsmanager")
        response = client.get_secret_value(SecretId="ds-secret-dev-nova-application-credentials")
        secret_data = json.loads(response["SecretString"])
        return secret_data["ws_auth_token"]


def get_azure_openai_client():
    """Get Azure OpenAI client using credentials from AWS Secrets with fallback"""
    import os
    import boto3

    # Try default credential chain first
    try:
        client = boto3.client("secretsmanager", region_name=AWS_REGION)
        # Test access
        client.describe_secret(SecretId="ds-secret-dev-nova-azure-openai-credentials")
    except Exception:
        # Fallback to manual session
        session = boto3.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            region_name=AWS_REGION,
        )
        client = session.client("secretsmanager")

    # Try multiple Azure OpenAI secrets as fallback
    secret_ids = [
        "ds-secret-dev-nova-azure-openai-credentials",  # Primary
        "ds-secret-nonprod-azure-openai-credentials",  # Nonprod fallback
        "ds-secret-dev-azure-openai-credentials",  # Dev fallback
        "ds-secret-test-azure-openai-credentials",  # Test fallback
    ]

    last_error = None
    for secret_id in secret_ids:
        try:
            print(f"Trying Azure secret: {secret_id}")
            response = client.get_secret_value(SecretId=secret_id)
            secret_data = json.loads(response["SecretString"])

            # Create client and test it works
            azure_client = AzureOpenAI(
                api_key=secret_data.get("api_key", secret_data.get("azure_openai_api_key")),
                api_version=secret_data.get("api_version", secret_data.get("azure_openai_api_version", "2024-10-21")),
                azure_endpoint=secret_data.get("endpoint", secret_data.get("azure_openai_endpoint")),
            )

            print(f"Successfully created client with: {secret_id}")
            return azure_client

        except Exception as e:
            print(f"Error: Failed with {secret_id}: {str(e)[:50]}...")
            last_error = e
            continue

    # If all secrets failed, raise the last error
    raise Exception(f"All Azure OpenAI secrets failed. Last error: {last_error}")


def run_single_evaluation(
    eval_name: str, question: str, answer: str, context: str, azure_client: AzureOpenAI = None
) -> EvaluationFeedback:
    """Run a single evaluation using Azure OpenAI"""

    # Create fresh client if not provided (handles token expiration)
    if azure_client is None:
        try:
            azure_client = get_azure_openai_client()
        except Exception as e:
            return EvaluationFeedback(score=0.0, comment=f"Failed to create Azure client: {str(e)[:100]}")

    template = EVALUATION_TEMPLATES.get(eval_name)
    if not template:
        return EvaluationFeedback(score=0.0, comment=f"Template not found for {eval_name}")

    # Format the template with placeholders (if they exist)
    try:
        formatted_template = template.format(question=question, answer=answer, context=context)
    except KeyError:
        # If placeholders don't exist, use old structured approach
        formatted_template = template
        user_prompt = f"""<text>
        answer: {answer}
        question: {question}
        context: {context}
        </text>"""

    try:
        # Get Azure OpenAI deployment name from secrets using hybrid approach
        import os

        # Try default credential chain first
        try:
            client = boto3.client("secretsmanager", region_name=AWS_REGION)
            client.describe_secret(SecretId="ds-secret-dev-nova-azure-openai-credentials")
        except Exception:
            # Fallback to manual session
            session = boto3.Session(
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
                region_name=AWS_REGION,
            )
            client = session.client("secretsmanager")
            
        response = client.get_secret_value(SecretId="ds-secret-dev-nova-azure-openai-credentials")
        secret_data = json.loads(response["SecretString"])
        deployment = secret_data.get("deployment", secret_data.get("azure_openai_deployment"))

        # Make API call - use single message if template was formatted with placeholders
        if "user_prompt" in locals():
            # Old structured approach (fallback for templates without placeholders)
            messages = [
                {"role": "system", "content": formatted_template},
                {"role": "user", "content": user_prompt},
            ]
        else:
            # New approach with formatted template containing all data
            messages = [{"role": "user", "content": formatted_template}]

        completion = azure_client.chat.completions.create(
            model=deployment,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=512,
            temperature=0.0,
        )

        content = completion.choices[0].message.content
        if content is None:
            raise ValueError("Response content is None")

        # Parse JSON response
        response_data = json.loads(content)
        return EvaluationFeedback.model_validate(response_data)

    except Exception as e:
        return EvaluationFeedback(score=0.0, comment=f"Evaluation error: {str(e)}")


def run_all_evaluations(
    question: str, answer: str, sources: List[Dict[str, Any]], intent: Optional[str] = None
) -> Dict[str, EvaluationFeedback]:
    """Run all evaluations using external pattern (no Nova imports)"""

    print(f"Running external evaluations for: {question[:50]}...")

    # Get Azure OpenAI client with error handling
    try:
        azure_client = get_azure_openai_client()
        print(f"Azure OpenAI client created successfully")
    except Exception as e:
        print(f"Error: Failed to create Azure OpenAI client: {e}")
        # Return empty results if we can't create the client
        evaluation_names = ["retrieval_relevance", "accuracy", "hallucination"]
        return {
            name: EvaluationFeedback(score=0.0, comment=f"Client creation failed: {str(e)}")
            for name in evaluation_names
        }

    # Format sources as context
    if sources:
        context_lines = []
        for i, source in enumerate(sources, 1):
            if isinstance(source, dict):
                url = source.get("url", "Unknown URL")
                source_type = source.get("source", "Unknown source")
                context_lines.append(f"Source {i}: {source_type} - {url}")
            else:
                context_lines.append(f"Source {i}: {str(source)}")
        context = "\n".join(context_lines)
    else:
        context = "No sources retrieved"

    # Run all evaluations
    evaluation_names = ["retrieval_relevance", "accuracy", "hallucination"]

    results = {}

    for eval_name in evaluation_names:
        print(f"Running {eval_name}...")

        feedback = run_single_evaluation(
            eval_name=eval_name,
            question=question,
            answer=answer,
            context=context,
            # Don't pass azure_client - let each evaluation create a fresh one
        )

        results[eval_name] = feedback
        print(f"{eval_name}: {feedback.score:.2f} - {feedback.comment}")

    return results


def ask_nova_external_evaluation(question: str, conversation_id: str = None, skip_evaluations: bool = False) -> dict:
    """
    External evaluation pattern - no Nova imports, completely self-contained.

    This follows the same pattern as simple_nova_with_langfuse.py but adds evaluations.
    """

    if not conversation_id:
        conversation_id = f"external_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Nova External Evaluation: {question}")
    print(f"Conversation ID: {conversation_id}")

    try:
        # Get credentials (same as working test)
        langfuse_data, pem_content = get_credentials()
        pem_path = write_temp_pem(pem_content)

        # Initialize Langfuse
        langfuse = Langfuse(
            host=langfuse_data["langfuse_host"],
            public_key=langfuse_data["langfuse_public_key"],
            secret_key=langfuse_data["langfuse_secret_key"],
        )

        # Get Nova WebSocket token
        token = get_nova_token()

        # Create Langfuse trace
        with langfuse.start_as_current_span(
            name="stand_alone_test",
            input={
                "question": question,
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "evaluation_type": "external_pattern",
            },
        ) as span:
            print("Connecting to Nova WebSocket...")

            # Connect to Nova WebSocket (same as working test)
            ws = websocket.WebSocket(sslopt={"cert_reqs": ssl.CERT_NONE})
            ws.connect(f"wss://{AWS_URL}/ws/query/{USER_ID}?token={token}")

            print("Collecting Nova's response...")

            # Track if question has been sent (may need to wait for customer confirmation)
            question_sent = False

            # Collect Nova's response (same logic as working test)
            result = {
                "question": question,
                "conversation_id": conversation_id,
                "answer": "",
                "intent": None,
                "sources": [],
                "trace_id": span.trace_id,
                "success": False,
                "error": None,
                "evaluations": {},
                "evaluation_summary": {},
            }

            try:
                # Process Nova's WebSocket messages (same as working test)
                for i in range(50):
                    # Set timeout - shorter for first message to handle customer confirmation
                    if i == 0:
                        ws.settimeout(5.0)  # 5 seconds for first message
                    else:
                        ws.settimeout(90.0)  # 90 seconds per message

                    try:
                        response_text = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        if i == 0 and not question_sent:
                            # No customer confirmation needed, send question
                            print("   No customer confirmation needed, sending question...")
                            question_msg = {
                                "type": "question",
                                "payload": {"question": question, "conversation_id": conversation_id},
                            }
                            ws.send(json.dumps(question_msg))
                            question_sent = True
                            ws.settimeout(90.0)
                            response_text = ws.recv()

                    try:
                        response_data = json.loads(response_text)
                        msg_type = response_data.get("type", "unknown")

                        # Show abbreviated logs
                        if len(response_text) > 200:
                            print(f"    {i + 1:2d}. {msg_type}: {response_text[:80]}...")
                        else:
                            print(f"    {i + 1:2d}. {msg_type}: {response_text}")

                        # Handle customer confirmation requirement
                        if msg_type == "input_required":
                            payload = response_data.get("payload", {})
                            print(f"      DEBUG input_required payload type: {type(payload)}, value: {payload}")

                            # Check if payload is just the string "NOVA" (unclear intent)
                            if payload == "NOVA":
                                # Handle unclear intent confirmation with SMART INTENT PREDICTION (Option 3)
                                print(f"      Unclear intent detected, applying smart prediction...")

                                # Smart Intent Prediction based on question keywords
                                question_lower = question.lower()
                                predicted_intent = "onduidelijk"  # Default fallback

                                # Billing/Payment keywords -> Usually "thuis" related
                                billing_keywords = [
                                    "factuur",
                                    "betaling",
                                    "creditering",
                                    "incasso",
                                    "betalingsregeling",
                                    "afkoopsom",
                                    "dunning",
                                    "betaal",
                                    "rekening",
                                    "kosten",
                                    "prijs",
                                ]

                                # Mobile/Device keywords -> Usually "mobiel" related
                                mobile_keywords = [
                                    "sim",
                                    "toestel",
                                    "smartphone",
                                    "telefoon",
                                    "mobiel",
                                    "prepaid",
                                    "postpaid",
                                    "nummer",
                                    "bellen",
                                    "sms",
                                    "data",
                                    "roaming",
                                ]

                                # Technical/Home keywords -> Usually "thuis" related
                                home_keywords = [
                                    "wifi",
                                    "internet",
                                    "router",
                                    "modem",
                                    "tv",
                                    "mediabox",
                                    "thuis",
                                    "glasvezel",
                                    "kabel",
                                    "aansluiting",
                                    "installatie",
                                ]

                                if any(keyword in question_lower for keyword in billing_keywords):
                                    predicted_intent = "mobiel"
                                    print(f"      Smart prediction: BILLING -> mobiel")
                                elif any(keyword in question_lower for keyword in mobile_keywords):
                                    predicted_intent = "mobiel"
                                    print(f"      Smart prediction: MOBILE -> mobiel")
                                elif any(keyword in question_lower for keyword in home_keywords):
                                    predicted_intent = "thuis"
                                    print(f"      Smart prediction: HOME -> thuis")
                                else:
                                    # Default: try "mobiel" first (mobile services are more common)
                                    predicted_intent = "mobiel"
                                    print(f"      Smart prediction: DEFAULT -> mobiel")

                                intent_msg = {"type": "response", "response": predicted_intent}
                                ws.send(json.dumps(intent_msg))
                                print(f"      Sent smart intent: {predicted_intent}")
                                continue

                            # Ensure payload is always a dict for other cases
                            if isinstance(payload, str):
                                try:
                                    payload = json.loads(payload)
                                except json.JSONDecodeError:
                                    payload = {"raw_payload": payload}
                            if "customer_id" in payload:
                                customer_id = payload.get("customer_id")
                                print(f"      Customer confirmation required for: {customer_id}")
                                # Send confirmation using the exact protocol format
                                confirm_msg = {
                                    "type": "confirm_customer",
                                    "payload": {"customer_id": customer_id, "confirmed": True},
                                }
                                ws.send(json.dumps(confirm_msg))
                                print(f"      Sent confirmation: {json.dumps(confirm_msg)}")
                                print(f"      Success: Customer confirmation sent")

                                # Now send the question after confirmation
                                if not question_sent:
                                    print(f"      Sending question after confirmation...")
                                    question_msg = {
                                        "type": "question",
                                        "payload": {"question": question, "conversation_id": conversation_id},
                                    }
                                    ws.send(json.dumps(question_msg))
                                    question_sent = True
                                    print(f"      Success: Question sent: {question}")
                                continue
                            elif payload.get("status") == "not_found":
                                # Handle case where no customer is found - use test customer ID
                                print(f"      Warning: No customer found for user, using test customer ID")
                                test_customer_id = "1.10032346"  # Test customer ID from the documentation
                                confirm_msg = {
                                    "type": "confirm_customer",
                                    "payload": {"customer_id": test_customer_id, "confirmed": True},
                                }
                                ws.send(json.dumps(confirm_msg))
                                print(f"      Sent test customer confirmation: {json.dumps(confirm_msg)}")
                                print(f"      Success: Test customer confirmation sent")

                                # Now send the question after confirmation
                                if not question_sent:
                                    print(f"      Sending question after test customer confirmation...")
                                    question_msg = {
                                        "type": "question",
                                        "payload": {"question": question, "conversation_id": conversation_id},
                                    }
                                    ws.send(json.dumps(question_msg))
                                    question_sent = True
                                    print(f"      Success: Question sent: {question}")
                                continue

                        elif msg_type == "intent_detected":
                            result["intent"] = response_data.get("intent")

                        elif msg_type == "tool_result":
                            payload = response_data.get("payload", {})
                            if isinstance(payload, dict) and "tool_name" in payload:
                                tool_name = payload.get("tool_name")
                                if tool_name == "get_intent":
                                    output = payload.get("output", {})
                                    if isinstance(output, dict):
                                        intent_value = output.get("intent")
                                        result["intent"] = intent_value
                                        print(f"      Intent extracted: {result['intent']}")
                                    elif isinstance(output, str):
                                        try:
                                            import ast

                                            output_dict = ast.literal_eval(output)
                                            if isinstance(output_dict, dict):
                                                intent_value = output_dict.get("intent")
                                                result["intent"] = intent_value
                                                print(f"      Intent extracted: {intent_value}")
                                        except Exception as e:
                                            print(f"      Warning: Could not parse intent: {e}")

                        elif msg_type == "context_retrieved":
                            result["sources"] = response_data.get("sources", [])

                        elif msg_type == "response_streaming":
                            chunk = response_data.get("chunk", "")
                            result["answer"] += chunk

                        elif msg_type == "final_result":
                            payload = response_data.get("payload", {})
                            print(f"      Final result payload: {str(payload)[:200]}...")
                            if payload:
                                try:
                                    # Check if payload is already a dict
                                    if isinstance(payload, dict):
                                        final_data = payload
                                    else:
                                        # Try to parse as string
                                        import ast

                                        final_data = ast.literal_eval(str(payload))

                                    if isinstance(final_data, dict):
                                        final_answer = final_data.get("final_answer", "")
                                        final_sources = final_data.get("sources", [])

                                        result["answer"] = final_answer if final_answer else ""
                                        result["sources"] = final_sources if isinstance(final_sources, list) else []

                                        print(
                                            f"      Success: Answer: {'Success: Found' if final_answer else 'Error: Empty'}"
                                        )
                                        print(f"      Sources: {len(result['sources'])} found")
                                except Exception as e:
                                    print(f"      Warning: Parse error: {e}")
                                    result["answer"] = str(payload)[:500] if payload else ""
                            result["success"] = True
                            print(f"      Success: FINAL RESULT PROCESSED!")
                            break

                        elif msg_type == "response_complete":
                            final_answer = response_data.get("answer", "")
                            if final_answer:
                                result["answer"] = final_answer
                            result["success"] = True
                            break

                        elif msg_type == "error":
                            result["error"] = response_data.get("payload", "Unknown error")
                            break

                    except json.JSONDecodeError:
                        print(f"    {i + 1:2d}. plain_text: {response_text[:80]}...")
                        if len(response_text) > 10:
                            result["answer"] += response_text

            except websocket.WebSocketTimeoutException:
                print(f"      Timeout after receiving {i + 1} messages")
                result["error"] = f"Nova response timeout after {i + 1} messages"

            ws.close()

            print(f"\nNOVA WORKFLOW COMPLETED:")
            print(f"   Answer: {result['answer'][:100] if result['answer'] else 'None'}...")
            print(f"   Intent: {result['intent']}")
            print(f"   Sources: {len(result['sources'])}")
            print(f"   Success: {result['success']}")

            # Run External Evaluations (no Nova imports needed!)
            if result["success"] and result["answer"] and not skip_evaluations:
                print(f"\nRUNNING EXTERNAL EVALUATIONS...")

                try:
                    # Run evaluations using external pattern
                    evaluations = run_all_evaluations(
                        question=question, answer=result["answer"], sources=result["sources"], intent=result["intent"]
                    )

                    result["evaluations"] = evaluations

                    # Add evaluation scores to Langfuse trace
                    scores_added = 0
                    for eval_name, feedback in evaluations.items():
                        try:
                            langfuse.score_current_trace(name=eval_name, value=feedback.score, comment=feedback.comment)
                            scores_added += 1
                        except Exception as e:
                            print(f"   Error: Failed to add {eval_name} score: {e}")

                    # Create summary
                    scores = {name: feedback.score for name, feedback in evaluations.items()}
                    average_score = sum(scores.values()) / len(scores) if scores else 0.0
                    best_eval = max(scores.items(), key=lambda x: x[1]) if scores else ("None", 0.0)
                    worst_eval = min(scores.items(), key=lambda x: x[1]) if scores else ("None", 0.0)

                    result["evaluation_summary"] = {
                        "total_evaluations": len(evaluations),
                        "average_score": average_score,
                        "scores": scores,
                        "best_evaluation": best_eval,
                        "worst_evaluation": worst_eval,
                        "scores_added_to_trace": scores_added,
                    }

                    print(f"Success: EXTERNAL EVALUATIONS COMPLETED:")
                    print(f"   Evaluations run: {len(evaluations)}")
                    print(f"   Scores added to trace: {scores_added}")
                    print(f"   Average score: {average_score:.2f}")

                except Exception as eval_error:
                    print(f"Error: EXTERNAL EVALUATION ERROR: {eval_error}")
                    result["evaluation_error"] = str(eval_error)

                    # Add evaluation error to trace
                    langfuse.score_current_trace(
                        name="evaluation_error", value=0.0, comment=f"External evaluation failed: {str(eval_error)}"
                    )

            elif skip_evaluations:
                print(f"Warning: Skipping evaluations (skip_evaluations=True)")
                # Add basic quality score for successful answers
                if result["success"] and result["answer"]:
                    if result["intent"] or result["sources"]:
                        score = 0.8
                    else:
                        score = 0.6
                    langfuse.score_current_trace(name="nova_basic_quality", value=score)
            else:
                print(f"Warning: Skipping evaluations (no successful answer)")
                # Add basic quality score
                if result["intent"] or result["sources"]:
                    score = 0.5
                else:
                    score = 0.0
                langfuse.score_current_trace(name="nova_basic_quality", value=score)

            # WORKING SOLUTION: Use update_current_trace for trace-level input/output
            langfuse.update_current_trace(
                input=question,  # CRITICAL: Set trace-level input (verified working!)
                output=result["answer"],  # CRITICAL: Set trace-level output (verified working!)
                metadata={
                    "intent": result["intent"],
                    "sources_count": len(result["sources"]),
                    "sources": result["sources"],
                    "success": result["success"],
                    "error": result["error"],
                    "evaluations_completed": len(result.get("evaluations", {})),
                    "average_evaluation_score": result.get("evaluation_summary", {}).get("average_score", 0.0),
                    "evaluation_type": "update_current_trace_verified_working",
                    "conversation_id": conversation_id,
                },
            )

        # Cleanup
        os.remove(pem_path)
        langfuse.flush()

        # Print final results
        print(f"\n" + "=" * 60)
        print(f"EXTERNAL EVALUATION COMPLETED!")
        print(f"=" * 60)

        if result["success"]:
            print(f"Success: Answer: {result['answer'][:200]}...")
            print(f"Intent: {result['intent']}")
            print(f"Sources: {len(result['sources'])}")

            if result.get("evaluations"):
                print(f"Evaluations: {len(result['evaluations'])}")
                print(f"Average Score: {result['evaluation_summary']['average_score']:.2f}")
                print(
                    f"Best: {result['evaluation_summary']['best_evaluation'][0]} ({result['evaluation_summary']['best_evaluation'][1]:.2f})"
                )
                print(
                    f"Lowest: {result['evaluation_summary']['worst_evaluation'][0]} ({result['evaluation_summary']['worst_evaluation'][1]:.2f})"
                )
        else:
            print(f"Error: Error: {result['error']}")

        print(f"Langfuse trace: {langfuse_data['langfuse_host']}")
        print(f"Trace ID: {result['trace_id']}")
        print(f"Check Langfuse for external evaluation trace + scores!")
        print(f"=" * 60)

        return result

    except Exception as e:
        print(f"Error: Failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "question": question,
            "error": str(e),
            "success": False,
            "answer": "",
            "trace_id": None,
            "evaluations": {},
            "evaluation_summary": {},
        }


def run_external_evaluation_demo():
    """Run demo with external evaluation pattern"""
    print("Nova External Evaluation (No Nova Imports)")
    print("=" * 60)
    print("This demo uses the external pattern:")
    print("- No Nova module imports")
    print("- Self-contained evaluation logic")
    print("- Works anywhere without Nova's venv")
    print("- Same WebSocket pattern as working test")
    print("=" * 60)

    # Test questions
    questions = [
        "Wat kost een mobiel abonnement?",
    ]

    results = []

    for i, question in enumerate(questions, 1):
        print(f"\n" + "-" * 60)
        print(f"EXTERNAL QUESTION {i}/{len(questions)}: {question}")
        print("-" * 60)

        result = ask_nova_external_evaluation(question)
        results.append(result)

        if result["success"]:
            print(f"External evaluation success!")
        else:
            print(f"Warning:  External evaluation partial: {result['error']}")

    # Final summary
    successful_results = [r for r in results if r["success"]]

    print(f"\n" + "=" * 60)
    print(f"EXTERNAL EVALUATION SUMMARY")
    print(f"=" * 60)
    print(f"Questions processed: {len(results)}")
    print(f"Successful workflows: {len(successful_results)}")

    if successful_results:
        total_evaluations = sum(len(r.get("evaluations", {})) for r in successful_results)
        avg_scores = [r["evaluation_summary"]["average_score"] for r in successful_results if "evaluation_summary" in r]
        overall_avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0

        print(f"Total external evaluations: {total_evaluations}")
        print(f"Overall average score: {overall_avg:.2f}")
        print(f"\nSuccess: Check Langfuse dashboard for:")
        print(f"   - External evaluation traces")
        print(f"   - Self-contained scoring (no Nova deps)")
        print(f"   - Comparison with internal approach")

    print(f"=" * 60)
    return results


def main():
    """Main entry point for uv script execution"""
    return run_external_evaluation_demo()


# Example usage
if __name__ == "__main__":
    main()