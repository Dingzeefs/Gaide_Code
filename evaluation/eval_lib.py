#!/usr/bin/env python3
"""
Nova Experiment Library - UPDATED with working patterns from successful experiments
Includes fixes for: double traces, proper input/output, full evaluations, working WebSocket
"""

import json
import logging
import os
import ssl
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from langfuse import Langfuse
from openai import AzureOpenAI

import websocket

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# CRITICAL: Disable OpenInference auto-instrumentation to prevent double traces
# This is the key fix from our working experiments
os.environ["OPENINFERENCE_HIDE_INPUTS"] = "true"
os.environ["OPENINFERENCE_HIDE_OUTPUTS"] = "true"
os.environ["OPENINFERENCE_DISABLED"] = "true"

import yaml

# Global cache for AWS secrets to avoid multiple fetches
_cached_secrets = None

# Load configuration from YAML
def get_experiment_config() -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    import yaml
    with open("configuration/experiment.yaml", "r") as f:
        config = yaml.safe_load(f)
        return config if isinstance(config, dict) else {}

def get_nova_websocket_url() -> str:
    """Get Nova WebSocket URL from configuration."""
    config = get_experiment_config()
    url = config["connection"]["websocket_url"]
    return str(url)


def get_nova_secrets() -> Dict[str, Any]:
    """Get Nova secrets from AWS Secrets Manager with caching."""
    global _cached_secrets

    if _cached_secrets is not None:
        return _cached_secrets

    try:
        session = boto3.Session()
        client = session.client("secretsmanager", region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-central-1"))
        response = client.get_secret_value(SecretId="ds-secret-dev-nova-application-credentials")
        secret_data = json.loads(response["SecretString"])
        _cached_secrets = secret_data if isinstance(secret_data, dict) else {}
        return _cached_secrets
    except Exception as e:
        raise Exception(f"Failed to load Nova secrets: {e}")


class EvalTemplates:
    """Load evaluation templates from TOML configuration."""

    def __init__(self, config_path: str = "configuration/evaluator_prompts.yaml"):
        """Initialize with config file path."""
        self.config_path = config_path
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.templates = self.config["experiment"]["evaluators"]

    def get_template(self, template_name: str) -> Optional[str]:
        """Get a specific evaluation template by name."""
        template = self.templates.get(template_name)
        return str(template) if template is not None else None


def init_langfuse_from_secrets() -> Langfuse:
    """Initialize Langfuse client from AWS Secrets Manager."""
    try:
        secrets = get_nova_secrets()
        return Langfuse(
            public_key=secrets["langfuse_public_key"],
            secret_key=secrets["langfuse_secret_key"],
            host=secrets["langfuse_host"],
        )
    except Exception as e:
        raise Exception(f"Failed to initialize Langfuse: {e}")


def init_azure_client() -> AzureOpenAI:
    """Initialize Azure OpenAI client from AWS Secrets with multiple fallbacks."""
    import os

    # Force fresh credential loading from environment
    session = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-central-1"),
    )
    client = session.client("secretsmanager")

    # Try multiple secret IDs like the working version
    secret_ids = [
        "ds-secret-dev-nova-azure-openai-credentials",  # Primary
        "ds-secret-nonprod-azure-openai-credentials",  # Nonprod fallback
        "ds-secret-dev-azure-openai-credentials",  # Dev fallback
        "ds-secret-test-azure-openai-credentials",  # Test fallback
    ]

    last_error = None
    for secret_id in secret_ids:
        try:
            print(f"   Trying Azure secret: {secret_id}")
            response = client.get_secret_value(SecretId=secret_id)
            secret_data = json.loads(response["SecretString"])

            # Create client with flexible key names
            azure_client = AzureOpenAI(
                api_key=secret_data.get("api_key", secret_data.get("azure_openai_api_key")),
                api_version=secret_data.get("api_version", secret_data.get("azure_openai_api_version", "2024-10-21")),
                azure_endpoint=secret_data.get("endpoint", secret_data.get("azure_openai_endpoint")),
            )

            print(f"   Success: Successfully created client with: {secret_id}")
            return azure_client

        except Exception as e:
            print(f"   Error: Failed with {secret_id}: {str(e)[:50]}...")
            last_error = e
            continue

    raise Exception(f"Failed to load Azure OpenAI credentials from all sources. Last error: {last_error}")


def get_nova_connection_info() -> tuple[str, str]:
    """Get Nova WebSocket connection info using same config as working nova_external_test.py."""
    # Same configuration as working test
    AWS_URL = get_nova_websocket_url()
    nova_url = f"wss://{AWS_URL}"

    # We don't actually need the PEM content for this test setup
    # (nova_external_test.py uses sslopt={"cert_reqs": ssl.CERT_NONE})
    pem_content = ""

    return nova_url, pem_content


def get_nova_token() -> str:
    """Get WebSocket auth token from secrets (same as nova_external_test.py)"""
    secret_data = get_nova_secrets()
    token = secret_data["ws_auth_token"]
    return str(token)


def ask_nova_question_original(question: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Ask Nova a single question - using exact working code from nova_external_test.py"""
    conversation_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Use default user ID if not provided (same as nova_external_test.py)
    if user_id is None:
        user_id = "liam.van.vliet@odido.nl"

    # Initialize result structure
    result: Dict[str, Any] = {
        "question": question,
        "conversation_id": conversation_id,
        "answer": "",
        "intent": None,
        "sources": [],
        "success": False,
        "error": None,
    }

    try:
        # Get credentials (same as nova_external_test.py)
        secret_data = get_nova_secrets()
        ws_token = secret_data["ws_auth_token"]

        # Nova WebSocket URL (exactly as in nova_external_test.py)
        AWS_URL = get_nova_websocket_url()
        nova_url = f"wss://{AWS_URL}"

        print(f"Connecting to Nova WebSocket...")
        print(f"Collecting Nova's response...")

        # Connect to WebSocket (exactly as in nova_external_test.py)
        ws = websocket.WebSocket(sslopt={"cert_reqs": ssl.CERT_NONE})
        ws.connect(f"{nova_url}/ws/query/{user_id}?token={ws_token}")

        # Send initial question
        question_msg = {"question": question, "conversation_id": conversation_id}
        ws.send(json.dumps(question_msg))

        question_sent = False
        ws.settimeout(60.0)  # Reduced initial timeout
        message_count = 0

        # Process WebSocket messages with timeout optimization
        while message_count < 50:  # Reduced limit for faster experiments
            try:
                response_text = ws.recv()
                if not response_text:
                    break

                response_data = json.loads(response_text)
                msg_type = response_data.get("type", "unknown")
                message_count += 1

                if len(response_text) > 200:
                    print(f"    {message_count:2d}. {msg_type}: {response_text[:80]}...")
                else:
                    print(f"    {message_count:2d}. {msg_type}: {response_text}")

                # Handle customer confirmation requirement
                if msg_type == "input_required":
                    payload = response_data.get("payload", {})
                    print(f"      DEBUG input_required payload type: {type(payload)}, value: {payload}")

                    # Check if payload is just the string "NOVA" (unclear intent)
                    if payload == "NOVA":
                        # Handle unclear intent confirmation
                        logger.debug("Unclear intent detected, sending intent selection...")
                        intent_msg = {"response": {"intent": "thuis en mobiel", "customer_id": "1.19459294"}}
                        ws.send(json.dumps(intent_msg))
                        print(f"      Sent intent selection: {json.dumps(intent_msg)}")
                        continue

                    # Handle normal customer confirmation
                    if "customer_id" in payload:
                        customer_id = payload.get("customer_id")
                        print(f"      Customer confirmation required for: {customer_id}")
                        # Send confirmation using the exact protocol format
                        confirm_msg: Dict[str, Any] = {
                            "type": "confirm_customer",
                            "payload": {"customer_id": customer_id, "confirmed": True},
                        }
                        ws.send(json.dumps(confirm_msg))
                        print(f"      Sent confirmation: {json.dumps(confirm_msg)}")
                        print(f"      Success: Customer confirmation sent")

                        # Now send the question after confirmation
                        if not question_sent:
                            print(f"       Sending question after confirmation...")
                            question_after_confirm: Dict[str, Any] = {
                                "type": "question",
                                "payload": {"question": question, "conversation_id": conversation_id},
                            }
                            ws.send(json.dumps(question_after_confirm))
                            print(f"      Success: Question sent: {question[:50]}...")
                            question_sent = True
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
                            print(f"       Sending question after test customer confirmation...")
                            question_after_test_confirm: Dict[str, Any] = {
                                "type": "question",
                                "payload": {"question": question, "conversation_id": conversation_id},
                            }
                            ws.send(json.dumps(question_after_test_confirm))
                            print(f"      Success: Question sent: {question[:50]}...")
                            question_sent = True
                        continue

                # Collect tool results for intent
                elif msg_type == "tool_result":
                    tool_data = response_data.get("payload", {})
                    if tool_data.get("tool_name") == "get_intent":
                        arguments = tool_data.get("arguments", {})
                        intent = arguments.get("intent")
                        if intent:
                            result["intent"] = intent
                            print(f"      Intent extracted: {intent}")

                # Collect sources from context retrieval
                elif msg_type == "tool_result":
                    tool_data = response_data.get("payload", {})
                    if tool_data.get("tool_name") == "rag_retrieval":
                        retrieval_result = tool_data.get("result", {})
                        if isinstance(retrieval_result, dict) and "contexts" in retrieval_result:
                            contexts = retrieval_result["contexts"]
                            result["sources"] = [
                                ctx.get("source", "Unknown") for ctx in contexts if isinstance(ctx, dict)
                            ]

                # Handle errors from Nova
                elif msg_type == "error":
                    error_msg = response_data.get("payload", "Unknown error")
                    print(f"      Error: Nova error: {error_msg}")
                    result["error"] = error_msg
                    result["success"] = False
                    # Don't break here, let it continue to see if there are more messages

                # Final result with complete answer
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
                                
                                print(f"      Success: Answer: {'Success: Found' if final_answer else 'Error: Empty'}")
                                print(f"      Sources: {len(result['sources'])} found")
                        except Exception as e:
                            print(f"      Warning: Parse error: {e}")
                            final_answer = payload.get("final_answer", "") if isinstance(payload, dict) else ""
                            result["answer"] = final_answer
                            
                    if result["answer"]:
                        result["success"] = True
                        
                    print(f"      Success: FINAL RESULT PROCESSED!")
                    break

            except websocket.WebSocketTimeoutException:
                print(f"      Timeout after receiving {message_count} messages")
                result["error"] = f"Nova response timeout after {message_count} messages"
                break
            except Exception as e:
                print(f"      Error: Error during WebSocket communication: {e}")
                result["error"] = str(e)
                break

        ws.close()

        # If we didn't get a successful response, ensure error is set
        if not result["success"] and not result["error"]:
            result["error"] = "No valid response received from Nova"

    except Exception as e:
        print(f"      Error: Connection error: {e}")
        result["error"] = str(e)

    return result


def ask_nova_question(question: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Ask Nova via WebSocket with evaluations - self-contained implementation using evaluator_prompts.yaml templates"""
    conversation_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Use default user ID if not provided
    if user_id is None:
        user_id = "liam.van.vliet@odido.nl"

    # Initialize result structure
    result: Dict[str, Any] = {
        "question": question,
        "conversation_id": conversation_id,
        "answer": "",
        "intent": None,
        "sources": [],
        "success": False,
        "error": None,
        "evaluations": {},
    }

    try:
        # Get Nova credentials from AWS Secrets
        secret_data = get_nova_secrets()
        ws_token = secret_data["ws_auth_token"]

        # Nova WebSocket URL (working configuration)
        AWS_URL = get_nova_websocket_url()
        nova_url = f"wss://{AWS_URL}"

        print(f"Nova External Evaluation: {question}")
        print(f"Conversation ID: {conversation_id}")
        print(f"Connecting to Nova WebSocket...")
        print(f"Collecting Nova's response...")

        # Connect to WebSocket (working pattern)
        ws = websocket.WebSocket(sslopt={"cert_reqs": ssl.CERT_NONE})
        ws.connect(f"{nova_url}/ws/query/{user_id}?token={ws_token}")

        # DON'T send initial question - wait for customer confirmation like working pattern
        question_msg = {"question": question, "conversation_id": conversation_id}
        question_sent = False
        message_count = 0

        # Process WebSocket messages (same as working pattern)
        while message_count < 50:
            # Set timeout - shorter for first message to handle customer confirmation
            if message_count == 0:
                ws.settimeout(5.0)  # 5 seconds for first message
            else:
                ws.settimeout(90.0)  # 90 seconds per message

            try:
                response_text = ws.recv()
                if not response_text:
                    break
            except websocket.WebSocketTimeoutException:
                if message_count == 0 and not question_sent:
                    # No customer confirmation needed, send question
                    print("    No customer confirmation needed, sending question...")
                    question_msg_formatted = {
                        "type": "question",
                        "payload": {"question": question, "conversation_id": conversation_id},
                    }
                    ws.send(json.dumps(question_msg_formatted))
                    question_sent = True
                    ws.settimeout(90.0)
                    continue
                else:
                    result["error"] = f"Nova response timeout after {message_count} messages"
                    break
            except Exception as e:
                result["error"] = str(e)
                break

            # Process the received message
            response_data = json.loads(response_text)
            msg_type = response_data.get("type", "unknown")
            message_count += 1

            if len(response_text) > 200:
                print(f"    {message_count:2d}. {msg_type}: {response_text[:80]}...")
            else:
                print(f"    {message_count:2d}. {msg_type}: {response_text}")

            # Handle customer confirmation and intent prediction (same as working pattern)
            if msg_type == "input_required":
                payload = response_data.get("payload", {})

                if payload == "NOVA":
                    # Smart intent prediction
                    question_lower = question.lower()
                    billing_keywords = ["factuur", "betaling", "creditering", "incasso", "betalingsregeling"]
                    mobile_keywords = ["sim", "toestel", "smartphone", "telefoon", "mobiel", "prepaid"]
                    home_keywords = ["wifi", "internet", "router", "modem", "tv", "thuis"]

                    if any(keyword in question_lower for keyword in billing_keywords):
                        predicted_intent = "mobiel"
                    elif any(keyword in question_lower for keyword in mobile_keywords):
                        predicted_intent = "mobiel"
                    elif any(keyword in question_lower for keyword in home_keywords):
                        predicted_intent = "thuis"
                    else:
                        predicted_intent = "mobiel"

                    print(f"      Smart prediction: {predicted_intent}")
                    intent_msg = {"response": {"intent": predicted_intent, "customer_id": "1.19459294"}}
                    ws.send(json.dumps(intent_msg))
                    continue

                # Handle customer confirmation (same as working pattern)
                if "customer_id" in payload:
                    customer_id = payload.get("customer_id")
                    confirm_msg = {
                        "type": "confirm_customer",
                        "payload": {"customer_id": customer_id, "confirmed": True},
                    }
                    ws.send(json.dumps(confirm_msg))
                    if not question_sent:
                        question_after_confirm2: Dict[str, Any] = {
                            "type": "question",
                            "payload": {"question": question, "conversation_id": conversation_id},
                        }
                        ws.send(json.dumps(question_after_confirm2))
                        question_sent = True
                    continue

                elif payload.get("status") == "not_found":
                    print(f"      Warning: No customer found for user, using test customer ID")
                    test_customer_id = "1.10032346"
                    confirm_msg = {
                        "type": "confirm_customer",
                        "payload": {"customer_id": test_customer_id, "confirmed": True},
                    }
                    ws.send(json.dumps(confirm_msg))
                    print(f"      Success: Test customer confirmation sent")
                    if not question_sent:
                        print(f"       Sending question after test customer confirmation...")
                        question_after_test_confirm2: Dict[str, Any] = {
                            "type": "question",
                            "payload": {"question": question, "conversation_id": conversation_id},
                        }
                        ws.send(json.dumps(question_after_test_confirm2))
                        question_sent = True
                        print(f"      Success: Question sent: {question}")
                    continue

            # Extract intent and sources (same as working pattern)
            elif msg_type == "tool_result":
                tool_data = response_data.get("payload", {})
                if tool_data.get("tool_name") == "get_intent":
                    intent = tool_data.get("arguments", {}).get("intent")
                    if intent:
                        result["intent"] = intent
                elif tool_data.get("tool_name") == "rag_retrieval":
                    retrieval_result = tool_data.get("result", {})
                    if isinstance(retrieval_result, dict) and "contexts" in retrieval_result:
                        contexts = retrieval_result["contexts"]
                        result["sources"] = [ctx.get("source", "Unknown") for ctx in contexts if isinstance(ctx, dict)]

            # Handle final result
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
                            
                            print(f"      Success: Answer: {'Success: Found' if final_answer else 'Error: Empty'}")
                            print(f"      Sources: {len(result['sources'])} found")
                    except Exception as e:
                        print(f"      Warning: Parse error: {e}")
                        final_answer = payload.get("final_answer", "") if isinstance(payload, dict) else ""
                        result["answer"] = final_answer
                        
                if result["answer"]:
                    result["success"] = True

                # Run evaluations if we have a real answer
                if result["success"] and final_answer.strip() and "Empty Response" not in final_answer:
                    print(f"\nRUNNING EXTERNAL EVALUATIONS...")
                    try:
                        azure_client = get_azure_openai_client()  # type: ignore  # type: ignore
                        if azure_client:
                            result["evaluations"] = _run_evaluations_from_config(
                                question, final_answer, result["sources"], azure_client
                            )
                            print(f"Success: EXTERNAL EVALUATIONS COMPLETED:")
                            print(f"   Evaluations run: {len(result['evaluations'])}")
                    except Exception as e:
                        print(f"Warning: Evaluation error: {e}")
                break

        ws.close()

        if not result["success"] and not result["error"]:
            result["error"] = "No valid response received from Nova"

    except Exception as e:
        result["error"] = str(e)

    return result


def _run_evaluations_from_config(question: str, answer: str, sources: List[str], azure_client: Optional[AzureOpenAI]) -> Dict[str, Any]:
    """Run evaluations using templates from evaluator_prompts.yaml"""
    evaluations: Dict[str, Any] = {}

    # Load evaluation templates from evaluator_prompts.yaml
    try:
        eval_templates = EvalTemplates(config_path="configuration/evaluator_prompts.yaml")
        context_str = "\\n".join(str(source) for source in sources) if sources else "No context available"

        # Get the evaluation list from experiment.yaml
        experiment_config_path = "configuration/experiment.yaml"
        with open(experiment_config_path, "r") as f:
            experiment_config = yaml.safe_load(f)
        evaluation_names = experiment_config["experiment"]["evaluations"]

        # Run each evaluation from config
        for eval_name in evaluation_names:
            template = eval_templates.get_template(eval_name)
            if not template:
                print(f"   Warning: Template not found for {eval_name}")
                continue

            try:
                # Format the template with placeholders
                formatted_prompt = template.format(question=question, answer=answer, context=context_str)

                print(f"   Running {eval_name}...")

                # Get deployment name from secrets like the working version
                import os

                session = boto3.Session(
                    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                    aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
                    region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-central-1"),
                )
                client = session.client("secretsmanager")
                response = client.get_secret_value(SecretId="ds-secret-dev-nova-azure-openai-credentials")
                secret_data = json.loads(response["SecretString"])
                deployment = secret_data.get("deployment", secret_data.get("azure_openai_deployment"))

                if azure_client is None:
                    return {"error": "Azure client not available"}
                response = azure_client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=0,
                    response_format={"type": "json_object"},
                )

                eval_result = json.loads(response.choices[0].message.content)

                # Create evaluation result object
                evaluations[eval_name] = type(
                    "EvaluationResult",
                    (),
                    {
                        "score": float(eval_result.get("score", 0.0)),
                        "comment": eval_result.get("comment", "No comment provided"),
                    },
                )()

                print(
                    f"   Success: {eval_name}: {evaluations[eval_name].score:.2f} - {evaluations[eval_name].comment[:100]}..."
                )

            except Exception as e:
                print(f"   Error: Evaluation error for {eval_name}: {e}")
                evaluations[eval_name] = type(
                    "EvaluationResult", (), {"score": 0.0, "comment": f"Error during evaluation: {str(e)}"}
                )()

    except Exception as e:
        print(f"Error: Could not load evaluation templates: {e}")

    return evaluations


# Global Azure client to prevent double traces (CRITICAL FIX from working experiments)
_azure_client = None


def get_azure_openai_client() -> Optional[AzureOpenAI]:
    """Get singleton Azure OpenAI client to prevent double traces - WORKING PATTERN"""
    global _azure_client
    if _azure_client is None:
        try:
            _azure_client = init_azure_client()
            print("Success: Created singleton Azure client (prevents double traces)")
        except Exception as e:
            print(f"Warning: Could not initialize Azure client: {e}")
            return None
    return _azure_client


def evaluate_answer(
    question: str, answer: str, sources: List[str], eval_templates: EvalTemplates, experiment_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate an answer using configured evaluation templates - REUSING SINGLE AZURE CLIENT."""
    # CRITICAL: Get reused Azure client to prevent double traces
    azure_client = get_azure_openai_client()  # type: ignore
    if not azure_client:
        print("Warning: No Azure client available for evaluations")
        return {}

    evaluations: Dict[str, Any] = {}
    context_str = "\\n".join(sources) if sources else "No context available"

    # Run each evaluation from config
    for eval_name in experiment_config["experiment"]["evaluations"]:
        template = eval_templates.get_template(eval_name)
        if not template:
            continue

        # Format the template
        formatted_prompt = template.format(question=question, answer=answer, context=context_str)

        try:
            response = azure_client.chat.completions.create(
                model="nova-gpt-4o-v3",
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )

            eval_result = json.loads(response.choices[0].message.content)
            evaluations[eval_name] = {
                "score": float(eval_result.get("score", 0.0)),
                "comment": eval_result.get("comment", ""),
            }

        except Exception as e:
            print(f"Warning: Evaluation error for {eval_name}: {e}")
            evaluations[eval_name] = {"score": 0.0, "comment": f"Error: {str(e)}"}

    return evaluations


def _load_json_dataset(json_filename: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(json_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to simple format expected by the experiment runner
    items = []
    for item in data:
        items.append(
            {
                "input": item.get("question", ""),
                "expected_output": item.get("expected_answer", ""),
                "metadata": {
                    "intent": item.get("intent", ""),
                    "knowledge_base": item.get("knowledge_base", ""),
                    "expected_sources": item.get("expected_sources", []),
                    "answer_quality": item.get("answer_quality", ""),
                    "follow_up": item.get("follow_up", ""),
                    "id": item.get("id", ""),
                },
            }
        )

    return items


def run_nova_evaluation_experiment(
    langfuse: Langfuse,
    experiment_config: Dict[str, Any],
    eval_templates: EvalTemplates,
    dataset_source: str = "nova_websocket",
    json_filename: Optional[str] = None,
) -> None:
    """
    Run a complete Nova evaluation experiment.

    Following the pattern from eval_experiment.py but adapted for Nova's WebSocket interface.
    """
    dataset_name = experiment_config["experiment"]["dataset_name"]
    experiment_name = experiment_config["experiment"]["experiment_name"]
    run_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Starting experiment: {run_name}")
    print(f"Dataset source: {dataset_source}")

    # Get dataset items based on source
    if dataset_source == "json" and json_filename:
        # Load from JSON file
        print(f"Loading JSON dataset: {json_filename}")
        items = _load_json_dataset(json_filename)
        print(f"Found {len(items)} items in JSON dataset")

        # Run experiment without Langfuse dataset items (now synchronous)
        _run_json_experiment_sync(items, run_name, eval_templates, experiment_config, langfuse)

    else:
        # Use existing Langfuse dataset (default)
        print(f"Dataset: {dataset_name}")
        dataset = langfuse.get_dataset(dataset_name)
        items = list(dataset.items)
        print(f"Found {len(items)} items in dataset")

        # Run experiment (now synchronous)
        _run_experiment_sync(items, run_name, eval_templates, experiment_config)


def _run_experiment_sync(items: List[Any], run_name: str, eval_templates: EvalTemplates, experiment_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Synchronous wrapper for running the experiment."""
    results = []

    # No progress files - keep experiments clean and simple
    completed_questions = set()

    for i, item in enumerate(items):
        question_id = f"{i}_{hash(item.input)}"  # Create unique ID for each question

        # Skip if already completed
        if question_id in completed_questions:
            print(f"Skipping question {i + 1}/{len(items)}: Already completed")
            continue

        print(f"\\n{i + 1:2d}/{len(items)}: {item.input[:60]}...", end=" ", flush=True)

        try:
            # Create trace for this item - WORKING PATTERN from fixed_adaptive_experiment.py
            with item.run(
                run_name=run_name,
                run_metadata={
                    "experiment_type": "nova_evaluation_fixed",
                    "fixes_applied": ["working_websocket", "proper_trace_management", "full_evaluations"],
                },
            ) as trace:
                # Ask Nova using WORKING approach from nova_external_test.py
                start_time = time.time()
                result = ask_nova_question(item.input)
                duration = time.time() - start_time

                has_real_answer = (
                    result.get("success", False)
                    and result.get("answer", "").strip()
                    and "Empty Response" not in result.get("answer", "")
                )

                # Handle evaluations safely
                evaluations = result.get("evaluations", {})
                evaluation_scores = {}

                if evaluations:
                    try:
                        print(f"Evaluations completed: {len(evaluations)}", end=" ", flush=True)
                        # Add evaluation scores to trace (match working pattern)
                        for metric, eval_data in evaluations.items():
                            if hasattr(eval_data, "score") and hasattr(eval_data, "comment"):
                                trace.score(name=metric, value=eval_data.score, comment=eval_data.comment)
                                evaluation_scores[metric] = eval_data.score
                            elif isinstance(eval_data, dict):
                                trace.score(
                                    name=metric, value=eval_data.get("score", 0.0), comment=eval_data.get("comment", "")
                                )
                                evaluation_scores[metric] = eval_data.get("score", 0.0)
                    except Exception as e:
                        print(f"Error: ERROR: {e}", end=" ", flush=True)
                        evaluation_scores = {}
                else:
                    print(f"No evaluations available", end=" ", flush=True)

                # CRITICAL: Use proper input/output management - WORKING PATTERN
                # Get langfuse client for update_current_trace (like working version)
                langfuse = init_langfuse_from_secrets()
                langfuse.update_current_trace(
                    input=item.input,  # CRITICAL: Set trace-level input
                    output=result.get("answer", "")
                    if has_real_answer
                    else f"Error: {result.get('error', 'No answer')}",
                    metadata={
                        "intent": result.get("intent"),
                        "sources": result.get("sources", []),
                        "error": result.get("error"),
                        "success": has_real_answer,
                        "duration": duration,
                        "conversation_id": result.get("conversation_id", ""),
                        "has_real_answer": has_real_answer,
                        "evaluation_scores": evaluation_scores,
                    },
                )

                # Print result summary
                if has_real_answer and evaluations:
                    # Handle both object and dict evaluation formats
                    scores = []
                    for eval_data in evaluations.values():
                        if hasattr(eval_data, "score"):
                            scores.append(eval_data.score)
                        elif isinstance(eval_data, dict):
                            scores.append(eval_data.get("score", 0.0))

                    if scores:
                        avg_eval_score = sum(scores) / len(scores)
                        eval_count = len(scores)
                        print(f"Success: ({duration:.1f}s) | {eval_count} evals avg: {avg_eval_score:.2f}")
                    else:
                        print(f"Success: ({duration:.1f}s)")
                elif has_real_answer:
                    print(f"Success: ({duration:.1f}s)")
                else:
                    print(f"Error: ({duration:.1f}s) - {result.get('error', 'No answer')}")

                # Store results for summary - handle both evaluation formats
                eval_scores = {}
                for k, v in evaluations.items():
                    if hasattr(v, "score"):
                        eval_scores[k] = v.score
                    elif isinstance(v, dict):
                        eval_scores[k] = v.get("score", 0.0)
                    else:
                        eval_scores[k] = 0.0

                # Handle sources which might be dicts or strings
                sources = result.get("sources", [])
                if sources and isinstance(sources[0], dict):
                    sources_str = "|".join([s.get("source", str(s)) for s in sources])
                else:
                    sources_str = "|".join([str(s) for s in sources])

                results.append(
                    {
                        "question": item.input,
                        "answer": result.get("answer", ""),
                        "intent": result.get("intent"),
                        "sources": sources_str,
                        "error": result.get("error"),
                        "success": has_real_answer,
                        "duration": duration,
                        **eval_scores,
                    }
                )

                # Mark question as completed (in memory only - no files)
                completed_questions.add(question_id)

                # Progress checkpoints every 10 questions - WORKING PATTERN
                if (i + 1) % 10 == 0:
                    successful = [r for r in results if r["success"]]
                    total_time = time.time() - start_time
                    print(
                        f"\\nCheckpoint {i + 1}/{len(items)}: {len(successful)} successful ({len(successful) / (i + 1) * 100:.0f}%)"
                    )
                    print(f"   Progress: {(i + 1) / len(items) * 100:.1f}% complete")
                    print("-" * 60)
        except KeyboardInterrupt:
            print(f"\\nWarning: Experiment interrupted by user")
            break
        except Exception as e:
            print(f"Error: ERROR: {str(e)[:50]}")
            # Still mark as completed to avoid infinite retry
            completed_questions.add(question_id)
            continue

    # Summary and save results
    print(f"\\n{'=' * 60}")
    print(f"Experiment completed: {run_name}")
    print(f"Success: Successful: {sum(1 for r in results if r.get('success'))}/{len(results)}")

    # Calculate average scores
    for eval_name in experiment_config["experiment"]["evaluations"]:
        scores = [r.get(eval_name, 0) for r in results if r.get(eval_name) is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"Average {eval_name}: {avg_score:.2f}")

    # Results available in Langfuse - no local CSV needed
    print("Results saved to Langfuse dataset runs")

    return results


def run_complete_nova_evaluation_experiment(
    experiment_config_path: str,
    eval_templates_config_path: str,
    dataset_source: str = "nova_websocket",
    nova_mode: str = "websocket",
    json_filename: Optional[str] = None,
) -> None:
    """
    Complete Nova evaluation experiment pipeline: load configs, retrieve dataset, and run experiment.

    This function handles the entire Nova evaluation workflow in a single call:
    1. Loads experiment and evaluation template configurations
    2. Initializes Langfuse connection from AWS Secrets
    3. Retrieves test dataset from specified source
    4. Runs the Nova evaluation experiment

    Args:
        experiment_config_path: Path to the experiment configuration TOML file
        eval_templates_config_path: Path to the evaluation templates configuration TOML file
        dataset_source: Source of test dataset. Options: "nova_websocket", "json"
        nova_mode: Mode for Nova dataset retrieval. Options: "websocket"
                  (only used when dataset_source="nova_websocket")
        json_filename: Filename for JSON dataset (used when dataset_source="json")

    Raises:
        ValueError: If an invalid dataset_source is provided
        FileNotFoundError: If config files are not found
    """

    # Initialize Langfuse connection from AWS Secrets
    print("Initializing Langfuse connection from AWS Secrets...")
    langfuse = init_langfuse_from_secrets()

    # Load experiment configuration
    print(f"Loading experiment config from: {experiment_config_path}")
    with open(experiment_config_path, "r") as f:
        experiment_config = yaml.safe_load(f)

    # Load evaluation templates
    print(f"Loading evaluation templates from: {eval_templates_config_path}")
    eval_templates = EvalTemplates(config_path=eval_templates_config_path)

    # Run experiment based on dataset source
    if dataset_source == "nova_websocket":
        print(f"Running Nova WebSocket experiment (mode: {nova_mode})...")
        run_nova_evaluation_experiment(
            langfuse=langfuse,
            experiment_config=experiment_config,
            eval_templates=eval_templates,
            dataset_source="nova_websocket",
        )
    elif dataset_source == "json":
        if not json_filename:
            raise ValueError("json_filename must be provided when dataset_source='json'")
        print(f"Running JSON dataset experiment from: {json_filename}")
        run_nova_evaluation_experiment(
            langfuse=langfuse,
            experiment_config=experiment_config,
            eval_templates=eval_templates,
            dataset_source="json",
            json_filename=json_filename,
        )
    else:
        raise ValueError(f"Invalid dataset_source: {dataset_source}. Valid options are: 'nova_websocket', 'json'")

    print("Success: Complete Nova evaluation experiment finished successfully!")


def _run_json_experiment_sync(
    items: List[Dict[str, Any]],
    run_name: str,
    eval_templates: EvalTemplates,
    experiment_config: Dict[str, Any],
    langfuse: Langfuse,
) -> List[Dict[str, Any]]:
    """Synchronous wrapper for running experiments on JSON dataset items."""
    results = []

    for i, item in enumerate(items):
        print(f"\\n{'=' * 60}")
        print(f"Question {i + 1}/{len(items)}: {item['input'][:100]}...")

        # Create a simple trace (not linked to Langfuse dataset item)
        trace = langfuse.trace(
            name=f"{run_name}_item_{i + 1}",
            input=item["input"],
            metadata={
                "run_name": run_name,
                "item_index": i + 1,
                "expected_intent": item.get("metadata", {}).get("intent", ""),
                "expected_sources": item.get("metadata", {}).get("expected_sources", []),
            },
        )

        try:
            # Ask Nova (now synchronous)
            start_time = time.time()
            result = ask_nova_question(item["input"])
            duration = time.time() - start_time

            # Evaluate if we have an answer
            evaluations: Dict[str, Any] = {}
            if result["answer"] and not result["error"]:
                evaluations = evaluate_answer(
                    item["input"], result["answer"], result["sources"], eval_templates, experiment_config
                )

                # Score the trace
                for metric, eval_data in evaluations.items():
                    trace.score(name=metric, value=eval_data["score"], comment=eval_data["comment"])

            # Update trace with results
            trace.update(
                input=item["input"],  # Set the question as input
                output=result["answer"]
                if result["answer"]
                else f"Error: {result['error']}",  # Set answer or error as output
                metadata={
                    "intent": result["intent"],
                    "sources": result["sources"],
                    "error": result["error"],
                    "duration": duration,
                    "conversation_id": result["conversation_id"],
                    "expected_intent": item.get("metadata", {}).get("intent", ""),
                    "expected_answer": item.get("expected_output", ""),
                },
            )

            # Print summary
            if result["error"]:
                print(f"Error: Error: {result['error']}")
            else:
                print(
                    f"Success: Intent: {result['intent']} (expected: {item.get('metadata', {}).get('intent', 'N/A')})"
                )
                print(f"Sources: {len(result['sources'])}")
                print(f"Answer length: {len(result['answer'])} chars")
                if evaluations:
                    scores_str = ", ".join([f"{k}: {v['score']:.2f}" for k, v in evaluations.items()])
                    print(f"Scores: {scores_str}")

            # Flatten evaluation scores for results
            eval_scores = {k: v["score"] for k, v in evaluations.items()}

            results.append(
                {
                    "question": item["input"],
                    "answer": result["answer"],
                    "intent": result["intent"],
                    "expected_intent": item.get("metadata", {}).get("intent", ""),
                    "sources": "|".join(result["sources"]),
                    "error": result["error"],
                    "duration": duration,
                    **eval_scores,
                }
            )

        except Exception as e:
            print(f"Error: Exception processing item {i + 1}: {str(e)}")
            results.append(
                {
                    "question": item["input"],
                    "answer": "",
                    "intent": "",
                    "expected_intent": item.get("metadata", {}).get("intent", ""),
                    "sources": "",
                    "error": str(e),
                    "duration": 0,
                }
            )

    # Summary and save results
    print(f"\\n{'=' * 60}")
    print(f"JSON Experiment completed: {run_name}")
    print(f"Success: Successful: {sum(1 for r in results if not r.get('error'))}/{len(results)}")

    # Calculate average scores
    for eval_name in experiment_config["experiment"]["evaluations"]:
        scores = [r.get(eval_name, 0) for r in results if r.get(eval_name) is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"Average {eval_name}: {avg_score:.2f}")

    # Results available in Langfuse - no local CSV needed
    print("Results saved to Langfuse dataset runs")

    return results
