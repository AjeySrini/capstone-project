

import os
import requests
import json

# IMPORTANT: Store your API key in Colab Secrets named 'LLM_API_KEY'
# If running locally, you might use python-dotenv to load from a .env file:
# from dotenv import load_dotenv
# load_dotenv()
# LLM_API_KEY = os.getenv('LLM_API_KEY')

try:
    from google.colab import userdata
    LLM_API_KEY = userdata.get('LLM_API_KEY')
except Exception:
    print("Colab secrets not available, trying environment variable. Please ensure 'LLM_API_KEY' is set.")
    LLM_API_KEY = os.environ.get('LLM_API_KEY')

if not LLM_API_KEY:
    raise ValueError("LLM_API_KEY not found. Please set it in Colab Secrets or as an environment variable.")

LLM_API_URL = "https://openrouter.ai/api/v1/chat/completions" 
LLM_MODEL_NAME = "openai/gpt-4o" 

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0, max_tokens: int = 512, model_name: str = LLM_MODEL_NAME):
    """
    Calls the LLM API with the given prompts and parameters.

    Args:
        system_prompt (str): The system prompt for the LLM.
        user_prompt (str): The user prompt for the LLM.
        temperature (float): The sampling temperature (0.0 for deterministic output).
        max_tokens (int): The maximum number of tokens to generate.
        model_name (str): The name of the LLM model to use.

    Returns:
        str: The content of the LLM's response, or None if an error occurs.
    """
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)

        response_json = response.json()
        if 'choices' in response_json and len(response_json['choices']) > 0:
            return response_json['choices'][0]['message']['content']
        else:
            print("Error: 'choices' key not found or empty in LLM response.")
            print(f"Full response: {response_json}")
            return None

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text}")
        return None
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
        return None
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"An unexpected request error occurred: {req_err}")
        return None
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")
        print(f"Raw response text: {response.text}")
        return None

# Demonstrate calling this function with a simple test prompt
system_test_prompt = "You are a helpful assistant."
user_test_prompt = "Reply with only the word: hello"

print(f"Testing call_llm with system_prompt: '{system_test_prompt}' and user_prompt: '{user_test_prompt}'")
output = call_llm(system_test_prompt, user_test_prompt, temperature=0)

if output:
    print(f"LLM Output: '{output.strip()}'")
    assert output.strip().lower() == "hello", "Test failed: LLM did not reply with 'hello'"
    print("Test successful!")
else:
    print("Test failed: No output from LLM.")

"""### 2. Prompt Design (Track B)

For Track B, we need a few-shot system prompt that defines the scoring rubric and includes one worked input-output example. The `temperature` will be set to `0` for consistent batch scoring.
"""

import jsonschema

# Define the JSON schema for the assessment output
assessment_schema = {
    "type": "object",
    "properties": {
        "risk_tier": {"type": "string", "enum": ["low", "medium", "high"]},
        "flag_for_review": {"type": "boolean"},
        "primary_signal": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommended_action": {"type": "string"}
    },
    "required": [
        "risk_tier",
        "flag_for_review",
        "primary_signal",
        "confidence",
        "recommended_action"
    ]
}

system_prompt_batch_scoring = """
You are an expert financial risk assessor. Your task is to evaluate customer records and provide a structured JSON assessment based on the following rubric:

**Scoring Rubric:**
-   **risk_tier**: Assign 'low', 'medium', or 'high' based on the customer's financial stability, transaction history, and credit score.
    -   `low`: Excellent credit, stable income, consistent positive transaction history.
    -   `medium`: Average credit, moderate income stability, some inconsistent transaction patterns.
    -   `high`: Poor credit, unstable income, frequent negative transaction patterns or high debt-to-income ratio.
-   **flag_for_review**: Set to `true` if any aspect of the record suggests potential fraud, high debt, or unusual activity that requires human review. Otherwise, `false`.
-   **primary_signal**: Identify the most significant factor contributing to the assigned risk tier (e.g., 'high credit score', 'unstable income', 'large cash withdrawals').
-   **confidence**: Your confidence in the assessment, 'low', 'medium', or 'high'. Be 'high' if all data points clearly align, 'low' if data is ambiguous or incomplete.
-   **recommended_action**: Suggest a concrete next step for the customer's profile (e.g., 'approve loan', 'request more documents', 'decline service', 'monitor closely').

**Output Format:**
Provide your assessment strictly as a JSON object matching the following schema:
```json
{
    "risk_tier": "string",
    "flag_for_review": "boolean",
    "primary_signal": "string",
    "confidence": "string",
    "recommended_action": "string"
}
```

**Worked Example:**

User Input Record:
```json
{
    "customer_id": "C101",
    "age": 35,
    "income": 75000,
    "credit_score": 780,
    "debt_to_income_ratio": 0.2,
    "transaction_history_3m": {"deposits": 50000, "withdrawals": 20000, "large_cash_withdrawals": 0}
}
```

LLM Assessment:
```json
{
    "risk_tier": "low",
    "flag_for_review": false,
    "primary_signal": "excellent credit score and stable income",
    "confidence": "high",
    "recommended_action": "approve loan"
}
```

Now, provide an assessment for the following customer record:
"""

# User prompt template for batch scoring
# This will be dynamically filled with the actual record JSON.
user_prompt_template_batch_scoring = """
```json
{record_json}
```
"""

print("System Prompt (excerpt):\n---\n" + system_prompt_batch_scoring[:500] + "...\n---")
print("User Prompt Template (excerpt):\n---\n" + user_prompt_template_batch_scoring.format(record_json="...") + "\n---")
print("Assessment Schema:\n" + json.dumps(assessment_schema, indent=2))

"""### 3. Structured Output Handling and Validation

We will implement a function to call the LLM, parse its JSON response, and validate it against the `assessment_schema`. If validation fails, a fallback dict will be returned.
"""

from jsonschema import validate, ValidationError, SchemaError

def get_validated_llm_assessment(
    system_prompt: str,
    record_json_str: str,
    schema: dict,
    temperature: float = 0.0
) -> dict:
    """
    Calls the LLM with a record, parses the JSON response, and validates it against a schema.
    Returns a validated dict or a fallback dict on failure.
    """
    user_prompt = user_prompt_template_batch_scoring.format(record_json=record_json_str)

    print(f"\nCalling LLM for record: {record_json_str[:100]}...")
    raw_llm_response = call_llm(system_prompt, user_prompt, temperature=temperature)

    fallback_assessment = {
        "risk_tier": None,
        "flag_for_review": None,
        "primary_signal": None,
        "confidence": None,
        "recommended_action": None
    }

    if not raw_llm_response:
        print("LLM call returned no response. Returning fallback.")
        return fallback_assessment

    # Strip whitespace and parse JSON
    try:
        llm_output_stripped = raw_llm_response.strip()
        # Handle cases where LLM might wrap JSON in markdown code block
        if llm_output_stripped.startswith('```json') and llm_output_stripped.endswith('```'):
            llm_output_stripped = llm_output_stripped[len('```json'):-len('```')].strip()

        parsed_assessment = json.loads(llm_output_stripped)
        print(f"Raw LLM Response: {llm_output_stripped[:200]}...")
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: Could not parse LLM response as JSON: {e}")
        print(f"Problematic response: {llm_output_stripped[:200]}...")
        return fallback_assessment

    # Validate against schema
    try:
        validate(instance=parsed_assessment, schema=schema)
        print("Validation Status: Pass")
        return parsed_assessment
    except ValidationError as e:
        print(f"Validation Status: Fail - {e.message}")
        print(f"Problematic assessment: {json.dumps(parsed_assessment, indent=2)}")
        return fallback_assessment
    except SchemaError as e:
        print(f"SchemaError: Invalid schema definition: {e}")
        return fallback_assessment

# Test with a dummy record and the function
dummy_record_json = json.dumps({
    "customer_id": "C102",
    "age": 40,
    "income": 60000,
    "credit_score": 650,
    "debt_to_income_ratio": 0.4,
    "transaction_history_3m": {"deposits": 30000, "withdrawals": 25000, "large_cash_withdrawals": 2}
})

print("\n--- Testing get_validated_llm_assessment function ---")
test_assessment = get_validated_llm_assessment(
    system_prompt_batch_scoring,
    dummy_record_json,
    assessment_schema,
    temperature=0
)
print("Test Assessment Result:")
print(json.dumps(test_assessment, indent=2))

"""### 4. Guardrails

Before every call to `call_llm`, we need to run a regex check on the user input to block personally identifiable information (PII).
"""

import re

def has_pii(text: str) -> bool:
    """
    Checks if the given text contains common patterns of Personally Identifiable Information (PII).

    Args:
        text (str): The input text to check.

    Returns:
        bool: True if PII is detected, False otherwise.
    """
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))

# --- Demonstrate the PII guardrail ---
print("\n--- Demonstrating PII Guardrail ---")

# Test 1: Input with email (should be blocked)
pii_input_email = json.dumps({
    "customer_id": "C103",
    "age": 28,
    "income": 45000,
    "email": "john.doe@example.com",
    "credit_score": 600,
    "debt_to_income_ratio": 0.5,
    "transaction_history_3m": {"deposits": 10000, "withdrawals": 9000, "large_cash_withdrawals": 1}
})

print(f"\nTesting input with PII (email):\n{pii_input_email}")
if has_pii(pii_input_email):
    print("Input blocked: PII (email) detected.")
else:
    print("Input allowed: No PII (email) detected. (This should not happen for this input)")

# Test 2: Input with phone number (should be blocked)
pii_input_phone = json.dumps({
    "customer_id": "C104",
    "age": 50,
    "income": 90000,
    "phone": "555-123-4567",
    "credit_score": 720,
    "debt_to_income_ratio": 0.3,
    "transaction_history_3m": {"deposits": 60000, "withdrawals": 30000, "large_cash_withdrawals": 0}
})

print(f"\nTesting input with PII (phone number):\n{pii_input_phone}")
if has_pii(pii_input_phone):
    print("Input blocked: PII (phone number) detected.")
else:
    print("Input allowed: No PII (phone number) detected. (This should not happen for this input)")

# Test 3: Clean input (should proceed)
clean_input = json.dumps({
    "customer_id": "C105",
    "age": 32,
    "income": 55000,
    "credit_score": 680,
    "debt_to_income_ratio": 0.45,
    "transaction_history_3m": {"deposits": 20000, "withdrawals": 15000, "large_cash_withdrawals": 0}
})

print(f"\nTesting clean input:\n{clean_input}")
if has_pii(clean_input):
    print("Input blocked: PII detected. (This should not happen for this input)")
else:
    print("Input allowed: No PII detected.")

"""### 5. Demonstrate the Feature End-to-End & Temperature A/B Comparison

Now, we'll run the complete pipeline on three distinct input records, applying the PII guardrail, and performing the LLM assessment with both `temperature=0` and `temperature=0.7`.
"""

from typing import Dict, Any, List

def process_record_with_llm(
    record: Dict[str, Any],
    system_prompt: str,
    schema: dict,
    temperatures: List[float] = [0.0, 0.7]
) -> Dict[str, Any]:
    """
    Processes a single record through the LLM pipeline, including PII check, LLM call,
    and schema validation, for multiple temperatures.
    """
    record_json_str = json.dumps(record)
    results = {
        "input_record": record,
        "guardrail_status": "Passed",
        "assessments": {}
    }

    print(f"\n--- Processing Record: {record.get('customer_id', 'N/A')} ---")
    print(f"Input: {record_json_str}")

    if has_pii(record_json_str):
        print("Input blocked: PII detected.")
        results["guardrail_status"] = "Blocked (PII detected)"
        for temp in temperatures:
            results["assessments"][f"temp_{temp}"] = {
                "raw_llm_output": None,
                "validated_assessment": {
                    "risk_tier": "BLOCKED",
                    "flag_for_review": True,
                    "primary_signal": "PII detected",
                    "confidence": "high",
                    "recommended_action": "Do not process"
                },
                "validation_status": "N/A (blocked)"
            }
        return results

    for temp in temperatures:
        print(f"\nCalling LLM with temperature={temp}")
        user_prompt = user_prompt_template_batch_scoring.format(record_json=record_json_str)

        raw_llm_response = call_llm(system_prompt, user_prompt, temperature=temp)

        assessment_entry = {
            "raw_llm_output": raw_llm_response,
            "validated_assessment": {},
            "validation_status": "Fail"
        }

        if raw_llm_response:
            llm_output_stripped = raw_llm_response.strip()
            if llm_output_stripped.startswith('```json') and llm_output_stripped.endswith('```'):
                llm_output_stripped = llm_output_stripped[len('```json'):-len('```')].strip()

            try:
                parsed_assessment = json.loads(llm_output_stripped)
                try:
                    validate(instance=parsed_assessment, schema=schema)
                    assessment_entry["validated_assessment"] = parsed_assessment
                    assessment_entry["validation_status"] = "Pass"
                except ValidationError as e:
                    print(f"Validation failed for temp={temp}: {e.message}")
                    assessment_entry["validated_assessment"] = {
                        "risk_tier": None, "flag_for_review": None, "primary_signal": None,
                        "confidence": None, "recommended_action": None
                    }
                    assessment_entry["validation_status"] = f"Fail: {e.message}"
            except json.JSONDecodeError as e:
                print(f"JSON decoding failed for temp={temp}: {e}")
                assessment_entry["validated_assessment"] = {
                    "risk_tier": None, "flag_for_review": None, "primary_signal": None,
                    "confidence": None, "recommended_action": None
                }
                assessment_entry["validation_status"] = f"Fail: {e}"
        else:
            print(f"LLM returned no response for temperature={temp}")
            assessment_entry["validated_assessment"] = {
                "risk_tier": None, "flag_for_review": None, "primary_signal": None,
                "confidence": None, "recommended_action": None
            }
            assessment_entry["validation_status"] = "Fail: No LLM response"

        results["assessments"][f"temp_{temp}"] = assessment_entry
    return results

# Define three distinct input records
records_to_process = [
    # Record 1: Low risk
    {
        "customer_id": "C001",
        "age": 45,
        "income": 120000,
        "credit_score": 810,
        "debt_to_income_ratio": 0.15,
        "transaction_history_3m": {"deposits": 80000, "withdrawals": 10000, "large_cash_withdrawals": 0, "late_payments": 0}
    },
    # Record 2: Medium risk, some flags
    {
        "customer_id": "C002",
        "age": 30,
        "income": 50000,
        "credit_score": 680,
        "debt_to_income_ratio": 0.4,
        "transaction_history_3m": {"deposits": 15000, "withdrawals": 12000, "large_cash_withdrawals": 2, "late_payments": 1}
    },
    # Record 3: High risk, potential PII (for guardrail test), and different risk factors
    {
        "customer_id": "C003",
        "age": 55,
        "income": 30000,
        "credit_score": 520,
        "debt_to_income_ratio": 0.65,
        "transaction_history_3m": {"deposits": 5000, "withdrawals": 6000, "large_cash_withdrawals": 5, "late_payments": 3},
        "contact_email": "bad.debt@example.org" # PII for guardrail test
    }
]

all_results = []
for record in records_to_process:
    all_results.append(process_record_with_llm(record, system_prompt_batch_scoring, assessment_schema))

print("\n--- All Processing Complete ---")