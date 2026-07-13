LLM-Powered Feature: Tabular Record Batch Scoring (Track B)
This document describes the implementation of an LLM-powered feature for batch scoring tabular records, focusing on Track B: Tabular Record Batch Scoring.

Chosen Track: (B) Tabular Record Batch Scoring
Prompt Design
System Prompt (verbatim):

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
Worked Example:

User Input Record:

{
    "customer_id": "C101",
    "age": 35,
    "income": 75000,
    "credit_score": 780,
    "debt_to_income_ratio": 0.2,
    "transaction_history_3m": {"deposits": 50000, "withdrawals": 20000, "large_cash_withdrawals": 0}
}
LLM Assessment:

{
    "risk_tier": "low",
    "flag_for_review": false,
    "primary_signal": "excellent credit score and stable income",
    "confidence": "high",
    "recommended_action": "approve loan"
}
Now, provide an assessment for the following customer record:


### User Prompt Template (with placeholders):
{record_json}

### Temperature Choice
For this task, `temperature=0` is chosen for the primary assessments because it produces more deterministic and predictable outputs. In structured data tasks like batch scoring, consistency and reproducibility are paramount. A low temperature near 0 forces the model to always pick the highest-probability next token, leading to consistent JSON outputs that are easier to validate and integrate into automated workflows.

## Structured Output Handling and Validation

The LLM's output is expected to be a JSON object conforming to the following schema:

```json
{
  "type": "object",
  "properties": {
    "risk_tier": {
      "type": "string",
      "enum": [
        "low",
        "medium",
        "high"
      ]
    },
    "flag_for_review": {
      "type": "boolean"
    },
    "primary_signal": {
      "type": "string"
    },
    "confidence": {
      "type": "string",
      "enum": [
        "low",
        "medium",
        "high"
      ]
    },
    "recommended_action": {
      "type": "string"
    }
  },
  "required": [
    "risk_tier",
    "flag_for_review",
    "primary_signal",
    "confidence",
    "recommended_action"
  ]
}
After each LLM call, the raw response is:

Stripped of leading/trailing whitespace.
Checked for and stripped of markdown JSON code block wrappers (e.g., json...).
Parsed into a Python dictionary using json.loads() within a try-except json.JSONDecodeError block.
Validated against the assessment_schema using jsonschema.validate() within a try-except jsonschema.ValidationError block.
If parsing or validation fails, a fallback dictionary with all required fields set to None (or 'BLOCKED' for PII cases) is returned, and the error is logged.

PII Guardrail Test Results
Input Type	Input Excerpt	Guardrail Result
Contains Email	..."contact_email": "john.doe@example.com"...	Input blocked: PII (email) detected.
Contains Phone	..."phone": "555-123-4567"...	Input blocked: PII (phone number) detected.
Clean Input	..."customer_id": "C105"...	Input allowed: No PII detected.
Demonstration of Feature End-to-End
Below is a table summarizing the processing of three distinct input records, including guardrail results, and the final LLM assessment.

Input	LLM Output (temp=0)	Valid JSON (pass/fail)	Pass/Block (guardrail result)
{	```json	Pass	Passed
"customer_id": "C001",	{		
"age": 45,	"risk_tier": "low",		
"income": 120000,	"flag_for_review": false,		
"credit_score": 810,	"primary_signal": "excellent credit score and stable income",		
"debt_to_income_ratio": 0.15,	"confidence": "high",		
"transaction_history_3m": {	"recommended_action": "approve loan"		
"deposits": 80000,	}		
"withdrawals": 10000,	```		
"large_cash_withdrawals": 0,			
"late_payments": 0			
}			
}			
{	```json	Pass	Passed
"customer_id": "C002",	{		
"age": 30,	"risk_tier": "medium",		
"income": 50000,	"flag_for_review": true,		
"credit_score": 680,	"primary_signal": "high debt-to-income ratio and large cash withdrawals",		
"debt_to_income_ratio": 0.4,	"confidence": "medium",		
"transaction_history_3m": {	"recommended_action": "monitor closely"		
"deposits": 15000,	}		
"withdrawals": 12000,	```		
"large_cash_withdrawals": 2,			
"late_payments": 1			
}			
}			
{	BLOCKED due to PII	N/A	Blocked (PII detected)
"customer_id": "C003",			
"age": 55,			
"income": 30000,			
"credit_score": 520,			
"debt_to_income_ratio": 0.65,			
"transaction_history_3m": {			
"deposits": 5000,			
"withdrawals": 6000,			
"large_cash_withdrawals": 5,			
"late_payments": 3			
},			
"contact_email": "bad.debt@example.org"			
}			
Temperature A/B Comparison
This comparison demonstrates the effect of temperature on LLM output variability.

Input Record (customer_id)	Output at temp=0 (primary_signal)	Output at temp=0.7 (primary_signal)	Key difference (primary_signal)
Input Record (customer_id)	Output at temp=0 (primary_signal)	Output at temp=0.7 (primary_signal)	Key difference (primary_signal)
:-----------------------------	:-----------------------------------------------------	:---------------------------------------------	:-------------------------------------------------------------------------------------------------------------------------
C001	excellent credit score and stable income	excellent credit score and stable income	No significant difference in primary_signal or validation status
C002	high debt-to-income ratio and large cash withdrawals	average credit and moderate income stability	temp=0: 'high debt-to-income ratio and large cash withdrawals', temp=0.7: 'average credit and moderate income stability'
C003	BLOCKED	BLOCKED	PII detected, not processed.
Explanation of Temperature Differences:
Temperature=0: Produces more deterministic outputs because the model always selects the token with the highest probability. This leads to consistent and reproducible responses, which is ideal for structured tasks where exact formatting and content are crucial, such as financial assessments.

Temperature=0.7: Introduces variability by sampling from a broader distribution of possible next tokens. This means the model might choose tokens with slightly lower probabilities, leading to more creative, diverse, or slightly different phrasing in repeated calls with the same input. While useful for creative generation, it can result in less consistent structured output and potentially introduce validation failures.