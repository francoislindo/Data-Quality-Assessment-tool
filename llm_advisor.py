import json
import openai
import os
from typing import Any
import pandas as pd
from openai import AuthenticationError, RateLimitError, OpenAIError

def suggest_fixes(report_path: str, openai_api_key: str = None, use_case: str = None) -> None:
    """
    Reads the JSON report, sends it to GPT-4o, and asks for a summary of the worst five issues and a pandas script to fix them.
    Also asks if the data quality is good enough for the provided use case, and if not, recommends how to address it.
    Prints the markdown summary and code snippet.
    Args:
        report_path (str): Path to the JSON report file.
        openai_api_key (str, optional): OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
        use_case (str, optional): The user's use case for the data.
    """
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass as argument.")
        return

    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    prompt = (
        "You are a data quality advisor. "
        "Given the following data quality report (as a JSON list of dicts), "
        "summarize the worst five issues and output a pandas script to fix them. "
        "The user wants to use this data for the following use case: '" + (use_case or "(no use case provided)") + "'. "
        "Analyze if the data quality is good enough for this use case. "
        "If it is not, make recommendations on how to address the issues for this use case. "
        "Output a markdown summary and a code snippet.\n\n"
        f"Report: {json.dumps(report)}"
    )

    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful data quality advisor."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.2,
    )

    content = response.choices[0].message.content
    print(content)

def get_openai_api_key(user_key=None):
    key = user_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY or pass as argument.")
    return key

def summarize_data_quality(profile_df, sample_rows, api_key=None, save_to_file=True):
    openai_api_key = get_openai_api_key(api_key)
    # Prepare the message for the LLM
    profile_summary = profile_df.to_markdown(index=False)
    sample_summary = pd.DataFrame(sample_rows).to_markdown(index=False) if isinstance(sample_rows, list) else pd.DataFrame(sample_rows).to_markdown(index=False)
    prompt = f"""
You are a senior data quality analyst reviewing the results of a data profiling assessment.

Please analyze the summary statistics and sample data provided.

Return a concise list of observations using clear, actionable bullet points.

• Focus on the most significant data quality issues, trends, and anomalies.
• Avoid technical jargon or unnecessary detail—assume the audience includes business stakeholders.
• Group observations by theme (e.g. completeness, consistency, duplication, formatting, etc).
• Highlight only meaningful risks or high-impact insights.

Use this structure:

Summary of Key Data Quality Findings:

[Finding 1 — e.g., “High null rate in ‘referrer_email’ (~42%) – may impact lead attribution.”]

[Finding 2 — e.g., “Phone number formatting is inconsistent; 30% do not match expected regex.”]

[Finding 3 — e.g., “Potential duplicate records identified: 12% repeated email addresses.”]

...

End with 1–2 short recommendations if applicable (e.g., “Consider applying regex validation and enforcing unique constraints on email.”)

Data Profile:
{profile_summary}

Sample Rows:
{sample_summary}
"""
    try:
        # DEBUG PRINTS
        print("[DEBUG] openai module:", openai)
        print("[DEBUG] openai module type:", type(openai))
        print("[DEBUG] openai version:", getattr(openai, '__version__', 'unknown'))
        print("[DEBUG] API key (masked):", openai_api_key[:4] + "..." + openai_api_key[-4:])
        print("[DEBUG] Prompt:\n", prompt[:500], "...\n[truncated]")
        client = openai.OpenAI(api_key=openai_api_key)
        print("[DEBUG] OpenAI client:", client)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful data quality analyst."},
                      {"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3
        )
        summary = response.choices[0].message.content.strip()
        if save_to_file:
            with open("ai_summary.md", "w", encoding="utf-8") as f:
                f.write(summary)
        return summary
    except AuthenticationError:
        return "Authentication error: Please check your OpenAI API key."
    except RateLimitError:
        return "Rate limit exceeded: Please try again later or check your OpenAI plan."
    except OpenAIError as e:
        return f"OpenAI error: {e}"
    except Exception as e:
        return f"An error occurred: {e}" 