"""
LLM-based report generation for model analysis.

Uses an LLM to provide high-level insights and recommendations
based on model performance metrics and feature importance.
"""

import logging
from typing import Dict, Any
from openai import OpenAI
import os
from dotenv import load_dotenv

from src import config

load_dotenv()

logger = logging.getLogger(__name__)


def generate_llm_report(metrics: Dict[str, Any], top_features: str) -> str:
    """
    Generate an LLM-based report analyzing model metrics.

    Args:
        metrics: Dictionary containing model performance metrics.
        top_features: String representation of top features.

    Returns:
        A string report with model analysis. Returns fallback messages if API is unavailable.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        logger.warning("OPENROUTER_API_KEY not found. Skipping LLM report generation.")
        return "LLM Report: API key not found. Please set OPENROUTER_API_KEY environment variable."

    try:
        logger.debug("Connecting to LLM API...")
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        prompt = f"""
You are a senior ML analyst.

Here are the model results:

Task type: {metrics["task_type"]}
MAE: {metrics["mae"]}
R2: {metrics["r2"]}

Top features:
{top_features}

Provide:
1. Explanation of metrics
2. Interpretation of model quality
3. Business insights
4. Risks
5. Recommendations
"""

        logger.debug("Sending request to LLM API...")
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.LLM_TEMPERATURE,
            timeout=config.LLM_TIMEOUT,
        )

        # Safely extract content, ensuring we always have a string
        content = response.choices[0].message.content

        if content is None:
            logger.warning("LLM returned empty response.")
            return "LLM returned empty response."

        logger.info("LLM report generated successfully.")
        return str(content)

    except Exception as e:
        logger.error(f"LLM Report generation failed: {str(e)}", exc_info=True)
        return f"LLM Report generation failed: {str(e)}"
