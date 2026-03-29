import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")

INSIGHT_PROMPT = PromptTemplate.from_template("""
You are a supply chain planning assistant for an automotive manufacturer in Germany.
Based on the following supply chain data, provide a concise planner briefing.

Demand Forecast Summary:
- Product: {product_name}
- Average daily demand (next 28 days): {avg_demand} units
- Total 28-day demand: {total_28d} units

Supply Risk Alerts:
{risk_alerts}

Inventory Status:
{inventory_status}

Write exactly 3 sentences for the supply chain planner:
1. Summarise the demand outlook and whether it requires urgent action.
2. Highlight the most critical supplier risk and its operational impact.
3. Give one specific and actionable recommendation the planner should take today.

Be specific, use the numbers provided, and focus on operational decisions.
""")


def format_risk_alerts(high_risk: list) -> str:
    if not high_risk:
        return "No high risk suppliers detected."
    lines = [
        f"  - {s['supplier_id']} ({s['supplier_name']}): risk score {s['risk_score']:.3f}"
        for s in high_risk[:3]
    ]
    return "\n".join(lines)


def build_chain():
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    return INSIGHT_PROMPT | llm | StrOutputParser()


def generate_insight(
    product_name:     str,
    avg_demand:       int,
    total_28d:        int,
    high_risk:        list,
    inventory_status: str
) -> str:
    chain = build_chain()
    return chain.invoke({
        "product_name":     product_name,
        "avg_demand":       avg_demand,
        "total_28d":        total_28d,
        "risk_alerts":      format_risk_alerts(high_risk),
        "inventory_status": inventory_status
    })


if __name__ == "__main__":
    from src.risk.scorer import get_high_risk_suppliers

    high_risk = get_high_risk_suppliers()

    insight = generate_insight(
        product_name     = "Brake Caliper Assembly (SKU-0001)",
        avg_demand       = 158,
        total_28d        = 4438,
        high_risk        = high_risk,
        inventory_status = "Current stock: 1200 units. Safety stock: 500 units. Reorder point: 800 units."
    )

    print("\nPlanner Insight:")
    print(insight)