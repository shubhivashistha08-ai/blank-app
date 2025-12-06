# agent.py
import os
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

DATA_PATH = "data/fmcg_sample.csv"

# ---------- Data helpers ----------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["sales_proxy"] = 1_000_000 / df["sales_rank"].clip(lower=1)
    return df

# ---------- Tools ----------
@tool("get_top_products", return_direct=False)
def get_top_products(n: int = 5) -> str:
    """
    Return the top N products by sales performance proxy.
    Input: integer n (default 5).
    Output: a short text summary of top products with key metrics.
    """
    df = load_data().sort_values("sales_proxy", ascending=False).head(int(n))
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"ASIN {r['asin']} | {r['title']} | brand={r['brand']} | "
            f"price={r['price']} | discount%={r['discount_pct']:.1f} | "
            f"sponsored={bool(r['is_sponsored'])} | rating={r['star_rating']} | "
            f"reviews={r['review_count']} | sales_proxy={r['sales_proxy']:.0f}"
        )
    return "\n".join(rows)


@tool("simulate_promo", return_direct=False)
def simulate_promo(asin: str, discount_pct: float, is_sponsored: bool) -> str:
    """
    Simulate a simple promotion scenario for a given ASIN.
    Uses a heuristic: more discount and sponsoring improve sales_proxy.
    Input: asin (string), discount_pct (float), is_sponsored (bool).
    Output: explanation and estimated relative change in sales_proxy.
    """
    df = load_data()
    row = df[df["asin"] == asin].head(1)
    if row.empty:
        return f"No product found for ASIN {asin}."

    base = row.iloc[0]
    base_proxy = base["sales_proxy"]

    # Simple heuristic (you can replace with ML model later):
    # base elasticity parameters
    discount_factor = 1.0 + (discount_pct - base["discount_pct"]) * 0.02
    sponsor_factor = 1.2 if is_sponsored and not base["is_sponsored"] else 1.0
    sponsor_factor = 0.9 if (not is_sponsored and base["is_sponsored"]) else sponsor_factor

    new_proxy = base_proxy * discount_factor * sponsor_factor
    change_pct = 100 * (new_proxy - base_proxy) / base_proxy

    explanation = (
        f"Base product: {base['title']} (ASIN {asin})\n"
        f"Current: discount={base['discount_pct']:.1f}%, sponsored={bool(base['is_sponsored'])}, "
        f"sales_proxy≈{base_proxy:.0f}.\n"
        f"Scenario: discount={discount_pct:.1f}%, sponsored={bool(is_sponsored)}.\n"
        f"Estimated new sales_proxy≈{new_proxy:.0f} "
        f"({change_pct:+.1f}% vs current).\n"
        f"This is a rough heuristic, not an exact forecast."
    )
    return explanation


@tool("compare_promo_strategies", return_direct=False)
def compare_promo_strategies(asin: str) -> str:
    """
    Compare a few predefined promotion strategies for a product and pick the best.
    Strategies: keep as-is, +10% discount, +20% discount, sponsor on/off.
    Returns a ranked list of scenarios.
    """
    df = load_data()
    row = df[df["asin"] == asin].head(1)
    if row.empty:
        return f"No product found for ASIN {asin}."

    base = row.iloc[0]
    base_proxy = base["sales_proxy"]

    scenarios = []
    for disc in [base["discount_pct"], base["discount_pct"] + 10, base["discount_pct"] + 20]:
        for spons in [0, 1]:
            discount_factor = 1.0 + (disc - base["discount_pct"]) * 0.02
            sponsor_factor = 1.2 if spons and not base["is_sponsored"] else 1.0
            sponsor_factor = 0.9 if (not spons and base["is_sponsored"]) else sponsor_factor
            new_proxy = base_proxy * discount_factor * sponsor_factor
            scenarios.append({
                "discount": disc,
                "sponsored": bool(spons),
                "sales_proxy": new_proxy,
            })

    scenarios = sorted(scenarios, key=lambda x: x["sales_proxy"], reverse=True)
    lines = [f"Promotion strategy ranking for {base['title']} (ASIN {asin}):"]
    for i, s in enumerate(scenarios, start=1):
        change_pct = 100 * (s["sales_proxy"] - base_proxy) / base_proxy
        lines.append(
            f"{i}. discount={s['discount']:.1f}%, sponsored={s['sponsored']}, "
            f"sales_proxy≈{s['sales_proxy']:.0f} ({change_pct:+.1f}%)."
        )
    return "\n".join(lines)


# ---------- Build LangChain agent ----------
def build_agent() -> AgentExecutor:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY env var for the LLM.")

    llm = ChatOpenAI(
        model="gpt-4o-mini",  # or 'gpt-4o', adjust as needed
        temperature=0.1,
    )

    tools = [get_top_products, simulate_promo, compare_promo_strategies]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a retail and marketing analytics assistant focused on FMCG products. "
                "Use the provided tools to reason about promotions and sales. "
                "Explain your answers clearly and concisely.",
            ),
            MessagesPlaceholder("agent_scratchpad"),
            ("human", "{input}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    return executor
