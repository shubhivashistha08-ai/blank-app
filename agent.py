import os
import requests
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
import streamlit as st

SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

#SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("SERPAPI_API_KEY present?", bool(SERPAPI_API_KEY))
print("OPENAI_API_KEY present?", bool(OPENAI_API_KEY))

if not SERPAPI_API_KEY:
    # do NOT raise here any more, just log; otherwise the app dies before showing UI
    print("WARNING: My_APIKey (SerpAPI) is missing or empty.")

def fetch_amazon_peanut_data(keyword: str = "high protein peanut butter",
                             max_items: int = 20) -> pd.DataFrame:
    params = {
        "engine": "amazon_search",
        "amazon_domain": "amazon.com",
        "q": keyword,
        "api_key": SERPAPI_API_KEY,
        "num": max_items,
    }
    try:
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        if resp.status_code != 200:
            print("SerpAPI error status:", resp.status_code)
            print("SerpAPI error body:", resp.text[:400])
            return pd.DataFrame()
    except requests.RequestException as e:
        print("SerpAPI request failed:", e)
        return pd.DataFrame()

    data = resp.json()

    products = []
    for p in data.get("organic_results", []):
        asin = p.get("asin")
        title = p.get("title")
        price = None
        list_price = None
        rating = p.get("rating")
        reviews = p.get("ratings_total")
        position = p.get("position")

        if isinstance(p.get("price"), dict):
            price = p["price"].get("value")
        elif isinstance(p.get("price"), (int, float)):
            price = p["price"]

        if isinstance(p.get("list_price"), dict):
            list_price = p["list_price"].get("value")
        elif isinstance(p.get("list_price"), (int, float)):
            list_price = p["list_price"]

        if not asin or not title:
            continue

        products.append(
            {
                "asin": asin,
                "title": title,
                "brand": p.get("brand"),
                "category": "Peanut Butter",
                "price": price,
                "list_price": list_price,
                "rating": rating,
                "review_count": reviews,
                "search_position": position,
                "is_sponsored": 1 if position and position <= 3 else 0,
            }
        )

    df = pd.DataFrame(products)
    if df.empty:
        return df

    df["discount_pct"] = 0.0
    mask = df["price"].notna() & df["list_price"].notna() & (df["list_price"] > 0)
    df.loc[mask, "discount_pct"] = (
        100.0 * (df.loc[mask, "list_price"] - df.loc[mask, "price"]) / df.loc[mask, "list_price"]
    )

    df["search_position"] = df["search_position"].fillna(1000)
    df["sales_proxy"] = 1_000.0 / df["search_position"].clip(lower=1)
    return df

@tool
def get_top_products(n: int = 5) -> str:
    """Return the top N products by sales_proxy from the latest SerpAPI Amazon search."""
    df = fetch_amazon_peanut_data()
    if df.empty:
        return "No products found from SerpAPI."

    df = df.sort_values("sales_proxy", ascending=False).head(int(n))
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"ASIN {r['asin']} | {r['title']} | brand={r['brand']} | "
            f"price={r['price']} | discount%={r['discount_pct']:.1f} | "
            f"sponsored={bool(r['is_sponsored'])} | rating={r['rating']} | "
            f"reviews={r['review_count']} | search_pos={r['search_position']} | "
            f"sales_proxy={r['sales_proxy']:.1f}"
        )
    return "\n".join(rows)

@tool
def simulate_promo(asin: str, discount_pct: float, is_sponsored: bool) -> str:
    """Simulate a promotion scenario for a given ASIN using simple heuristics."""
    df = fetch_amazon_peanut_data()
    if df.empty:
        return "No products available to simulate."

    row = df[df["asin"] == asin].head(1)
    if row.empty:
        return f"No live product found for ASIN {asin}."

    base = row.iloc[0]
    base_proxy = base["sales_proxy"]
    base_discount = base["discount_pct"]
    base_sponsored = bool(base["is_sponsored"])

    discount_factor = 1.0 + (discount_pct - base_discount) * 0.02
    sponsor_factor = 1.2 if is_sponsored and not base_sponsored else 1.0
    sponsor_factor = 0.9 if (not is_sponsored and base_sponsored) else sponsor_factor

    new_proxy = base_proxy * discount_factor * sponsor_factor
    change_pct = 100 * (new_proxy - base_proxy) / base_proxy

    explanation = (
        f"Base product: {base['title']} (ASIN {asin})\n"
        f"Current: discount={base_discount:.1f}%, sponsored={base_sponsored}, "
        f"sales_proxy≈{base_proxy:.1f}.\n"
        f"Scenario: discount={discount_pct:.1f}%, sponsored={bool(is_sponsored)}.\n"
        f"Estimated new sales_proxy≈{new_proxy:.1f} ({change_pct:+.1f}% vs current).\n"
        f"This is a rough heuristic based on search position."
    )
    return explanation

@tool
def compare_promo_strategies(asin: str) -> str:
    """Compare several promotion strategies for a product and rank them."""
    df = fetch_amazon_peanut_data()
    if df.empty:
        return "No products available to compare."

    row = df[df["asin"] == asin].head(1)
    if row.empty:
        return f"No live product found for ASIN {asin}."

    base = row.iloc[0]
    base_proxy = base["sales_proxy"]
    base_discount = base["discount_pct"]
    base_sponsored = bool(base["is_sponsored"])

    scenarios = []
    for disc in [base_discount, base_discount + 10, base_discount + 20]:
        for spons in [0, 1]:
            discount_factor = 1.0 + (disc - base_discount) * 0.02
            sponsor_factor = 1.2 if spons and not base_sponsored else 1.0
            sponsor_factor = 0.9 if (not spons and base_sponsored) else sponsor_factor
            new_proxy = base_proxy * discount_factor * sponsor_factor
            scenarios.append(
                {"discount": disc, "sponsored": bool(spons), "sales_proxy": new_proxy}
            )

    scenarios = sorted(scenarios, key=lambda x: x["sales_proxy"], reverse=True)
    lines = [f"Promotion strategy ranking for {base['title']} (ASIN {asin}):"]
    for i, s in enumerate(scenarios, start=1):
        change_pct = 100 * (s["sales_proxy"] - base_proxy) / base_proxy
        lines.append(
            f"{i}. discount={s['discount']:.1f}%, sponsored={s['sponsored']}, "
            f"sales_proxy≈{s['sales_proxy']:.1f} ({change_pct:+.1f}%)."
        )
    return "\n".join(lines)

def build_agent():
    """Return a simple tool-calling chain (prompt + llm_with_tools)."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY env var for the LLM.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    tools = [get_top_products, simulate_promo, compare_promo_strategies]

    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a retail and marketing analytics assistant focused on FMCG food products. "
                "Use the tools to fetch live Amazon data via SerpAPI, analyze promotions, "
                "and recommend effective campaigns.",
            ),
            MessagesPlaceholder("agent_scratchpad"),
            ("human", "{input}"),
        ]
    )

    def chain(inputs: dict):
        messages = prompt.format_messages(
            input=inputs["input"],
            agent_scratchpad=[],
        )
        return llm_with_tools.invoke(messages)

    return chain
