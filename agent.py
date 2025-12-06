import requests
import pandas as pd
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -------------------------------
# Load API Keys from Streamlit Secrets
# -------------------------------
SERPAPI_API_KEY = st.secrets.get("SERPAPI_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

if not SERPAPI_API_KEY:
    raise RuntimeError("❌ SERPAPI_API_KEY missing in Streamlit secrets.")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY missing in Streamlit secrets.")

# -------------------------------
# Google Shopping Fetch Function (Amazon Proxy Data)
# -------------------------------
def fetch_google_shopping_peanut_data(
    keyword: str = "high protein peanut butter",
    max_items: int = 20
) -> pd.DataFrame:

    params = {
        "engine": "google_shopping",   # ✅ FREE PLAN SAFE
        "q": keyword,
        "hl": "en",
        "gl": "us",
        "api_key": SERPAPI_API_KEY,
        "num": max_items,
    }

    try:
        resp = requests.get(
            "https://serpapi.com/search.json",
            params=params,
            timeout=30
        )

        if resp.status_code != 200:
            st.error(f"❌ SerpAPI Error {resp.status_code}")
            return pd.DataFrame()

        data = resp.json()

    except Exception as e:
        st.error(f"❌ SerpAPI Request Failed: {e}")
        return pd.DataFrame()

    results = data.get("shopping_results", [])

    if not results:
        st.warning("⚠️ No Google Shopping results returned.")
        return pd.DataFrame()

    products = []

    for p in results:
        product_id = p.get("product_id")
        title = p.get("title")

        if not product_id or not title:
            continue

        price = None
        if isinstance(p.get("price"), dict):
            price = p["price"].get("value")

        rating = p.get("rating")
        reviews = p.get("reviews")
        position = p.get("position", 1000)

        products.append(
            {
                "product_id": product_id,
                "title": title,
                "brand": p.get("source"),
                "category": "Peanut / Nut Butter",
                "price": price,
                "list_price": None,
                "rating": rating,
                "review_count": reviews,
                "search_position": position,
                "is_sponsored": int(p.get("sponsored", False)),
            }
        )

    df = pd.DataFrame(products)

    if df.empty:
        return df

    # -------------------------------
    # Derived Metrics
    # -------------------------------
    df["discount_pct"] = 0.0
    df["search_position"] = df["search_position"].fillna(1000)
    df["sales_proxy"] = 1_000.0 / df["search_position"].clip(lower=1)

    return df

# -------------------------------
# LangChain Tools
# -------------------------------
@tool
def get_top_products(n: int = 5) -> str:
    """Return the top N products by sales proxy from Google Shopping."""
    df = fetch_google_shopping_peanut_data()

    if df.empty:
        return "No products found from Google Shopping."

    df = df.sort_values("sales_proxy", ascending=False).head(int(n))

    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"ID {r['product_id']} | {r['title']} | brand={r['brand']} | "
            f"price={r['price']} | sponsored={bool(r['is_sponsored'])} | "
            f"rating={r['rating']} | reviews={r['review_count']} | "
            f"search_pos={r['search_position']} | sales_proxy={r['sales_proxy']:.1f}"
        )

    return "\n".join(rows)

@tool
def simulate_promo(product_id: str, discount_pct: float, is_sponsored: bool) -> str:
    """Simulate a promotion scenario for a product using search visibility heuristics."""
    df = fetch_google_shopping_peanut_data()

    if df.empty:
        return "No products available to simulate."

    row = df[df["product_id"] == product_id].head(1)
    if row.empty:
        return f"No live product found for ID {product_id}."

    base = row.iloc[0]
    base_proxy = base["sales_proxy"]
    base_discount = base["discount_pct"]
    base_sponsored = bool(base["is_sponsored"])

    discount_factor = 1.0 + (discount_pct - base_discount) * 0.02
    sponsor_factor = 1.2 if is_sponsored and not base_sponsored else 1.0
    sponsor_factor = 0.9 if (not is_sponsored and base_sponsored) else sponsor_factor

    new_proxy = base_proxy * discount_factor * sponsor_factor
    change_pct = 100 * (new_proxy - base_proxy) / base_proxy

    return (
        f"Base product: {base['title']} (ID {product_id})\n"
        f"Current: discount={base_discount:.1f}%, sponsored={base_sponsored}, "
        f"sales_proxy≈{base_proxy:.1f}.\n"
        f"Scenario: discount={discount_pct:.1f}%, sponsored={bool(is_sponsored)}.\n"
        f"Estimated new sales_proxy≈{new_proxy:.1f} ({change_pct:+.1f}% vs current).\n"
        f"This is a directional estimate based on Google Shopping visibility."
    )

@tool
def compare_promo_strategies(product_id: str) -> str:
    """Compare multiple promotion strategies and rank them."""
    df = fetch_google_shopping_peanut_data()

    if df.empty:
        return "No products available to compare."

    row = df[df["product_id"] == product_id].head(1)
    if row.empty:
        return f"No live product found for ID {product_id}."

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
                {
                    "discount": disc,
                    "sponsored": bool(spons),
                    "sales_proxy": new_proxy
                }
            )

    scenarios = sorted(scenarios, key=lambda x: x["sales_proxy"], reverse=True)

    lines = [f"Promotion strategy ranking for {base['title']} (ID {product_id}):"]

    for i, s in enumerate(scenarios, start=1):
        change_pct = 100 * (s["sales_proxy"] - base_proxy) / base_proxy
        lines.append(
            f"{i}. discount={s['discount']:.1f}%, sponsored={s['sponsored']}, "
            f"sales_proxy≈{s['sales_proxy']:.1f} ({change_pct:+.1f}%)."
        )

    return "\n".join(lines)

# -------------------------------
# Build the Agent
# -------------------------------
def build_agent():
    """Return Google Shopping FMCG Promotion Agent."""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=OPENAI_API_KEY
    )

    tools = [
        get_top_products,
        simulate_promo,
        compare_promo_strategies
    ]

    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a retail and marketing analytics assistant focused on FMCG food products. "
                "You use Google Shopping market data as a proxy for Amazon demand signals, "
                "analyze promotions, and recommend effective campaigns."
            ),
            MessagesPlaceholder("agent_scratchpad"),
            ("human", "{input}"),
        ]
    )

    def chain(inputs: dict):
        messages = prompt.format_messages(
            input=inputs["input"],
            agent_scratchpad=[]
        )
        return llm_with_tools.invoke(messages)

    return chain
