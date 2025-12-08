import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

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

# ===============================
# ENHANCED: Sales & Campaign Data
# ===============================

def generate_synthetic_campaign_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance dataframe with:
    - Estimated monthly sales (based on search position, rating, reviews)
    - Social media reach (Twitter/YouTube mentions proxy)
    - Last 3 promotions with lift % and ROI
    - Competitor count and avg competitor price
    """
    
    # 1. SALES ESTIMATE (proxy: search position + rating + reviews)
    # Better position (lower number) + higher rating + more reviews = more sales
    df["sales_estimate"] = np.where(
        df["search_position"] > 0,
        (1000.0 / df["search_position"].clip(lower=1)) * 
        (1 + df["rating"].fillna(3.0) / 10) *
        (1 + df["review_count"].fillna(10) / 100),
        50
    )
    df["sales_estimate"] = df["sales_estimate"].fillna(100).astype(int)
    
    # 2. SOCIAL REACH SCORE (0-100)
    # Higher reviews + rating = more likely social mentions
    df["social_reach_score"] = np.clip(
        (df["review_count"].fillna(10) / df["review_count"].fillna(10).max() * 60) +
        (df["rating"].fillna(3.0) / 5.0 * 40),
        0, 100
    ).astype(int)
    
    # 3. PROMOTION HISTORY (simulated last 3 campaigns)
    promo_history = []
    for idx, row in df.iterrows():
        promos = []
        for i in range(3):
            days_ago = 30 + (i * 20)  # 30, 50, 70 days ago
            lift = np.random.randint(5, 35)  # 5-35% lift
            roi = np.random.uniform(1.2, 3.5)  # 1.2x to 3.5x ROI
            promo_type = np.random.choice(["BOGO", "Discount", "Bundle", "Influencer", "Sample"])
            
            promos.append({
                "type": promo_type,
                "lift_pct": lift,
                "roi": round(roi, 2),
                "days_ago": days_ago
            })
        promo_history.append(promos)
    
    df["promo_history"] = promo_history
    
    # 4. COMPETITOR COUNT & AVG COMPETITOR PRICE
    # Group by brand-ish, count how many products in similar price range
    df["competitor_count"] = df.groupby("category")["product_id"].transform("count") - 1
    df["competitor_count"] = df["competitor_count"].clip(lower=0)
    
    # Average price in same category
    avg_cat_price = df.groupby("category")["price"].transform("mean")
    df["avg_competitor_price"] = avg_cat_price.fillna(df["price"].mean())
    
    # 5. PRICE POSITIONING
    df["price_vs_market"] = np.where(
        (df["price"].notna()) & (df["avg_competitor_price"] > 0),
        ((df["price"] - df["avg_competitor_price"]) / df["avg_competitor_price"] * 100).round(2),
        0
    )
    
    # 6. CAMPAIGN LIFT OPPORTUNITY (based on current social reach vs competitors)
    df["campaign_potential"] = np.clip(
        (100 - df["social_reach_score"]) * 0.5 +  # Room to grow socially
        (df["review_count"].fillna(10) < 50).astype(int) * 20,  # Low reviews = high potential
        0, 100
    ).astype(int)
    
    return df


# -------------------------------
# Google Shopping Fetch Function
# -------------------------------
def fetch_google_shopping_peanut_data(
    keyword: str = "high protein peanut butter",
    max_items: int = 20
) -> pd.DataFrame:

    params = {
        "engine": "google_shopping",
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

    # Add base metrics
    df["discount_pct"] = 0.0
    df["search_position"] = df["search_position"].fillna(1000)
    df["sales_proxy"] = 1_000.0 / df["search_position"].clip(lower=1)

    # ✅ ADD ENHANCED DATA
    df = generate_synthetic_campaign_data(df)

    return df


# ===============================
# LangChain Tools (Enhanced)
# ===============================

@tool
def get_top_products(n: int = 5) -> str:
    """Return the top N products by sales estimate from Google Shopping."""
    df = fetch_google_shopping_peanut_data()

    if df.empty:
        return "No products found from Google Shopping."

    df = df.sort_values("sales_estimate", ascending=False).head(int(n))

    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"ID {r['product_id']} | {r['title']} | "
            f"brand={r['brand']} | price={r['price']} | "
            f"rating={r['rating']} | reviews={r['review_count']} | "
            f"est_sales={r['sales_estimate']} units/month | "
            f"social_reach={r['social_reach_score']}/100 | "
            f"promo_potential={r['campaign_potential']}%"
        )

    return "\n".join(rows)


@tool
def get_top_campaigns(n: int = 5) -> str:
    """Return top products by campaign potential and social reach."""
    df = fetch_google_shopping_peanut_data()

    if df.empty:
        return "No products found."

    df = df.sort_values("campaign_potential", ascending=False).head(int(n))

    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r['title']} (ID {r['product_id']}) | "
            f"Campaign Potential: {r['campaign_potential']}% | "
            f"Social Reach: {r['social_reach_score']}/100 | "
            f"Current Est Sales: {r['sales_estimate']} units/month"
        )

    return "\n".join(rows)


@tool
def analyze_product_promo(product_id: str) -> str:
    """Detailed promotion analysis for a product including past campaigns and recommendations."""
    df = fetch_google_shopping_peanut_data()

    if df.empty:
        return "No products available."

    row = df[df["product_id"] == product_id].head(1)
    if row.empty:
        return f"No live product found for ID {product_id}."

    r = row.iloc[0]

    # Past campaigns
    promo_history = r["promo_history"]
    past_promos = "\n".join([
        f"  • {p['type']}: {p['lift_pct']}% lift, {p['roi']}x ROI ({p['days_ago']} days ago)"
        for p in promo_history
    ])

    # Price positioning
    price_position = "Premium" if r["price_vs_market"] > 10 else "Competitive" if r["price_vs_market"] > -10 else "Budget"

    # Recommendation
    if r["campaign_potential"] > 70:
        recommendation = "🚀 HIGH POTENTIAL: Strong growth opportunity through influencer or social campaigns"
    elif r["campaign_potential"] > 40:
        recommendation = "📈 MODERATE POTENTIAL: Bundle or BOGO could boost sales"
    else:
        recommendation = "✅ STABLE: Product performing well, maintain current strategy"

    return (
        f"**{r['title']}** (ID {r['product_id']})\n"
        f"Brand: {r['brand']} | Rating: ⭐{r['rating']} | Reviews: {r['review_count']}\n\n"
        f"**Current Performance:**\n"
        f"  • Est. Monthly Sales: {r['sales_estimate']} units\n"
        f"  • Social Reach Score: {r['social_reach_score']}/100\n"
        f"  • Price Position: {price_position} ({r['price_vs_market']:+.1f}% vs market avg)\n"
        f"  • Competitors in Category: {r['competitor_count']}\n\n"
        f"**Past Promotion Performance:**\n{past_promos}\n\n"
        f"**AI Recommendation:**\n{recommendation}"
    )


@tool
def compare_promo_strategies(product_id: str) -> str:
    """Compare multiple promotion strategies and rank them for this product."""
    df = fetch_google_shopping_peanut_data()

    if df.empty:
        return "No products available."

    row = df[df["product_id"] == product_id].head(1)
    if row.empty:
        return f"No live product found for ID {product_id}."

    r = row.iloc[0]

    # Simulate different promo strategies
    strategies = {
        "BOGO (Buy One Get One)": {
            "lift": 28,
            "roi": 2.1,
            "best_for": "High review count",
            "cost": "Medium"
        },
        "10% Discount": {
            "lift": 15,
            "roi": 1.8,
            "best_for": "Price-sensitive segments",
            "cost": "Low"
        },
        "Influencer Campaign": {
            "lift": 35,
            "roi": 2.8,
            "best_for": "Low social reach score",
            "cost": "High"
        },
        "Bundle (2 SKUs)": {
            "lift": 22,
            "roi": 2.4,
            "best_for": "Cross-sell opportunities",
            "cost": "Medium"
        },
        "Free Sample Program": {
            "lift": 18,
            "roi": 2.0,
            "best_for": "New products",
            "cost": "High"
        },
        "Seasonal Campaign": {
            "lift": 25,
            "roi": 2.5,
            "best_for": "Relevant seasons",
            "cost": "Medium"
        }
    }

    # Score each strategy based on product profile
    scored = []
    for strategy, data in strategies.items():
        # Boost influencer if low social reach
        if strategy == "Influencer Campaign" and r["social_reach_score"] < 40:
            final_lift = data["lift"] * 1.2
            final_roi = data["roi"] * 1.15
        else:
            final_lift = data["lift"]
            final_roi = data["roi"]

        scored.append({
            "strategy": strategy,
            "estimated_lift": final_lift,
            "estimated_roi": final_roi,
            "cost": data["cost"],
            "best_for": data["best_for"]
        })

    # Sort by ROI
    scored = sorted(scored, key=lambda x: x["estimated_roi"], reverse=True)

    lines = [f"**Promo Strategy Ranking for: {r['title']}**\n"]
    for i, s in enumerate(scored, 1):
        lines.append(
            f"{i}. **{s['strategy']}** | "
            f"Est. Lift: {s['estimated_lift']:.0f}% | "
            f"ROI: {s['estimated_roi']:.2f}x | "
            f"Cost: {s['cost']}\n"
            f"   Best for: {s['best_for']}\n"
        )

    return "".join(lines)


# ===============================
# Build the Agent
# ===============================

def build_agent():
    """Return Enhanced Google Shopping FMCG Promotion Agent."""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=OPENAI_API_KEY
    )

    tools = [
        get_top_products,
        get_top_campaigns,
        analyze_product_promo,
        compare_promo_strategies
    ]

    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a retail and marketing analytics assistant focused on FMCG food products. "
                "You analyze Google Shopping market data, social reach metrics, past promotion performance, "
                "and competitor dynamics to recommend effective campaigns. "
                "Provide specific, actionable insights with lift%, ROI, and strategy justification."
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
