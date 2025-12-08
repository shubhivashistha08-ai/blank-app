import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from agent import build_agent, fetch_google_shopping_peanut_data

st.set_page_config(page_title="FMCG Campaign Intelligence", layout="wide")

# Set dark theme
st.markdown("""
    <style>
        body { background-color: #0a0e27; }
        .stMetric { background-color: rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    "<h2 style='text-align:center; color:white;'>🎯 FMCG Campaign & Promotion Intelligence</h2>",
    unsafe_allow_html=True,
)

st.write(
    "Analyze live Google Shopping market data for peanut butter products. "
    "See brand performance, review trends, and competitive dynamics to plan winning campaigns."
)

# ================================================
# Helper Function: Format Large Numbers
# ================================================
def format_large_number(num, unit=""):
    """Format numbers as 4.2M, 1.5B, etc. with unit"""
    if num is None or pd.isna(num):
        return "N/A"
    
    num = float(num)
    if abs(num) >= 1e9:
        return f"{num / 1e9:.1f}B {unit}".strip()
    elif abs(num) >= 1e6:
        return f"{num / 1e6:.1f}M {unit}".strip()
    elif abs(num) >= 1e3:
        return f"{num / 1e3:.1f}K {unit}".strip()
    else:
        return f"{num:.0f} {unit}".strip()

# ================================================
# Data refresh control
# ================================================
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

refresh_clicked = st.button("🔄 Refresh market data")

if refresh_clicked:
    with st.spinner("Fetching latest market data..."):
        st.session_state.df = fetch_google_shopping_peanut_data()

df = st.session_state.df

# ================================================
# SECTION 1: MARKET OVERVIEW & KEY KPIs
# ================================================
st.markdown("---")
st.markdown("### 📊 Market Overview")

if df.empty:
    st.warning("❌ No data available. Click **Refresh market data** above.")
else:
    # Safe column access
    total_products = df["product_id"].nunique() if "product_id" in df.columns else 0
    total_sales_proxy = df["sales_proxy"].sum() if "sales_proxy" in df.columns else 0
    unique_brands = df["brand"].nunique() if "brand" in df.columns else 0
    avg_rating = df["rating"].dropna().mean() if "rating" in df.columns else 0
    avg_reviews = df["review_count"].dropna().mean() if "review_count" in df.columns else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🛍️ Products", total_products)
    col2.metric("📈 Total Units Proxy", format_large_number(total_sales_proxy, "units"))
    col3.metric("⭐ Avg Rating", f"{avg_rating:.2f}" if avg_rating > 0 else "N/A")
    col4.metric("💬 Avg Reviews", f"{avg_reviews:.0f}" if avg_reviews > 0 else "N/A")
    col5.metric("🏢 Brands", unique_brands)

    # ================================================
    # SECTION 2: MARKET ANALYSIS CHARTS
    # ================================================
    st.markdown("---")
    st.markdown("### 📊 Brand Analysis")

    col_brand_sales, col_brand_pie = st.columns(2)

    # LEFT: Brand Sales & Rating (Dark background)
    with col_brand_sales:
        st.markdown("#### 📈 Top Brands - Sales & Quality")

        brand_stats = df.groupby("brand").agg({
            "sales_proxy": "sum",
            "rating": "mean"
        }).rename(columns={
            "sales_proxy": "total_sales",
            "rating": "avg_rating"
        }).sort_values("total_sales", ascending=False).head(10)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a2e')  # Dark background
        ax1.set_facecolor('#1a1a2e')  # Dark background

        # Bar for sales
        bars = ax1.bar(
            range(len(brand_stats)),
            brand_stats["total_sales"],
            color="#0d7377",
            alpha=0.85,
            label="Sales Proxy (Units)",
            edgecolor='white',
            linewidth=1.5
        )

        # Value labels on bars
        for i, val in enumerate(brand_stats["total_sales"]):
            ax1.text(i, val + 50, f"{format_large_number(val)}", ha='center', va='bottom',
                    color='white', fontweight='bold', fontsize=8)

        ax1.set_ylabel("Sales Proxy (Units)", color="white", fontsize=10, fontweight="bold")
        ax1.tick_params(axis='y', labelcolor="white", labelsize=8)
        ax1.set_xticks(range(len(brand_stats)))
        ax1.set_xticklabels(brand_stats.index, rotation=45, ha="right", color="white", fontsize=8)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('#0d7377')
        ax1.spines['bottom'].set_color('white')
        ax1.grid(axis='y', alpha=0.15, color='white', linestyle='--')

        # Line for rating on secondary axis
        ax2 = ax1.twinx()
        ax2.set_facecolor('#1a1a2e')
        line = ax2.plot(
            range(len(brand_stats)),
            brand_stats["avg_rating"],
            color="#14b8a6",
            marker="o",
            linewidth=3,
            markersize=7,
            label="Avg Rating"
        )

        for i, val in enumerate(brand_stats["avg_rating"]):
            ax2.text(i, val + 0.12, f"{val:.2f}", ha='center', va='bottom',
                    color='#14b8a6', fontweight='bold', fontsize=7)

        ax2.set_ylabel("Avg Rating", color="#14b8a6", fontsize=10, fontweight="bold")
        ax2.tick_params(axis='y', labelcolor="#14b8a6", labelsize=8)
        ax2.set_ylim([0, 5.5])
        ax2.spines['right'].set_color('#14b8a6')
        ax2.spines['top'].set_visible(False)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                  facecolor='#1a1a2e', edgecolor='white', labelcolor='white', fontsize=8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # RIGHT: Brand Market Share Pie (Top 5 + Others)
    with col_brand_pie:
        st.markdown("#### 🥧 Brand Market Share (Top 5)")

        brand_share = df["brand"].value_counts().head(5)
        other_count = df["brand"].value_counts().iloc[5:].sum() if len(df["brand"].value_counts()) > 5 else 0

        if other_count > 0:
            brand_share = pd.concat([brand_share, pd.Series({"Others": other_count})])

        fig_pie, ax_pie = plt.subplots(figsize=(7, 6))
        fig_pie.patch.set_facecolor('#1a1a2e')

        # Green & Blue color palette
        colors_palette = ["#0d7377", "#14b8a6", "#06b6d4", "#22d3ee", "#67e8f9", "#cbd5e1"]

        wedges, texts, autotexts = ax_pie.pie(
            brand_share.values,
            labels=brand_share.index,
            autopct='%1.1f%%',
            colors=colors_palette[:len(brand_share)],
            startangle=90,
            textprops={'fontsize': 10, 'weight': 'bold'}
        )

        # Format text colors
        for text in texts:
            text.set_color('white')
            text.set_fontsize(10)
            text.set_fontweight('bold')

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')

        plt.tight_layout()
        st.pyplot(fig_pie, use_container_width=True)

    # ================================================
    # SECTION 3: TOP PRODUCTS BY REVIEWS
    # ================================================
    st.markdown("---")
    st.markdown("### 🌟 Top Products by Reviews")

    col_most_reviewed, col_top_rated = st.columns(2)

    with col_most_reviewed:
        st.markdown("#### 💬 Most Reviewed Products")
        if "review_count" in df.columns and "rating" in df.columns:
            most_reviewed = df.nlargest(5, "review_count")[["title", "brand", "review_count", "rating"]].reset_index(drop=True)
            display_df = most_reviewed.copy()
            display_df.columns = ["Product", "Brand", "Reviews", "Rating"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("Review data not available")

    with col_top_rated:
        st.markdown("#### ⭐ Highest Rated Products")
        if "rating" in df.columns and "review_count" in df.columns:
            # Top rated with min 10 reviews for credibility
            top_rated = df[df["review_count"] >= 10].nlargest(5, "rating")[["title", "brand", "rating", "review_count"]].reset_index(drop=True)
            if not top_rated.empty:
                display_df = top_rated.copy()
                display_df.columns = ["Product", "Brand", "Rating", "Reviews"]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No products with sufficient reviews")
        else:
            st.info("Rating data not available")

    # ================================================
    # SECTION 4: DETAILED PRODUCT ANALYSIS
    # ================================================
    st.markdown("---")
    st.markdown("### 🔍 Detailed Product Analysis")

    if not df.empty:
        # Product selector - CLEAN: Show only product title
        product_options = df[
            ["product_id", "title", "brand", "rating", "review_count"]
        ].drop_duplicates().reset_index(drop=True)

        selected_idx = st.selectbox(
            "Select a product for detailed analysis:",
            options=range(len(product_options)),
            format_func=lambda i: f"{product_options.iloc[i]['title']}",
            key="product_selector"
        )

        selected_prod = product_options.iloc[selected_idx]

        # ✅ CLEAN PRODUCT DETAILS CARD (below dropdown)
        col_card1, col_card2, col_card3, col_card4 = st.columns(4)

        with col_card1:
            st.metric("🏢 Brand", selected_prod["brand"])

        with col_card2:
            st.metric("⭐ Rating", f"{selected_prod['rating']:.2f}" if pd.notna(selected_prod['rating']) else "N/A")

        with col_card3:
            st.metric("💬 Reviews", int(selected_prod["review_count"]) if pd.notna(selected_prod["review_count"]) else 0)

        with col_card4:
            # Get search position from main df
            search_pos = df[df["product_id"] == selected_prod["product_id"]]["search_position"].values
            if len(search_pos) > 0:
                st.metric("🔍 Search Position", f"#{int(search_pos[0])}")
            else:
                st.metric("🔍 Search Position", "N/A")

        st.info("📊 **Note**: Additional data fields (estimated sales, social score, campaign potential, price) require additional API integration. See 'API Integration' section below.")

    else:
        st.warning("Refresh data first to select a product.")
        selected_prod = None

    # ================================================
    # SECTION 5: AI CAMPAIGN ADVISOR CHAT
    # ================================================
    st.markdown("---")
    st.markdown("### 💬 AI Campaign Advisor")

    if "agent" not in st.session_state:
        st.session_state.agent = build_agent()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input(
        "Ask campaign strategies (BOGO, influencer, samples, seasonal, pricing)...",
        key="chat_input"
    ):
        # Add product context if selected
        if selected_prod is not None:
            context_prompt = (
                f"For product '{selected_prod['title']}' (brand: {selected_prod['brand']}, "
                f"rating: {selected_prod['rating']:.2f}, reviews: {int(selected_prod['review_count'])}), "
                f"provide campaign strategy. User asks: {prompt}"
            )
        else:
            context_prompt = prompt

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing campaign options..."):
                try:
                    result = st.session_state.agent({"input": context_prompt})
                    answer = getattr(result, "content", str(result))
                except Exception as e:
                    error_msg = str(e)
                    if "insufficient_quota" in error_msg.lower():
                        answer = (
                            "⚠️ **OpenAI API Quota Exceeded**\n\n"
                            "Add credits at [OpenAI Billing](https://platform.openai.com/account/billing/overview)"
                        )
                    else:
                        answer = f"❌ Error: {error_msg}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

# ================================================
# SECTION 6: API INTEGRATION GUIDE
# ================================================
st.markdown("---")
st.markdown("### 📡 API Integration Options")

with st.expander("ℹ️ Add Missing Data (Estimated Sales, Price, Social Metrics)"):
    st.markdown("""
    ### Available APIs for Enhanced Data

    #### 1. **Product Pricing & Availability**
    - **Amazon Product Advertising API** (Free, affiliate links required)
      - Get product prices, availability
      - Limited to affiliate products
      
    - **RealData API** (Paid - FMCG focused)
      - Zepto, FirstCry FMCG prices
      - Real-time inventory & pricing
      - Starts at ~$10/month
      
    - **SerpAPI Enhancement** (Already using, can add more fields)
      - Extract prices from search results
      - No additional setup needed

    #### 2. **Sales Volume Estimation**
    - **Amazon Selling Partner API** (For sellers)
      - Real sales metrics if you're an Amazon seller
      - Requires seller account & credentials
      
    - **Synthetic Estimation** (Current approach)
      - Based on search position + ratings + reviews
      - Free, no API calls needed

    #### 3. **Social Media Metrics**
    - **Twitter API** (Paid - $100-$200/month)
      - Track brand mentions & sentiment
      - Requires approval
      
    - **Manual Integration** (Budget-friendly)
      - Script to collect hashtag mentions weekly
      - Store in database

    #### 4. **Competitor Price Tracking**
    - **Keepa API** ($15/month)
      - Amazon price history
      - Great for competitor tracking
      
    - **CamelCamelCamel** (Free)
      - Amazon price alerts
      - Historical data available

    ### Recommendation for Your Use Case:

    **Phase 1 (Current)**: Use SerpAPI for Google Shopping + synthetic estimation ✅

    **Phase 2 (Next)**: Add Keepa API for Amazon competitor prices (~$15/month)

    **Phase 3 (Later)**: Integrate Twitter API for brand social metrics (~$100/month)

    ### To Add Keepa API:
    ```python
    # In agent.py, add:
    import requests
    
    def get_keepa_price_history(asin):
        url = f"https://api.keepa.com/product"
        params = {
            "ASIN": asin,
            "key": KEEPA_API_KEY,
            "stats": "90"  # Last 90 days
        }
        resp = requests.get(url, params=params)
        return resp.json()
    ```

    Let me know which API you'd like to integrate! 🚀
    """)

st.markdown("---")
st.caption("Data from Google Shopping API via SerpAPI. Charts show brand market share, product quality, and review metrics.")
