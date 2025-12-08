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
    "See sales estimates, social reach, promotion history, and competitor dynamics to plan winning campaigns."
)

# ------------------------------------------------
# Data refresh control
# ------------------------------------------------
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
st.markdown("### 📊 Market Overview & Key Metrics")

if df.empty:
    st.warning("❌ No data available. Click **Refresh market data** above.")
else:
    # KPIs
    total_products = df["product_id"].nunique()
    total_est_sales = df["sales_estimate"].sum()
    avg_social_reach = df["social_reach_score"].mean()
    avg_campaign_potential = df["campaign_potential"].mean()
    unique_brands = df["brand"].nunique()
    avg_rating = df["rating"].dropna().mean()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("🛍️ Products", total_products)
    col2.metric("📈 Total Est. Sales", f"{total_est_sales:,} units/mo")
    col3.metric("📱 Avg Social Reach", f"{avg_social_reach:.0f}/100")
    col4.metric("🚀 Campaign Potential", f"{avg_campaign_potential:.0f}%")
    col5.metric("⭐ Avg Rating", f"{avg_rating:.2f}")
    col6.metric("🏢 Brands", unique_brands)

    # ================================================
    # SECTION 2: MARKET ANALYSIS CHARTS
    # ================================================
    st.markdown("---")
    st.markdown("### 📉 Market Analysis")

    col_brand_sales, col_social_reach = st.columns(2)

    # LEFT: Brand Sales & Rating Analysis
    with col_brand_sales:
        st.markdown("#### 📈 Brand Sales Estimate & Quality")

        brand_stats = df.groupby("brand").agg({
            "sales_estimate": "sum",
            "rating": "mean"
        }).rename(columns={
            "sales_estimate": "total_sales",
            "rating": "avg_rating"
        }).sort_values("total_sales", ascending=False).head(10)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0a0e27')
        ax1.set_facecolor('#0a0e27')

        # Bar for sales
        bars = ax1.bar(
            range(len(brand_stats)),
            brand_stats["total_sales"],
            color="#0d7377",
            alpha=0.8,
            label="Est. Monthly Sales",
            edgecolor='white',
            linewidth=1.5
        )

        # Value labels on bars
        for i, val in enumerate(brand_stats["total_sales"]):
            ax1.text(i, val + 20, f"{int(val)}", ha='center', va='bottom',
                    color='white', fontweight='bold', fontsize=8)

        ax1.set_ylabel("Est. Monthly Sales (units)", color="white", fontsize=10, fontweight="bold")
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
        ax2.set_facecolor('#0a0e27')
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
                  facecolor='#0a0e27', edgecolor='white', labelcolor='white', fontsize=8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # RIGHT: Social Reach Distribution
    with col_social_reach:
        st.markdown("#### 📱 Social Reach Distribution")

        # Create scatter: Campaign Potential vs Social Reach (sized by sales)
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0a0e27')
        ax.set_facecolor('#0a0e27')

        scatter = ax.scatter(
            df["social_reach_score"],
            df["campaign_potential"],
            s=df["sales_estimate"] / 2,  # Size by sales
            c=df["rating"].fillna(3),
            cmap="viridis",
            alpha=0.6,
            edgecolors='white',
            linewidth=0.5
        )

        ax.set_xlabel("Social Reach Score (0-100)", color="white", fontsize=10, fontweight="bold")
        ax.set_ylabel("Campaign Potential (%)", color="white", fontsize=10, fontweight="bold")
        ax.tick_params(axis='both', labelcolor="white", labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.grid(alpha=0.15, color='white', linestyle='--')

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Rating", color="white", fontsize=9, fontweight="bold")
        cbar.ax.tick_params(labelcolor="white", labelsize=8)

        # Add legend for bubble size
        bubble_sizes = [50, 200, 500]
        bubble_labels = ["Low Sales", "Med Sales", "High Sales"]
        for size, label in zip(bubble_sizes, bubble_labels):
            ax.scatter([], [], s=size/2, c='#14b8a6', alpha=0.6, edgecolors='white', label=label)
        ax.legend(loc='upper left', facecolor='#0a0e27', edgecolor='white',
                 labelcolor='white', fontsize=8, scatterpoints=1)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # ================================================
    # SECTION 3: CAMPAIGN PERFORMANCE & INSIGHTS
    # ================================================
    st.markdown("---")
    st.markdown("### 🎯 Campaign Performance & Top Opportunities")

    col_top_campaigns, col_price_pos = st.columns(2)

    with col_top_campaigns:
        st.markdown("#### 🚀 Products with Highest Campaign Potential")
        top_campaigns = df.nlargest(5, "campaign_potential")[
            ["title", "brand", "campaign_potential", "social_reach_score", "sales_estimate"]
        ].reset_index(drop=True)

        # Format for display
        display_df = top_campaigns.copy()
        display_df.columns = ["Product", "Brand", "Potential %", "Social Score", "Est. Sales/mo"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with col_price_pos:
        st.markdown("#### 💰 Price Positioning Strategy")

        price_bins = pd.cut(df["price_vs_market"], bins=5)
        price_dist = df.groupby(price_bins)["product_id"].count()

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0a0e27')
        ax.set_facecolor('#0a0e27')

        colors = ['#c0152f', '#e67961', '#14b8a6', '#0d7377', '#134252']
        bars = ax.bar(range(len(price_dist)), price_dist.values, color=colors[:len(price_dist)],
                     alpha=0.8, edgecolor='white', linewidth=1.5)

        for i, val in enumerate(price_dist.values):
            ax.text(i, val + 0.1, str(int(val)), ha='center', va='bottom',
                   color='white', fontweight='bold', fontsize=9)

        ax.set_ylabel("Product Count", color="white", fontsize=10, fontweight="bold")
        ax.set_xlabel("Price Position vs Market Avg", color="white", fontsize=10, fontweight="bold")
        ax.tick_params(axis='both', labelcolor="white", labelsize=8)
        ax.set_xticklabels([f"{i}" for i in range(len(price_dist))], color="white")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.grid(axis='y', alpha=0.15, color='white', linestyle='--')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # ================================================
    # SECTION 4: PRODUCT SELECTOR & DETAILED ANALYSIS
    # ================================================
    st.markdown("---")
    st.markdown("### 🔍 Detailed Product Analysis")

    if not df.empty:
        # Product selector - CLEAN: Show only product title
        product_options = df[
            ["product_id", "title", "brand", "rating", "review_count",
             "sales_estimate", "social_reach_score", "campaign_potential", "price"]
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
            st.metric("⭐ Rating", f"{selected_prod['rating']:.2f}")

        with col_card3:
            st.metric("💬 Reviews", int(selected_prod["review_count"]))

        with col_card4:
            st.metric("📈 Est. Sales", f"{int(selected_prod['sales_estimate'])} units/mo")

        # Additional product insights
        col_details1, col_details2, col_details3 = st.columns(3)

        with col_details1:
            st.metric("📱 Social Score", f"{int(selected_prod['social_reach_score'])}/100")

        with col_details2:
            st.metric("🚀 Campaign Potential", f"{int(selected_prod['campaign_potential'])}%")

        with col_details3:
            price_pos = selected_prod["price"]
            st.metric("💵 Price", f"${price_pos:.2f}" if price_pos else "N/A")

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
                f"rating: {selected_prod['rating']:.2f}, reviews: {int(selected_prod['review_count'])}, "
                f"est. sales: {int(selected_prod['sales_estimate'])} units/mo, "
                f"social score: {int(selected_prod['social_reach_score'])}/100), "
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

st.markdown("---")
st.caption("Data from Google Shopping API. Campaign metrics are AI-estimated proxies based on market positioning, social signals, and historical performance patterns.")
