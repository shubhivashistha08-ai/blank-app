import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tweepy
from googleapiclient.discovery import build
import time

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Oreo Social Media Analytics",
    page_icon="🍪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# SECURE API CREDENTIALS
# ============================================
def get_secret(key):
    """Get secret from Streamlit secrets"""
    try:
        return st.secrets[key]
    except Exception as e:
        st.error(f"🚨 Error loading secret '{key}': {str(e)}")
        st.info("Please add secrets in Streamlit Cloud dashboard: Settings → Secrets")
        st.stop()

TWITTER_BEARER_TOKEN = get_secret("TWITTER_BEARER_TOKEN")
YOUTUBE_API_KEY = get_secret("YOUTUBE_API_KEY")

# ============================================
# CUSTOM STYLING
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #0051BA;
        margin-bottom: 5px;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 20px;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    .stMetric label {
        font-size: 0.9rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# OREO PRODUCT & FLAVOR CONFIGURATION
# ============================================
OREO_PRODUCTS = [
    "Oreo Original",
    "Oreo Double Stuf",
    "Oreo Thins",
    "Oreo Golden",
    "Oreo Mega Stuf",
    "Oreo Cakesters",
    "Oreo Bites"
]

OREO_FLAVORS = [
    "chocolate", "vanilla", "mint", "strawberry", "birthday cake",
    "peanut butter", "caramel", "red velvet", "golden", "lemon",
    "matcha", "coffee", "cinnamon", "pumpkin", "eggnog",
    "candy corn", "cherry cola", "coconut", "cookies and cream",
    "dark chocolate", "dulce de leche", "hazelnut", "irish cream",
    "key lime pie", "maple creme", "orange", "pistachio",
    "raspberry", "salted caramel", "s'mores", "tiramisu",
    "turkey stuffing", "watermelon", "winter", "chocolate mint"
]

# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_product_mentions(text, products):
    """Extract which Oreo product is mentioned"""
    if not isinstance(text, str):
        return "Other"
    text_lower = text.lower()
    
    # Check for specific product variants first
    for product in products:
        if product.lower() in text_lower:
            return product
    
    # If just "oreo" mentioned without variant
    if "oreo" in text_lower:
        return "Oreo Original"
    
    return "Other"

def extract_flavor_mentions(text, flavors):
    """Extract flavor keywords from text"""
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    found_flavors = []
    for flavor in flavors:
        if flavor.lower() in text_lower:
            found_flavors.append(flavor)
    return found_flavors if found_flavors else None

# ============================================
# DATA COLLECTION FUNCTIONS
# ============================================

@st.cache_data(ttl=3600)
def fetch_twitter_data(query, bearer_token, max_results=100):
    """Fetch Twitter data - last 7 days by default"""
    try:
        client = tweepy.Client(bearer_token=bearer_token)
        
        tweets = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=['created_at', 'public_metrics', 'lang'],
            expansions=['author_id']
        )
        
        data = []
        if tweets.data:
            for tweet in tweets.data:
                data.append({
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'like_count': tweet.public_metrics['like_count'],
                    'retweet_count': tweet.public_metrics['retweet_count'],
                    'reply_count': tweet.public_metrics['reply_count'],
                    'lang': tweet.lang
                })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        st.error(f"Twitter API Error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_youtube_data(query, api_key, max_results=50):
    """Fetch YouTube videos and comments - recent uploads"""
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            type='video',
            maxResults=max_results,
            order='date'
        ).execute()
        
        videos_data = []
        comments_data = []
        
        for item in search_response.get('items', []):
            video_id = item['id']['videoId']
            
            video_stats = youtube.videos().list(
                part='statistics,snippet',
                id=video_id
            ).execute()
            
            if video_stats['items']:
                stats = video_stats['items'][0]['statistics']
                snippet = video_stats['items'][0]['snippet']
                
                videos_data.append({
                    'video_id': video_id,
                    'title': snippet['title'],
                    'published_at': snippet['publishedAt'],
                    'view_count': int(stats.get('viewCount', 0)),
                    'like_count': int(stats.get('likeCount', 0)),
                    'comment_count': int(stats.get('commentCount', 0))
                })
                
                try:
                    comments_response = youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        maxResults=20
                    ).execute()
                    
                    for comment_item in comments_response.get('items', []):
                        comment = comment_item['snippet']['topLevelComment']['snippet']
                        comments_data.append({
                            'video_id': video_id,
                            'comment': comment['textDisplay'],
                            'like_count': comment['likeCount'],
                            'published_at': comment['publishedAt']
                        })
                except:
                    pass
        
        return pd.DataFrame(videos_data), pd.DataFrame(comments_data)
    
    except Exception as e:
        st.error(f"YouTube API Error: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# ============================================
# HEADER
# ============================================
st.markdown('<p class="main-header">🍪 Oreo Social Media Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-Time Product & Flavor Trend Tracking</p>', unsafe_allow_html=True)

# ============================================
# BUILD SEARCH QUERY
# ============================================
# Search for "Oreo" + specific variants
selected_products = ["Oreo Double Stuf", "Oreo Thins", "Oreo Golden", "Oreo"]
query = f"Oreo ({' OR '.join(['Double Stuf', 'Thins', 'Golden', 'Cakesters'])})"

# ============================================
# CREATE TWO SEPARATE PAGES
# ============================================
page = st.radio("", ["🐦 Twitter Analytics", "📺 YouTube Analytics"], horizontal=True)

st.markdown("---")

# ============================================
# TWITTER ANALYTICS PAGE
# ============================================
if page == "🐦 Twitter Analytics":
    
    with st.spinner("🔄 Fetching Twitter data..."):
        twitter_df = fetch_twitter_data(query, TWITTER_BEARER_TOKEN, max_results=100)
    
    if not twitter_df.empty:
        # Process data
        twitter_df['product'] = twitter_df['text'].apply(lambda x: extract_product_mentions(x, OREO_PRODUCTS))
        twitter_df['flavors'] = twitter_df['text'].apply(lambda x: extract_flavor_mentions(x, OREO_FLAVORS))
        twitter_df['date'] = pd.to_datetime(twitter_df['created_at']).dt.date
        twitter_df['month'] = pd.to_datetime(twitter_df['created_at']).dt.to_period('M')
        
        # ============================================
        # EXECUTIVE SUMMARY - COMPACT
        # ============================================
        st.markdown("### 📊 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_tweets = len(twitter_df)
            st.metric(
                label="📝 Total Tweets",
                value=f"{total_tweets:,}"
            )
        
        with col2:
            unique_products = twitter_df[twitter_df['product'] != 'Other']['product'].nunique()
            st.metric(
                label="🍪 Variants Mentioned",
                value=unique_products
            )
        
        with col3:
            most_consumed = twitter_df[twitter_df['product'] != 'Other']['product'].value_counts().index[0] if not twitter_df[twitter_df['product'] != 'Other'].empty else "Oreo Original"
            st.metric(
                label="🏆 Most Discussed",
                value=most_consumed.replace("Oreo ", "")
            )
        
        with col4:
            all_flavors = [flavor for flavors in twitter_df['flavors'].dropna() for flavor in flavors]
            popular_flavor = pd.Series(all_flavors).value_counts().index[0].capitalize() if all_flavors else "Chocolate"
            st.metric(
                label="🎨 Popular Flavor",
                value=popular_flavor
            )
        
        st.markdown("---")
        
        # ============================================
        # PRODUCT TREND OVER TIME
        # ============================================
        st.markdown("### 📈 Product Trends Over Time")
        
        # Product dropdown with better defaults
        available_products = twitter_df[twitter_df['product'] != 'Other']['product'].unique().tolist()
        if not available_products:
            available_products = ["Oreo Original"]
        
        selected_product_trend = st.selectbox(
            "Select Product to View Trend:",
            options=available_products,
            index=0
        )
        
        # Filter for selected product
        product_trend_df = twitter_df[twitter_df['product'] == selected_product_trend]
        
        if not product_trend_df.empty:
            daily_trend = product_trend_df.groupby('date').size().reset_index(name='mentions')
            
            fig_trend = px.line(
                daily_trend,
                x='date',
                y='mentions',
                title=f"{selected_product_trend} Mentions Over Time",
                markers=True
            )
            fig_trend.update_traces(line_color='#0051BA', marker=dict(size=8))
            fig_trend.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Number of Tweets",
                hovermode='x unified'
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info(f"No tweets found for {selected_product_trend}")
        
        # ============================================
        # PRODUCT COMPARISON
        # ============================================
        st.markdown("### 🔥 Product Variant Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Product Mentions")
            product_counts = twitter_df[twitter_df['product'] != 'Other']['product'].value_counts().reset_index()
            product_counts.columns = ['Product', 'Mentions']
            
            # Clean product names for display
            product_counts['Product'] = product_counts['Product'].str.replace("Oreo ", "")
            
            fig_bar = px.bar(
                product_counts,
                x='Product',
                y='Mentions',
                color='Mentions',
                color_continuous_scale='Blues',
                text='Mentions'
            )
            fig_bar.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.subheader("Product Share")
            fig_pie = px.pie(
                product_counts,
                values='Mentions',
                names='Product',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Blues
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # ============================================
        # FLAVOR INTELLIGENCE
        # ============================================
        st.markdown("### 🎨 Flavor Intelligence")
        
        all_flavors_data = []
        for idx, row in twitter_df.iterrows():
            if row['flavors'] and len(row['flavors']) > 0:
                for flavor in row['flavors']:
                    all_flavors_data.append({
                        'flavor': flavor.capitalize(),
                        'product': row['product']
                    })
        
        flavors_df = pd.DataFrame(all_flavors_data)
        
        if not flavors_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                flavor_counts = flavors_df['flavor'].value_counts().reset_index()
                flavor_counts.columns = ['Flavor', 'Mentions']
                
                fig_flavors = px.bar(
                    flavor_counts.head(10),
                    x='Flavor',
                    y='Mentions',
                    color='Mentions',
                    color_continuous_scale='Sunset',
                    text='Mentions',
                    title="Top 10 Most Popular Flavors"
                )
                fig_flavors.update_traces(texttemplate='%{text:,}', textposition='outside')
                fig_flavors.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_flavors, use_container_width=True)
            
            with col2:
                # Flavor by product breakdown
                flavor_product = flavors_df.groupby(['product', 'flavor']).size().reset_index(name='mentions')
                flavor_product['product'] = flavor_product['product'].str.replace("Oreo ", "")
                
                fig_flavor_product = px.sunburst(
                    flavor_product,
                    path=['product', 'flavor'],
                    values='mentions',
                    title="Flavor Distribution by Product Variant"
                )
                fig_flavor_product.update_layout(height=400)
                st.plotly_chart(fig_flavor_product, use_container_width=True)
        else:
            st.info("🎨 No flavor data detected in current tweets. Try selecting different products or refreshing data.")
            
            # Show example flavors users can search for
            st.markdown("**Popular Oreo Flavors to Look For:**")
            st.markdown("🍫 Chocolate • 🍦 Vanilla • 🍃 Mint • 🍓 Strawberry • 🎂 Birthday Cake • 🥜 Peanut Butter • 🍮 Caramel • 🧁 Red Velvet • 🎃 Pumpkin • ☕ Coffee")
    
    else:
        st.warning("⚠️ No Twitter data found. Please check API credentials or try again later.")

# ============================================
# YOUTUBE ANALYTICS PAGE
# ============================================
elif page == "📺 YouTube Analytics":
    
    with st.spinner("🔄 Fetching YouTube data..."):
        youtube_videos_df, youtube_comments_df = fetch_youtube_data(query, YOUTUBE_API_KEY, max_results=50)
    
    if not youtube_videos_df.empty or not youtube_comments_df.empty:
        
        # Process YouTube comments
        if not youtube_comments_df.empty:
            youtube_comments_df['product'] = youtube_comments_df['comment'].apply(lambda x: extract_product_mentions(x, OREO_PRODUCTS))
            youtube_comments_df['flavors'] = youtube_comments_df['comment'].apply(lambda x: extract_flavor_mentions(x, OREO_FLAVORS))
            youtube_comments_df['date'] = pd.to_datetime(youtube_comments_df['published_at']).dt.date
        
        # ============================================
        # EXECUTIVE SUMMARY - COMPACT
        # ============================================
        st.markdown("### 📊 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_videos = len(youtube_videos_df)
            st.metric(
                label="🎥 Total Videos",
                value=f"{total_videos:,}"
            )
        
        with col2:
            if not youtube_comments_df.empty:
                unique_products = youtube_comments_df[youtube_comments_df['product'] != 'Other']['product'].nunique()
                st.metric(
                    label="🍪 Variants Mentioned",
                    value=unique_products
                )
            else:
                st.metric(label="🍪 Variants Mentioned", value="0")
        
        with col3:
            if not youtube_comments_df.empty and not youtube_comments_df[youtube_comments_df['product'] != 'Other'].empty:
                most_consumed = youtube_comments_df[youtube_comments_df['product'] != 'Other']['product'].value_counts().index[0]
                st.metric(
                    label="🏆 Most Discussed",
                    value=most_consumed.replace("Oreo ", "")
                )
            else:
                st.metric(label="🏆 Most Discussed", value="Original")
        
        with col4:
            if not youtube_comments_df.empty:
                all_flavors = [flavor for flavors in youtube_comments_df['flavors'].dropna() for flavor in flavors]
                popular_flavor = pd.Series(all_flavors).value_counts().index[0].capitalize() if all_flavors else "Chocolate"
                st.metric(
                    label="🎨 Popular Flavor",
                    value=popular_flavor
                )
            else:
                st.metric(label="🎨 Popular Flavor", value="Chocolate")
        
        st.markdown("---")
        
        # ============================================
        # VIDEO PERFORMANCE
        # ============================================
        st.markdown("### 🎥 Top Oreo Videos by Views")
        
        top_videos = youtube_videos_df.nlargest(10, 'view_count')
        
        fig_videos = px.bar(
            top_videos,
            x='view_count',
            y='title',
            orientation='h',
            color='view_count',
            color_continuous_scale='Blues',
            text='view_count'
        )
        fig_videos.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_videos.update_layout(
            height=500,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Views",
            yaxis_title=""
        )
        st.plotly_chart(fig_videos, use_container_width=True)
        
        # ============================================
        # PRODUCT MENTIONS IN COMMENTS
        # ============================================
        if not youtube_comments_df.empty:
            st.markdown("### 🍪 Product Variant Mentions in Comments")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Product Distribution")
                product_counts = youtube_comments_df[youtube_comments_df['product'] != 'Other']['product'].value_counts().reset_index()
                product_counts.columns = ['Product', 'Mentions']
                product_counts['Product'] = product_counts['Product'].str.replace("Oreo ", "")
                
                if not product_counts.empty:
                    fig_bar = px.bar(
                        product_counts,
                        x='Product',
                        y='Mentions',
                        color='Mentions',
                        color_continuous_scale='Blues',
                        text='Mentions'
                    )
                    fig_bar.update_traces(texttemplate='%{text:,}', textposition='outside')
                    fig_bar.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                st.subheader("Product Share")
                if not product_counts.empty:
                    fig_pie = px.pie(
                        product_counts,
                        values='Mentions',
                        names='Product',
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Blues
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # ============================================
            # FLAVOR INTELLIGENCE
            # ============================================
            st.markdown("### 🎨 Flavor Intelligence")
            
            all_flavors_data = []
            for idx, row in youtube_comments_df.iterrows():
                if row['flavors'] and len(row['flavors']) > 0:
                    for flavor in row['flavors']:
                        all_flavors_data.append({
                            'flavor': flavor.capitalize()
                        })
            
            flavors_df = pd.DataFrame(all_flavors_data)
            
            if not flavors_df.empty:
                flavor_counts = flavors_df['flavor'].value_counts().reset_index()
                flavor_counts.columns = ['Flavor', 'Mentions']
                
                fig_flavors = px.bar(
                    flavor_counts.head(10),
                    x='Flavor',
                    y='Mentions',
                    color='Mentions',
                    color_continuous_scale='Sunset',
                    text='Mentions',
                    title="Top 10 Most Popular Flavors in Comments"
                )
                fig_flavors.update_traces(texttemplate='%{text:,}', textposition='outside')
                fig_flavors.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_flavors, use_container_width=True)
            else:
                st.info("🎨 No flavor data detected in comments")
                st.markdown("**Popular Oreo Flavors to Look For:**")
                st.markdown("🍫 Chocolate • 🍦 Vanilla • 🍃 Mint • 🍓 Strawberry • 🎂 Birthday Cake • 🥜 Peanut Butter • 🍮 Caramel • 🧁 Red Velvet")
    
    else:
        st.warning("⚠️ No YouTube data found. Please check API credentials or try again later.")

# ============================================
# AUTO-REFRESH (Bottom of page)
# ============================================
st.markdown("---")
st.markdown("### 🔄 Auto-Refresh Settings")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)

with col2:
    refresh_interval = st.slider("Interval (seconds)", 30, 300, 120)

if auto_refresh:
    st.info(f"🔄 Dashboard will refresh every {refresh_interval} seconds")
    time.sleep(refresh_interval)
    st.rerun()

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(f"**🍪 Oreo Social Media Analytics Dashboard | Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
