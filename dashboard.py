import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="E-commerce Market Intelligence Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        padding: 10px;
        border-radius: 5px;
    }
    .upload-info {
        padding: 1rem;
        background-color: #e3f2fd;
        border-radius: 5px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
        color: #1E2125;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df, "uploaded"
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None, None
    else:
        try:
            df = pd.read_csv('Clean_dataset.csv')
            return df, "default"
        except:
            return None, None

@st.cache_resource
def create_tfidf_model(_df):
    if 'cleaned_name' not in _df.columns:
        st.warning("Column 'cleaned_name' not found. Similarity feature unavailable.")
        return None, None
    
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(_df['cleaned_name'])
    return vectorizer, tfidf_matrix

def validate_dataset(df):
    required_columns = ['Name', 'Brand', 'Price', 'Sold', 'Rating', 'Location', 'Store']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, missing_columns
    return True, []

def find_similar_products(df, tfidf_matrix, product_index, top_n=5):
    target_vector = tfidf_matrix[product_index]
    sim_scores = cosine_similarity(target_vector, tfidf_matrix)[0]
    similar_indices = sim_scores.argsort()[::-1][1:top_n+1]
    similar_products = df.iloc[similar_indices].copy()
    similar_products['Similarity_Score'] = sim_scores[similar_indices]
    return similar_products

st.sidebar.title("📁 Data Source")

data_source = st.sidebar.radio(
    "Select data source:",
    ["📂 Use Default File", "⬆️ Upload CSV File"],
    help="Choose default file or upload new CSV"
)

uploaded_file = None
if data_source == "⬆️ Upload CSV File":
    st.sidebar.markdown("""
    <div class='upload-info'>
    <b>📋 File Format:</b><br>
    • Format: CSV<br>
    • Encoding: UTF-8<br>
    • Required columns: Name, Brand, Price, Sold, Rating, Location, Store
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload preprocessed file"
    )
    
    if uploaded_file is not None:
        file_size = uploaded_file.size / 1024
        st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
        st.sidebar.info(f"📊 Size: {file_size:.2f} KB")

df, source_type = load_data(uploaded_file)

if df is None:
    st.title("🔍 E-commerce Market Intelligence Dashboard")
    st.error("❌ Data not found!")
    
    st.markdown("""
    ### 📌 How to Use the Dashboard:
    
    **Option 1: Upload File**
    1. Select "⬆️ Upload CSV File" in the sidebar
    2. Upload your preprocessed `Clean_dataset.csv` file
    3. Dashboard will automatically process the data
    
    **Option 2: Use Default File**
    1. Make sure `Clean_dataset.csv` is in the same folder as `dashboard.py`
    2. Select "📂 Use Default File" in the sidebar
    
    ---
    
    ### 📋 Required File Format:
    
    **Required Columns:**
    - Name (product name)
    - Brand (any brand/competitor)
    - Price (price)
    - Sold (units sold)
    - Rating (product rating)
    - Location (city/location)
    - Store (store name)
    
    **Optional Columns (for additional features):**
    - cleaned_name (for product similarity)
    - Segment / Segment_Label (for segmentation analysis)
    - Price_scaled, Sold_scaled (for visualization)
    
    ---
    
    ### 💡 Tips:
    - File must be in CSV format
    - Encoding: UTF-8
    - Works with any brand/competitor data
    - Ensure preprocessing is completed according to your pipeline
    """)
    
    with st.expander("📄 See Data Format Example"):
        sample_data = {
            'Name': ['Nike Air Max 90', 'Adidas Samba OG', 'Puma Suede Classic'],
            'Brand': ['Nike', 'Adidas', 'Puma'],
            'Price': [1500000, 1200000, 900000],
            'Sold': [150, 200, 180],
            'Rating': [4.8, 4.7, 4.6],
            'Location': ['Jakarta', 'Surabaya', 'Bandung'],
            'Store': ['Nike Official', 'Adidas Store', 'Puma Store']
        }
        st.dataframe(pd.DataFrame(sample_data))
    
    st.stop()

is_valid, missing_cols = validate_dataset(df)

if not is_valid:
    st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
    st.info("Make sure your CSV file has been properly preprocessed.")
    st.stop()

if source_type == "uploaded":
    st.sidebar.success("✅ Using uploaded file")
else:
    st.sidebar.info("📂 Using default file")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.metric("Total Records", f"{len(df):,}")
st.sidebar.metric("Total Columns", len(df.columns))

if 'Brand' in df.columns:
    brand_counts = df['Brand'].value_counts()
    st.sidebar.markdown("**Brand Distribution:**")
    for brand, count in brand_counts.items():
        st.sidebar.write(f"• {brand}: {count:,} products")

vectorizer, tfidf_matrix = create_tfidf_model(df)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔎 Filter Data")

selected_brands = st.sidebar.multiselect(
    "Brand:",
    options=sorted(df['Brand'].unique()),
    default=df['Brand'].unique()
)

price_range = st.sidebar.slider(
    "Price Range (Rp):",
    min_value=int(df['Price'].min()),
    max_value=int(df['Price'].quantile(0.99)),
    value=(int(df['Price'].min()), int(df['Price'].quantile(0.99)))
)

df_filtered = df[
    (df['Brand'].isin(selected_brands)) &
    (df['Price'] >= price_range[0]) &
    (df['Price'] <= price_range[1])
].copy()

if len(df_filtered) < len(df):
    st.sidebar.info(f"📊 Showing {len(df_filtered):,} of {len(df):,} products")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Export Data")

csv = df_filtered.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Download Filtered Data",
    data=csv,
    file_name="filtered_data.csv",
    mime="text/csv"
)

st.sidebar.markdown("---")
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["📊 Overview", "🏷️ Market Segmentation", "🏪 Store Analysis", "🔎 Product Similarity", "📈 Price Analysis", "⚔️ Competitor Analysis"]
)

if page == "📊 Overview":
    st.title("🔍 E-commerce Market Intelligence Dashboard")
    st.markdown("### Market Overview & Key Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Products", f"{len(df_filtered):,}")
    with col2:
        st.metric("Total Sales", f"{df_filtered['Sold'].sum():,.0f}")
    with col3:
        st.metric("Avg Price", f"Rp {df_filtered['Price'].mean():,.0f}")
    with col4:
        st.metric("Avg Rating", f"{df_filtered['Rating'].mean():.2f} ⭐")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Brand Distribution")
        brand_counts = df_filtered['Brand'].value_counts()
        colors = px.colors.qualitative.Set3[:len(brand_counts)]
        fig = px.pie(
            values=brand_counts.values,
            names=brand_counts.index,
            color_discrete_sequence=colors,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Price Distribution by Brand")
        fig = px.box(
            df_filtered,
            x='Brand',
            y='Price',
            color='Brand',
            color_discrete_sequence=colors
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 10 Best Selling Products")
    top_sellers = df_filtered.nlargest(10, 'Sold')[['Name', 'Brand', 'Price', 'Sold', 'Rating']]
    fig = px.bar(
        top_sellers,
        x='Sold',
        y='Name',
        color='Brand',
        orientation='h',
        color_discrete_sequence=colors,
        text='Sold'
    )
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 10 Cities by Product Count")
    city_counts = df_filtered['Location'].value_counts().head(10)
    fig = px.bar(
        x=city_counts.values,
        y=city_counts.index,
        orientation='h',
        labels={'x': 'Product Count', 'y': 'City'},
        color=city_counts.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

elif page == "🏷️ Market Segmentation":
    st.title("🏷️ Market Segmentation")
    st.markdown("### Product segment analysis based on NMF Topic Modeling")
    
    if 'Segment_Label' not in df_filtered.columns and 'Segment' not in df_filtered.columns:
        st.warning("⚠️ Segmentation data not available in this dataset.")
        st.info("""
        **To enable segmentation analysis:**
        1. Ensure complete preprocessing (run all cells in preprocessing script)
        2. Dataset must have 'Segment' or 'Segment_Label' column
        3. Upload complete CSV file
        """)
    else:
        segment_col = 'Segment_Label' if 'Segment_Label' in df_filtered.columns else 'Segment'
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Segment Distribution")
            segment_counts = df_filtered[segment_col].value_counts()
            fig = px.bar(
                x=segment_counts.index,
                y=segment_counts.values,
                labels={'x': 'Segment', 'y': 'Product Count'},
                color=segment_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Brand Distribution per Segment")
            segment_brand = pd.crosstab(df_filtered[segment_col], df_filtered['Brand'])
            colors = px.colors.qualitative.Set3[:len(segment_brand.columns)]
            fig = px.bar(
                segment_brand,
                barmode='group',
                color_discrete_sequence=colors
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Segment Characteristics")
        segment_stats = df_filtered.groupby(segment_col).agg({
            'Price': 'mean',
            'Sold': 'sum',
            'Rating': 'mean',
            'Name': 'count'
        }).round(2)
        segment_stats.columns = ['Avg Price', 'Total Sold', 'Avg Rating', 'Product Count']
        segment_stats = segment_stats.sort_values('Total Sold', ascending=False)
        
        st.dataframe(
            segment_stats.style.format({
                'Avg Price': 'Rp {:,.0f}',
                'Total Sold': '{:,.0f}',
                'Avg Rating': '{:.2f}'
            }).background_gradient(cmap='YlOrRd', subset=['Avg Price', 'Total Sold']),
            use_container_width=True
        )
        
        st.subheader("Price vs Sales by Segment")
        fig = px.scatter(
            df_filtered,
            x='Price',
            y='Sold',
            color=segment_col,
            size='Rating',
            hover_data=['Name', 'Brand'],
            opacity=0.6
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

elif page == "🏪 Store Analysis":
    st.title("🏪 Store Analysis")
    st.markdown("### Store Performance & Analysis")
    
    store_df = df_filtered.groupby('Store').agg({
        'Price': 'mean',
        'Sold': 'sum',
        'Rating': 'mean',
        'Name': 'count'
    }).reset_index()
    store_df.columns = ['Store', 'Avg_Price', 'Total_Sold', 'Avg_Rating', 'Product_Count']
    
    st.subheader("Top 15 Stores by Sales")
    top_stores = store_df.nlargest(15, 'Total_Sold')
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Total Sales", "Average Price")
    )
    
    fig.add_trace(
        go.Bar(x=top_stores['Total_Sold'], y=top_stores['Store'], orientation='h', name='Sales'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=top_stores['Avg_Price'], y=top_stores['Store'], orientation='h', name='Price', marker_color='coral'),
        row=1, col=2
    )
    
    fig.update_layout(height=500, showlegend=False)
    fig.update_yaxes(tickfont=dict(size=9))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Store Performance Matrix")
    fig = px.scatter(
        store_df,
        x='Avg_Price',
        y='Total_Sold',
        size='Product_Count',
        color='Avg_Rating',
        hover_data=['Store'],
        color_continuous_scale='RdYlGn',
        size_max=30
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Store Details")
    st.dataframe(
        store_df.sort_values('Total_Sold', ascending=False).head(20).style.format({
            'Avg_Price': 'Rp {:,.0f}',
            'Total_Sold': '{:,.0f}',
            'Avg_Rating': '{:.2f}'
        }).background_gradient(cmap='Blues', subset=['Total_Sold']),
        use_container_width=True
    )

elif page == "🔎 Product Similarity":
    st.title("🔎 Product Similarity Finder")
    st.markdown("### Find similar products using Cosine Similarity")
    
    if vectorizer is None or tfidf_matrix is None:
        st.warning("⚠️ Similarity feature unavailable.")
        st.info("""
        **To enable product similarity:**
        1. Dataset must have 'cleaned_name' column
        2. Upload CSV file that has gone through complete text preprocessing
        
        **Alternative:** Model will be created automatically if 'cleaned_name' column is available.
        """)
    else:
        search_term = st.text_input("🔍 Search product:", placeholder="Example: nike air max, adidas samba, puma suede")
        
        if search_term:
            matching_products = df[df['Name'].str.contains(search_term, case=False, na=False)]
            
            if len(matching_products) > 0:
                selected_product = st.selectbox(
                    "Select product:",
                    options=matching_products.index,
                    format_func=lambda x: f"{df.loc[x, 'Name'][:60]} - Rp {df.loc[x, 'Price']:,.0f}"
                )
                
                if st.button("🔍 Find Similar Products"):
                    target_iloc = df.index.get_loc(selected_product)
                    similar_products = find_similar_products(df, tfidf_matrix, target_iloc, top_n=10)
                    
                    st.markdown("### 🎯 Target Product")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Price", f"Rp {df.loc[selected_product, 'Price']:,.0f}")
                    with col2:
                        st.metric("Sold", f"{df.loc[selected_product, 'Sold']:,.0f}")
                    with col3:
                        st.metric("Rating", f"{df.loc[selected_product, 'Rating']:.2f} ⭐")
                    with col4:
                        st.metric("Brand", df.loc[selected_product, 'Brand'])
                    
                    st.markdown("---")
                    
                    st.markdown("### 🔗 Similar Products (Top 10)")
                    
                    colors = px.colors.qualitative.Set3[:len(similar_products['Brand'].unique())]
                    fig = px.bar(
                        similar_products,
                        x='Similarity_Score',
                        y='Name',
                        color='Brand',
                        orientation='h',
                        color_discrete_sequence=colors,
                        hover_data=['Price', 'Sold', 'Rating']
                    )
                    fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    display_cols = ['Name', 'Brand', 'Price', 'Sold', 'Rating', 'Similarity_Score']
                    st.dataframe(
                        similar_products[display_cols].style.format({
                            'Price': 'Rp {:,.0f}',
                            'Sold': '{:,.0f}',
                            'Rating': '{:.2f}',
                            'Similarity_Score': '{:.4f}'
                        }).background_gradient(cmap='Greens', subset=['Similarity_Score']),
                        use_container_width=True
                    )
            else:
                st.warning("Product not found. Try another keyword!")

elif page == "📈 Price Analysis":
    st.title("📈 Price Analysis")
    st.markdown("### Price analysis and pricing strategy")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Min Price", f"Rp {df_filtered['Price'].min():,.0f}")
    with col2:
        st.metric("Median Price", f"Rp {df_filtered['Price'].median():,.0f}")
    with col3:
        st.metric("Mean Price", f"Rp {df_filtered['Price'].mean():,.0f}")
    with col4:
        st.metric("Max Price", f"Rp {df_filtered['Price'].max():,.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        colors = px.colors.qualitative.Set3[:len(df_filtered['Brand'].unique())]
        fig = px.histogram(
            df_filtered,
            x='Price',
            nbins=50,
            color='Brand',
            color_discrete_sequence=colors,
            opacity=0.7
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Price vs Rating")
        fig = px.scatter(
            df_filtered,
            x='Price',
            y='Rating',
            color='Brand',
            size='Sold',
            color_discrete_sequence=colors,
            opacity=0.6
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Price Comparison by Brand")
    brand_price = df_filtered.groupby('Brand')['Price'].describe()
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(
            brand_price.style.format("{:,.0f}").background_gradient(cmap='Blues'),
            use_container_width=True
        )
    
    with col2:
        fig = px.violin(
            df_filtered,
            x='Brand',
            y='Price',
            color='Brand',
            color_discrete_sequence=colors,
            box=True
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Price Segments")
    price_bins = [0, 200000, 500000, 1000000, 2000000, df_filtered['Price'].max()]
    price_labels = ['Budget (<200k)', 'Low (200k-500k)', 'Mid (500k-1M)', 'High (1M-2M)', 'Premium (>2M)']
    df_filtered['Price_Segment'] = pd.cut(df_filtered['Price'], bins=price_bins, labels=price_labels)
    
    segment_dist = df_filtered['Price_Segment'].value_counts().sort_index()
    fig = px.bar(
        x=segment_dist.index,
        y=segment_dist.values,
        color=segment_dist.values,
        color_continuous_scale='Viridis',
        labels={'x': 'Price Segment', 'y': 'Product Count'}
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

elif page == "⚔️ Competitor Analysis":
    st.title("⚔️ Competitor Analysis")
    st.markdown("### Multi-brand competitive intelligence")
    
    st.subheader("Brand Performance Comparison")
    brand_stats = df_filtered.groupby('Brand').agg({
        'Price': ['mean', 'median', 'min', 'max'],
        'Sold': ['sum', 'mean'],
        'Rating': 'mean',
        'Name': 'count'
    }).round(2)
    
    brand_stats.columns = ['Avg Price', 'Median Price', 'Min Price', 'Max Price', 
                            'Total Sales', 'Avg Sales per Product', 'Avg Rating', 'Product Count']
    
    st.dataframe(
        brand_stats.style.format({
            'Avg Price': 'Rp {:,.0f}',
            'Median Price': 'Rp {:,.0f}',
            'Min Price': 'Rp {:,.0f}',
            'Max Price': 'Rp {:,.0f}',
            'Total Sales': '{:,.0f}',
            'Avg Sales per Product': '{:.2f}',
            'Avg Rating': '{:.2f}',
            'Product Count': '{:,.0f}'
        }).background_gradient(cmap='RdYlGn', subset=['Avg Rating', 'Total Sales']),
        use_container_width=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Market Share by Sales Volume")
        brand_sales = df_filtered.groupby('Brand')['Sold'].sum().sort_values(ascending=False)
        fig = px.pie(
            values=brand_sales.values,
            names=brand_sales.index,
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Market Share by Product Count")
        brand_products = df_filtered['Brand'].value_counts()
        fig = px.pie(
            values=brand_products.values,
            names=brand_products.index,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Price Strategy Comparison")
    fig = px.box(
        df_filtered,
        x='Brand',
        y='Price',
        color='Brand',
        color_discrete_sequence=px.colors.qualitative.Set3,
        points="outliers"
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Sales Performance by Brand")
    fig = px.box(
        df_filtered,
        x='Brand',
        y='Sold',
        color='Brand',
        color_discrete_sequence=px.colors.qualitative.Set3,
        points="outliers"
    )
    fig.update_layout(showlegend=False, height=400, yaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)
    
    # Top performers per brand
    st.subheader("Top Performers by Brand")
    n_products = st.slider("Number of products to show per brand:", 3, 10, 5)
    
    for brand in df_filtered['Brand'].unique():
        with st.expander(f"🏆 Top {n_products} Best Sellers - {brand}"):
            brand_top = df_filtered[df_filtered['Brand'] == brand].nlargest(n_products, 'Sold')
            st.dataframe(
                brand_top[['Name', 'Price', 'Sold', 'Rating', 'Store']].style.format({
                    'Price': 'Rp {:,.0f}',
                    'Sold': '{:,.0f}',
                    'Rating': '{:.2f}'
                }),
                use_container_width=True
            )

st.markdown("---")
st.markdown(f"""
    <div style='text-align: center'>
        <p>E-commerce Market Intelligence Dashboard | Built with Streamlit</p>
        <p style='font-size: 0.8em; color: gray;'>Data Source: {source_type.upper()} | Records: {len(df_filtered):,} | Brands: {len(df_filtered['Brand'].unique())}</p>
    </div>
    """, unsafe_allow_html=True)