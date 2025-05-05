import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go

########################################
# 1. DATA LOADING & PREPROCESSING
########################################
def load_data():
    # Load the CSV files (assumes UTF-8 encoding)
    asset_df = pd.read_csv("FAR-Trans-Data/asset_information.csv")
    customer_df = pd.read_csv("FAR-Trans-Data/customer_information.csv")
    transactions_df = pd.read_csv("FAR-Trans-Data/transactions.csv")
    limit_prices_df = pd.read_csv("FAR-Trans-Data/limit_prices.csv")
    
    # Fix any column naming issues
    if 'ustomerID' in transactions_df.columns:
        transactions_df.rename(columns={'ustomerID': 'customerID'}, inplace=True)
    
    return asset_df, customer_df, transactions_df, limit_prices_df

def preprocess_data(asset_df, customer_df, transactions_df):
    # Work only with "Buy" transactions (as a positive signal)
    trans_buy = transactions_df[transactions_df['transactionType'] == "Buy"].copy()
    
    # Aggregate transactions: you can use count or sum of totalValue if available.
    # Here we use count as a proxy for interest.
    rating_df = trans_buy.groupby(['customerID', 'ISIN']).size().reset_index(name="count")
    
    # Build the user-item rating matrix
    rating_matrix = rating_df.pivot(index='customerID', columns='ISIN', values='count').fillna(0)
    
    return rating_matrix, rating_df, trans_buy

########################################
# 2. COLLABORATIVE FILTERING COMPONENT
########################################
def matrix_factorization(rating_matrix, n_components=5):
    # Perform low-rank approximation with TruncatedSVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(rating_matrix)
    V = svd.components_.T  # shape: (num_assets, n_components)
    
    pred_ratings = np.dot(U, V.T)
    pred_df = pd.DataFrame(pred_ratings, index=rating_matrix.index, columns=rating_matrix.columns)
    
    # Also return the explained variance for visualization
    explained_variance = svd.explained_variance_ratio_
    
    return pred_df, explained_variance

########################################
# 3. CONTENT-BASED FILTERING COMPONENT
########################################
def content_based_scores(customer_id, rating_df, asset_df, limit_prices_df):
    """
    Create asset feature vectors from:
      - assetCategory and assetSubCategory (one-hot encoded)
      - Profitability from limit_prices (normalized)
    For a given customer, construct a profile by averaging the features of assets they bought.
    Then compute cosine similarity between that profile and each asset feature.
    """
    # Merge asset info with profitability info from limit_prices
    asset_features = asset_df[['ISIN', 'assetCategory', 'assetSubCategory']].copy()
    
    # Merge profitability – note: not all assets may have limit price records.
    asset_features = asset_features.merge(limit_prices_df[['ISIN', 'profitability']], on='ISIN', how='left')
    asset_features['profitability'].fillna(asset_features['profitability'].median(), inplace=True)
    
    # One-hot encode assetCategory and assetSubCategory
    cat_dummies = pd.get_dummies(asset_features['assetCategory'], prefix="cat")
    sub_dummies = pd.get_dummies(asset_features['assetSubCategory'], prefix="sub")
    
    asset_feat = pd.concat([asset_features[['ISIN', 'profitability']], cat_dummies, sub_dummies], axis=1)
    asset_feat.set_index("ISIN", inplace=True)
    
    # Normalize the profitability column to [0,1]
    asset_feat['profitability_norm'] = (asset_feat['profitability'] - asset_feat['profitability'].min()) / \
                                       (asset_feat['profitability'].max() - asset_feat['profitability'].min())
    asset_feat.drop(columns=['profitability'], inplace=True)
    
    # Build the customer profile: average features of assets they bought
    cust_assets = rating_df[rating_df['customerID'] == customer_id]['ISIN']
    
    if len(cust_assets) > 0 and any(cust_assets.isin(asset_feat.index)):
        cust_vector = asset_feat.loc[cust_assets].mean(axis=0).values.reshape(1, -1)
        sim_scores = cosine_similarity(cust_vector, asset_feat)[0]
        content_score = pd.Series(sim_scores, index=asset_feat.index)
    else:
        # No prior transaction => use neutral score (e.g., 0.5)
        content_score = pd.Series(0.5, index=asset_feat.index)
    return content_score

########################################
# 4. DEMOGRAPHIC-BASED COMPONENT
########################################
def demographic_score(customer_id, customer_df, asset_df, limit_prices_df):
    """
    A more advanced demographic matching:
      - Uses customer's riskLevel and investmentCapacity.
      - Incorporates a simplified risk-return tradeoff using asset profitability.
    The mapping below is illustrative: aggressive or premium customers might favor assets with higher
    profitability (though with higher volatility), whereas conservative customers may prefer lower profitability assets.
    """
    # Risk mapping: assign a target profitability range based on risk level.
    risk_map = {
        "Conservative": (0, 0.4),
        "Predicted_Conservative": (0, 0.4),
        "Income": (0.3, 0.6),
        "Balanced": (0.4, 0.7),
        "Aggressive": (0.6, 1.0),
        "Predicted_Income": (0.3, 0.6),
        "Predicted_Balanced": (0.4, 0.7),
        "Predicted_Aggressive": (0.6, 1.0)
    }
    
    cust_info = customer_df[customer_df['customerID'] == customer_id]
    if cust_info.empty:
        risk = "Balanced"
    else:
        risk = cust_info.iloc[-1]['riskLevel']  # assume most recent record
    
    # Default target profitability range
    target_range = risk_map.get(risk, (0.4, 0.7))
    
    # Merge asset info with profitability
    asset_demo = asset_df[['ISIN', 'assetCategory', 'assetSubCategory']].copy()
    asset_demo = asset_demo.merge(limit_prices_df[['ISIN', 'profitability']], on='ISIN', how='left')
    asset_demo['profitability'].fillna(asset_demo['profitability'].median(), inplace=True)
    
    # Score assets higher if their profitability is close to the center of target_range.
    target_center = (target_range[0] + target_range[1]) / 2
    def score_profit(prof):
        # A simple inverse distance score normalized to [0,1]
        return 1 - min(abs(prof - target_center) / (target_center), 1)
    
    asset_demo['demo_score'] = asset_demo['profitability'].apply(score_profit)
    demo_score = pd.Series(asset_demo['demo_score'].values, index=asset_demo['ISIN'])
    return demo_score, risk

########################################
# 5. HYBRID RECOMMENDATION COMBINING THE THREE COMPONENTS
########################################
def normalize_scores(s):
    if s.max() - s.min() > 0:
        return (s - s.min()) / (s.max() - s.min())
    else:
        return s

def hybrid_recommendation(customer_id, rating_matrix, pred_df, rating_df, asset_df, 
                          customer_df, limit_prices_df, weights, top_n=5):
    """
    Combines:
      - Collaborative filtering (CF) score from matrix factorization.
      - Content-based (CB) score from asset features (including profitability).
      - Demographic (DEMO) score based on customer's risk and asset profitability.
      
    'weights' is a tuple: (CF_weight, CB_weight, DEMO_weight)
    """
    # 1. Collaborative Filtering
    if customer_id in pred_df.index:
        cf_scores = pred_df.loc[customer_id]
    else:
        cf_scores = pd.Series(0, index=rating_matrix.columns)
    
    # 2. Content-based Scores
    content_scores = content_based_scores(customer_id, rating_df, asset_df, limit_prices_df)
    
    # 3. Demographic-based Scores
    demo_scores, risk_level = demographic_score(customer_id, customer_df, asset_df, limit_prices_df)
    
    # Normalize each score component to [0,1]
    cf_norm = normalize_scores(cf_scores)
    cb_norm = normalize_scores(content_scores)
    demo_norm = normalize_scores(demo_scores)
    
    # Weighted hybrid score
    final_score = weights[0]*cf_norm + weights[1]*cb_norm + weights[2]*demo_norm
    
    # Exclude assets that the customer has already bought
    bought_assets = rating_df[rating_df['customerID'] == customer_id]['ISIN'].unique() if not rating_df[rating_df['customerID'] == customer_id].empty else []
    final_score = final_score.drop(labels=bought_assets, errors='ignore')
    
    recommendations = final_score.sort_values(ascending=False).head(top_n)
    
    # Return component scores as well for visualization
    component_scores = {
        'CF': cf_norm,
        'Content': cb_norm,
        'Demographic': demo_norm,
        'Final': final_score
    }
    
    return recommendations, component_scores, risk_level

########################################
# 6. PERFORMANCE EVALUATION METRICS
########################################
def split_train_test(rating_df, test_size=0.2):
    """
    Split transactions data into training and testing sets
    by randomly hiding some transactions for each user.
    """
    # Group by customer and keep test_size% of transactions for testing
    train_data = []
    test_data = []
    
    for customer_id, group in rating_df.groupby('customerID'):
        if len(group) >= 5:  # Only evaluate customers with at least 5 transactions
            train, test = train_test_split(group, test_size=test_size, random_state=42)
            train_data.append(train)
            test_data.append(test)
    
    train_df = pd.concat(train_data) if train_data else pd.DataFrame()
    test_df = pd.concat(test_data) if test_data else pd.DataFrame()
    
    return train_df, test_df

def calculate_precision_recall_at_k(recommendations, ground_truth, k=5):
    """
    Calculate precision and recall at K for a given customer
    """
    rec_set = set(recommendations[:k].index)
    gt_set = set(ground_truth)
    
    if not gt_set:  # If no ground truth items, precision and recall are undefined
        return None, None
    
    if not rec_set:  # If no recommendations, precision is 0/0 (undefined) and recall is 0
        return None, 0
    
    n_relevant = len(rec_set.intersection(gt_set))
    precision = n_relevant / len(rec_set)
    recall = n_relevant / len(gt_set)
    
    return precision, recall

def mean_average_precision(recommendations, ground_truth, k=10):
    """
    Calculate Mean Average Precision (MAP) at K for a given customer
    """
    if len(ground_truth) == 0:  # Fixed: check length instead of truthy value
        return None
    
    relevant_items = set(ground_truth)
    ap = 0.0
    num_hits = 0
    
    for i, item in enumerate(recommendations[:k].index):
        if item in relevant_items:
            num_hits += 1
            ap += num_hits / (i + 1)
    
    if not num_hits:
        return 0.0
    
    return ap / min(len(relevant_items), k)
    """
    Calculate Mean Average Precision (MAP) at K for a given customer
    """
    if not ground_truth:
        return None
    
    relevant_items = set(ground_truth)
    ap = 0.0
    num_hits = 0
    
    for i, item in enumerate(recommendations[:k].index):
        if item in relevant_items:
            num_hits += 1
            ap += num_hits / (i + 1)
    
    if not num_hits:
        return 0.0
    
    return ap / min(len(relevant_items), k)

def evaluate_recommendations(customer_ids, train_df, test_df, rating_matrix, pred_df, 
                           asset_df, customer_df, limit_prices_df, weights, k=5):
    """
    Evaluate the recommendation system on multiple customers
    """
    results = {
        'customer_id': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'map': []
    }
    
    for customer_id in customer_ids:
        # Get ground truth items from test set
        ground_truth = test_df[test_df['customerID'] == customer_id]['ISIN'].unique()
        
        if len(ground_truth) == 0:
            continue
        
        # Generate recommendations based on training data
        recommendations, _, _ = hybrid_recommendation(
            customer_id, rating_matrix, pred_df, train_df, 
            asset_df, customer_df, limit_prices_df, weights, top_n=k
        )
        
        # Calculate metrics
        precision, recall = calculate_precision_recall_at_k(recommendations, ground_truth, k)
        
        if precision is not None and recall is not None:
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            f1 = None
        
        map_score = mean_average_precision(recommendations, ground_truth, k)
        
        results['customer_id'].append(customer_id)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)
        results['map'].append(map_score)
    
    return pd.DataFrame(results)

########################################
# 7. VISUALIZATION FUNCTIONS
########################################
def plot_component_weights(recommendations, component_scores, asset_df):
    """
    Create a stacked bar chart showing the contribution 
    of each recommendation component
    """
    # Prepare data for the top recommended items
    top_items = recommendations.index
    
    # Create dataframe with component scores
    comp_df = pd.DataFrame({
        'CF': component_scores['CF'].loc[top_items],
        'Content': component_scores['Content'].loc[top_items],
        'Demographic': component_scores['Demographic'].loc[top_items]
    })
    
    # Get asset names for better labels
    asset_names = asset_df.loc[asset_df['ISIN'].isin(top_items), ['ISIN', 'assetName']]
    asset_names = asset_names.set_index('ISIN')
    
    # Get short names for display
    short_names = {}
    for isin in top_items:
        if isin in asset_names.index:
            name = asset_names.loc[isin, 'assetName']
            short_names[isin] = name[:20] + '...' if len(name) > 20 else name
        else:
            short_names[isin] = isin
    
    comp_df.index = [short_names.get(isin, isin) for isin in top_items]
    
    # Create stacked bar chart with Plotly
    comp_df_melt = comp_df.reset_index().melt(id_vars='index', var_name='Component', value_name='Score')
    
    fig = px.bar(comp_df_melt, x='Score', y='index', color='Component', 
                 orientation='h', height=400, 
                 title='Component Contribution to Top Recommendations')
    
    fig.update_layout(yaxis_title='Asset', xaxis_title='Component Score')
    
    return fig

def plot_asset_category_distribution(recommendations, asset_df):
    """
    Create a pie chart showing the distribution of asset categories 
    in the recommendations
    """
    rec_assets = asset_df[asset_df['ISIN'].isin(recommendations.index)]
    category_counts = rec_assets['assetCategory'].value_counts()
    
    fig = px.pie(
        names=category_counts.index,
        values=category_counts.values,
        title='Asset Category Distribution in Recommendations',
        hole=0.4
    )
    
    return fig

def plot_recommendation_scores(recommendations):
    """
    Create a horizontal bar chart of recommendation scores
    """
    fig = px.bar(
        x=recommendations.values,
        y=recommendations.index,
        orientation='h',
        title='Recommendation Scores',
        labels={'x': 'Score', 'y': 'Asset ID'}
    )
    
    return fig

def plot_explained_variance(explained_variance):
    """
    Plot the explained variance ratio from SVD components
    """
    cum_explained_var = np.cumsum(explained_variance)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(explained_variance) + 1)),
        y=explained_variance,
        name='Individual'
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cum_explained_var) + 1)),
        y=cum_explained_var,
        name='Cumulative',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Explained Variance by SVD Components',
        xaxis_title='Component',
        yaxis_title='Explained Variance Ratio',
        legend_title='Variance Type'
    )
    
    return fig

def plot_metrics_comparison(metrics_df):
    """
    Plot a comparison of different recommendation metrics
    """
    # Calculate mean metrics
    mean_metrics = {
        'Precision': metrics_df['precision'].mean(),
        'Recall': metrics_df['recall'].mean(),
        'F1 Score': metrics_df['f1'].mean(),
        'MAP': metrics_df['map'].mean()
    }
    
    # Create bar chart
    fig = px.bar(
        x=list(mean_metrics.keys()),
        y=list(mean_metrics.values()),
        title='Mean Recommendation Performance Metrics',
        labels={'x': 'Metric', 'y': 'Value'}
    )
    
    return fig

########################################
# 8. STREAMLIT FRONTEND
########################################
def main():
    st.set_page_config(layout="wide", page_title="Asset Selection Strategy")
    
    st.title("Asset Selection Strategy Asset Recommender")
    st.write("An enhanced hybrid recommendation system leveraging the Asset Selection Strategy dataset, combining collaborative filtering, content-based filtering, demographic matching with advanced visualizations and performance metrics.")
    
    # Load all required data
    with st.spinner("Loading datasets..."):
        asset_df, customer_df, transactions_df, limit_prices_df = load_data()
    
    # Preprocess transactions data
    with st.spinner("Preprocessing transactions..."):
        rating_matrix, rating_df, trans_buy = preprocess_data(asset_df, customer_df, transactions_df)
    
    # Create tabs for different app sections
    tab1, tab2, tab3 = st.tabs(["Recommendations", "Performance Evaluation", "Dataset Insights"])
    
    with tab1:
        # Sidebar: parameters for recommendations
        st.sidebar.header("Recommendation Settings")
        
        # Customer selection: list all customers (from rating matrix) and also allow manual entry
        customer_list = list(set(rating_matrix.index.tolist()) | set(customer_df['customerID'].unique()))
        customer_id_input = st.sidebar.selectbox("Select Customer ID", sorted(customer_list))
        
        top_n = st.sidebar.number_input("Top N Recommendations", min_value=1, value=5)
        
        st.sidebar.subheader("Component Weights")
        cf_weight = st.sidebar.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.5, 0.1)
        cb_weight = st.sidebar.slider("Content-Based Weight", 0.0, 1.0, 0.3, 0.1)
        demo_weight = st.sidebar.slider("Demographic Weight", 0.0, 1.0, 0.2, 0.1)
        
        # Normalize weights to sum to 1
        total_weight = cf_weight + cb_weight + demo_weight
        if total_weight > 0:
            norm_weights = (cf_weight/total_weight, cb_weight/total_weight, demo_weight/total_weight)
        else:
            norm_weights = (0.33, 0.33, 0.34)  # Default equal weights
            
        st.sidebar.info(f"Normalized weights: CF={norm_weights[0]:.2f}, CB={norm_weights[1]:.2f}, Demo={norm_weights[2]:.2f}")
        
        # SVD components for collaborative filtering
        n_components = st.sidebar.slider("SVD Components", 2, 20, 5)
        
        # Run collaborative filtering when needed
        if 'svd_run' not in st.session_state or st.session_state.n_components != n_components:
            with st.spinner("Performing matrix factorization..."):
                st.session_state.pred_df, st.session_state.explained_variance = matrix_factorization(
                    rating_matrix, n_components=n_components)
                st.session_state.svd_run = True
                st.session_state.n_components = n_components
        
        # Button to generate recommendations
        if st.button("Generate Recommendations"):
            with st.spinner("Generating recommendations..."):
                # Use the stored prediction matrix
                recs, component_scores, risk_level = hybrid_recommendation(
                    customer_id_input, rating_matrix, st.session_state.pred_df, 
                    rating_df, asset_df, customer_df, limit_prices_df, 
                    norm_weights, top_n=int(top_n)
                )
                
                # Create two columns layout for recommendations
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.write(f"### Top {top_n} Recommendations for Customer: **{customer_id_input}**")
                    st.write(f"Customer Risk Level: **{risk_level}**")
                    
                    # Show recommendations with scores
                    st.write(recs.to_frame("Score"))
                    
                    # Show asset details for recommended assets
                    st.write("### Asset Details")
                    rec_asset_info = asset_df[asset_df['ISIN'].isin(recs.index)]
                    st.write(rec_asset_info)
                
                with col2:
                    # Plot component contribution
                    st.plotly_chart(
                        plot_component_weights(recs, component_scores, asset_df),
                        use_container_width=True
                    )
                    
                    # Plot category distribution
                    st.plotly_chart(
                        plot_asset_category_distribution(recs, asset_df),
                        use_container_width=True
                    )
                
                # Plot recommendation scores
                st.plotly_chart(
                    plot_recommendation_scores(recs),
                    use_container_width=True
                )
                
                # Plot explained variance from SVD
                st.plotly_chart(
                    plot_explained_variance(st.session_state.explained_variance),
                    use_container_width=True
                )
    
    with tab2:
        st.header("Recommendation System Evaluation")
        st.write("Evaluate the performance of the recommendation system using various metrics.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
            eval_k = st.slider("K Value for Evaluation", 1, 20, 5)
            
        with col2:
            eval_cf_weight = st.slider("Eval CF Weight", 0.0, 1.0, 0.5, 0.1, key="eval_cf")
            eval_cb_weight = st.slider("Eval Content Weight", 0.0, 1.0, 0.3, 0.1, key="eval_cb")
            eval_demo_weight = st.slider("Eval Demo Weight", 0.0, 1.0, 0.2, 0.1, key="eval_demo")
            
            # Normalize weights
            eval_total = eval_cf_weight + eval_cb_weight + eval_demo_weight
            if eval_total > 0:
                eval_weights = (eval_cf_weight/eval_total, eval_cb_weight/eval_total, eval_demo_weight/eval_total)
            else:
                eval_weights = (0.33, 0.33, 0.34)
        
        if st.button("Run Evaluation"):
            with st.spinner("Splitting data into train and test sets..."):
                train_df, test_df = split_train_test(rating_df, test_size=test_size)
                
                # Create train rating matrix
                train_matrix = train_df.pivot(index='customerID', columns='ISIN', values='count').fillna(0)
                
                # Run SVD on training data
                with st.spinner("Running matrix factorization on training data..."):
                    train_pred_df, _ = matrix_factorization(train_matrix, n_components=n_components)
                
                # Get customers with enough data for evaluation
                eval_customers = test_df['customerID'].value_counts()
                eval_customers = eval_customers[eval_customers >= 3].index.tolist()
                
                # Limit to first 50 customers for performance
                eval_customers = eval_customers[:50]
                
                # Evaluate recommendations
                with st.spinner(f"Evaluating on {len(eval_customers)} customers..."):
                    metrics_df = evaluate_recommendations(
                        eval_customers, train_df, test_df, train_matrix, 
                        train_pred_df, asset_df, customer_df, limit_prices_df, 
                        eval_weights, k=eval_k
                    )
                
                # Display metrics table
                st.write("### Performance Metrics")
                st.write(f"Number of customers evaluated: {len(metrics_df)}")
                
                # Calculate mean metrics
                mean_metrics = metrics_df[['precision', 'recall', 'f1', 'map']].mean().to_frame('Mean Value').T
                st.write("### Mean Performance")
                st.write(mean_metrics)
                
                # Plot metrics comparison
                st.plotly_chart(
                    plot_metrics_comparison(metrics_df),
                    use_container_width=True
                )
                
                # Show detailed metrics
                st.write("### Detailed Metrics by Customer")
                st.dataframe(metrics_df)
                
    with tab3:
        st.header("Dataset Insights")
        st.write("Explore the dataset and gain insights into customer behavior and asset characteristics.")
        
        insight_tabs = st.tabs(["Transaction Overview", "Customer Analysis", "Asset Analysis"])
        
        with insight_tabs[0]:
            st.subheader("Transaction Patterns")
            
            # Transaction count over time
            if 'transactionDate' in trans_buy.columns:
                trans_buy['transactionDate'] = pd.to_datetime(trans_buy['transactionDate'])
                trans_by_date = trans_buy.groupby(trans_buy['transactionDate'].dt.to_period("M")).size().reset_index()
                trans_by_date.columns = ['Month', 'Transaction Count']
                trans_by_date['Month'] = trans_by_date['Month'].astype(str)
                
                fig = px.line(
                    trans_by_date, x='Month', y='Transaction Count',
                    title='Transaction Volume Over Time',
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Most popular assets
            popular_assets = trans_buy['ISIN'].value_counts().head(10).reset_index()
            popular_assets.columns = ['ISIN', 'Transaction Count']
            
            # Join with asset names if available
            if 'assetName' in asset_df.columns:
                popular_assets = popular_assets.merge(
                    asset_df[['ISIN', 'assetName', 'assetCategory']],
                    on='ISIN',
                    how='left'
                )
                
                fig = px.bar(
                    popular_assets, 
                    x='Transaction Count', 
                    y='assetName' if 'assetName' in popular_assets.columns else 'ISIN',
                    color='assetCategory' if 'assetCategory' in popular_assets.columns else None,
                    title='Top 10 Most Popular Assets',
                    orientation='h'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(
                    popular_assets, 
                    x='Transaction Count', 
                    y='ISIN',
                    title='Top 10 Most Popular Assets',
                    orientation='h'
                )
                st.plotly_chart(fig, use_container_width=True)
                
        with insight_tabs[1]:
            st.subheader("Customer Analysis")
            
            # Customer distribution by risk level
            if 'riskLevel' in customer_df.columns:
                risk_dist = customer_df['riskLevel'].value_counts().reset_index()
                risk_dist.columns = ['Risk Level', 'Count']
                
                fig = px.bar(
                    risk_dist,
                    x='Risk Level',
                    y='Count',
                    title='Customer Distribution by Risk Level',
                    color='Risk Level'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Customer activity analysis
            customer_activity = trans_buy.groupby('customerID').size().reset_index(name='Transaction Count')
            
            fig = px.histogram(
                customer_activity,
                x='Transaction Count',
                title='Customer Activity Distribution',
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with insight_tabs[2]:
            st.subheader("Asset Analysis")
            
            # Asset distribution by category
            cat_dist = asset_df['assetCategory'].value_counts().reset_index()
            cat_dist.columns = ['Asset Category', 'Count']
            
            fig = px.pie(
                cat_dist,
                values='Count',
                names='Asset Category',
                title='Asset Distribution by Category'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Asset profitability analysis if available
            if 'profitability' in limit_prices_df.columns:
                # Merge asset info with profitability
                asset_profit = asset_df.merge(
                    limit_prices_df[['ISIN', 'profitability']],
                    on='ISIN',
                    how='inner'
                )
                
                # Box plot of profitability by asset category
                fig = px.box(
                    asset_profit,
                    x='assetCategory',
                    y='profitability',
                    title='Profitability Distribution by Asset Category',
                    color='assetCategory'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot of profitability vs popularity
                asset_popularity = trans_buy['ISIN'].value_counts().reset_index()
                asset_popularity.columns = ['ISIN', 'Popularity']
                
                asset_profit_pop = asset_profit.merge(
                    asset_popularity,
                    on='ISIN',
                    how='left'
                )
                asset_profit_pop['Popularity'].fillna(0, inplace=True)
                
                fig = px.scatter(
                    asset_profit_pop,
                    x='profitability',
                    y='Popularity',
                    color='assetCategory',
                    title='Asset Profitability vs Popularity',
                    hover_data=['ISIN', 'assetName'] if 'assetName' in asset_profit_pop.columns else ['ISIN']
                )
                fig.update_layout(xaxis_title='Profitability', yaxis_title='Number of Transactions')
                st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()