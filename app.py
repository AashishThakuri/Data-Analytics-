import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import yfinance as yf
from prophet import Prophet
import requests
from PIL import Image
import io
import os
import pathlib

# Import custom modules
from modules.numeric_analysis import NumericAnalysis
from modules.text_analysis import TextAnalysis
from modules.image_analysis import ImageAnalysis
from modules.realtime_analysis import RealTimeAnalysis

# Set page config
st.set_page_config(
    page_title="Advanced Data Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.title("📊 Advanced Analytics Platform")
    
    # Plot customization options in sidebar
    st.sidebar.header("Plot Customization")
    plot_config = {
        'theme': st.sidebar.selectbox(
            "Color Theme",
            ["plotly", "plotly_white", "plotly_dark", "seaborn", "simple_white"],
            index=1
        ),
        'plot_type': st.sidebar.selectbox(
            "Default Plot Type",
            ["line", "bar", "scatter", "area", "box"],
            index=0
        ),
        'color_scheme': st.sidebar.selectbox(
            "Color Scheme",
            ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Set1", "Set2", "Paired"],
            index=0
        )
    }
    
    # Analysis type selection
    st.sidebar.header("Select Analysis Type")
    analysis_type = st.sidebar.selectbox(
        "",
        ["Numeric Analysis", "Text Analysis", "Image Analysis", "Real-time Analysis"]
    )
    
    if analysis_type == "Numeric Analysis":
        numeric_analysis_page(plot_config)
    elif analysis_type == "Text Analysis":
        text_analysis_page(plot_config)
    elif analysis_type == "Image Analysis":
        image_analysis_page(plot_config)
    else:
        realtime_analysis_page(plot_config)

def numeric_analysis_page(plot_config):
    st.title("📊 Numeric Analysis")
    
    # Additional plot options specific to numeric analysis
    col1, col2 = st.columns(2)
    with col1:
        show_grid = st.checkbox("Show Grid", value=True)
    with col2:
        show_legend = st.checkbox("Show Legend", value=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Dataset Overview
            st.header("📊 Dataset Overview")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            
            if len(numeric_cols) > 0:
                # Create plot based on selected plot type
                if plot_config['plot_type'] in ['scatter', 'line', 'area']:
                    plot_func = getattr(px, plot_config['plot_type'])
                    for col in numeric_cols:
                        fig = plot_func(
                            df,
                            y=col,
                            title=f"{col} Distribution",
                            template=plot_config['theme'],
                            color_discrete_sequence=getattr(px.colors.sequential, plot_config['color_scheme'])
                        )
                        fig.update_layout(
                            showlegend=show_legend,
                            xaxis=dict(showgrid=show_grid),
                            yaxis=dict(showgrid=show_grid),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                elif plot_config['plot_type'] == 'box':
                    fig = px.box(
                        df[numeric_cols],
                        title="Numeric Attributes Distribution",
                        template=plot_config['theme'],
                        color_discrete_sequence=getattr(px.colors.sequential, plot_config['color_scheme'])
                    )
                    fig.update_layout(
                        showlegend=show_legend,
                        xaxis=dict(showgrid=show_grid),
                        yaxis=dict(showgrid=show_grid),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:  # bar or other types
                    for col in numeric_cols:
                        fig = px.histogram(
                            df,
                            x=col,
                            title=f"{col} Distribution",
                            template=plot_config['theme'],
                            color_discrete_sequence=getattr(px.colors.sequential, plot_config['color_scheme'])
                        )
                        fig.update_layout(
                            showlegend=show_legend,
                            xaxis=dict(showgrid=show_grid),
                            yaxis=dict(showgrid=show_grid),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Check if this might be a prediction dataset
            if len(df.columns) > 1 and (df.columns[-1].lower().find('target') >= 0 or 
                                      df.columns[-1].lower().find('prediction') >= 0 or
                                      df.columns[-1].lower().find('class') >= 0):
                st.header("🎯 Prediction Dataset Analysis")
                analyzer = NumericAnalysis(df)  
                analysis = analyzer.analyze_prediction_dataset(plot_config=plot_config, 
                                                            show_grid=show_grid,
                                                            show_legend=show_legend)
                
                if analysis:
                    # Dataset Overview
                    st.subheader("Dataset Overview")
                    overview = analysis['overview']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", overview['total_samples'])
                    with col2:
                        st.metric("Features", overview['features'])
                    with col3:
                        st.metric("Target Classes", overview['target_classes'])
                    
                    # Class Distribution
                    st.subheader("Class Distribution")
                    st.plotly_chart(overview['class_distribution']['figure'], use_container_width=True)
                    st.write("This chart shows the distribution of target classes in your dataset.")
                    
                    # Feature Correlations
                    st.subheader("Feature Correlations")
                    st.plotly_chart(analysis['correlations']['heatmap'], use_container_width=True)
                    st.write("The heatmap shows how different features are correlated with each other.")
                    
                    # Feature Importance
                    st.subheader("Feature Importance")
                    st.plotly_chart(analysis['feature_importance']['figure'], use_container_width=True)
                    st.write("This chart shows which features have the strongest relationship with the target variable.")
                    
                    # Feature Distributions
                    st.subheader("Top Feature Distributions")
                    for feature, plot_data in analysis['feature_distributions'].items():
                        st.write(f"### {feature}")
                        st.plotly_chart(plot_data['figure'], use_container_width=True)
                        stats = plot_data['stats']
                        st.write(f"- Mean: {stats['mean']:.2f}")
                        st.write(f"- Median: {stats['median']:.2f}")
                        st.write(f"- Standard Deviation: {stats['std']:.2f}")
            
            else:
                # Initialize analyzer
                analyzer = NumericAnalysis(df)
                
                # Sidebar options
                analysis_option = st.sidebar.selectbox(
                    "Choose Analysis",
                    ["Overview", "Statistical Analysis", "Distribution Analysis", 
                     "Correlation Analysis", "Outlier Detection", "Feature Engineering",
                     "Dimensionality Reduction", "Time Series Analysis"]
                )
                
                if analysis_option == "Overview":
                    st.subheader("Dataset Overview")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Rows", df.shape[0])
                    col2.metric("Columns", df.shape[1])
                    col3.metric("Missing Values", df.isna().sum().sum())
                    
                    st.dataframe(df.head())
                    st.write("Data Types:", df.dtypes)
                    
                elif analysis_option == "Statistical Analysis":
                    st.subheader("Statistical Analysis")
                    column = st.selectbox("Select Column", df.select_dtypes(include=[np.number]).columns)
                    stats = analyzer.basic_statistics(column)
                    st.dataframe(stats)
                    
                elif analysis_option == "Distribution Analysis":
                    st.subheader("Distribution Analysis")
                    column = st.selectbox("Select Column", df.select_dtypes(include=[np.number]).columns)
                    hist_fig, box_fig, qq_fig = analyzer.distribution_analysis(column)
                    st.plotly_chart(hist_fig)
                    st.plotly_chart(box_fig)
                    st.plotly_chart(qq_fig)
                    
                elif analysis_option == "Correlation Analysis":
                    st.subheader("Correlation Analysis")
                    corr_figs = analyzer.correlation_analysis()
                    for method, fig in corr_figs.items():
                        st.subheader(f"{method} Correlation")
                        st.plotly_chart(fig)
                    
                elif analysis_option == "Outlier Detection":
                    st.subheader("Outlier Detection")
                    column = st.selectbox("Select Column", df.select_dtypes(include=[np.number]).columns)
                    method = st.selectbox("Detection Method", ["zscore", "iqr", "isolation_forest"])
                    outliers, fig = analyzer.outlier_detection(column, method)
                    st.plotly_chart(fig)
                    st.write(f"Number of outliers detected: {len(outliers)}")
                    
                elif analysis_option == "Feature Engineering":
                    st.subheader("Feature Engineering")
                    columns = st.multiselect("Select Columns", df.select_dtypes(include=[np.number]).columns)
                    if columns:
                        poly_df, log_df, scaled_df = analyzer.feature_engineering(columns)
                        st.write("Polynomial Features:")
                        st.dataframe(poly_df.head())
                        st.write("Log Transformed Features:")
                        st.dataframe(log_df.head())
                        st.write("Scaled Features:")
                        st.dataframe(scaled_df.head())
                    
                elif analysis_option == "Dimensionality Reduction":
                    st.subheader("Dimensionality Reduction")
                    columns = st.multiselect("Select Columns", df.select_dtypes(include=[np.number]).columns)
                    method = st.selectbox("Method", ["pca", "tsne", "umap"])
                    if columns:
                        if method == "pca":
                            reduced_df, fig, explained_var = analyzer.dimensionality_reduction(columns, method)
                            st.plotly_chart(fig)
                            st.write("Explained Variance Ratio:", explained_var)
                        else:
                            reduced_df, fig = analyzer.dimensionality_reduction(columns, method)
                            st.plotly_chart(fig)
                    
                elif analysis_option == "Time Series Analysis":
                    st.subheader("Time Series Analysis")
                    date_column = st.selectbox("Select Date Column", df.columns)
                    value_column = st.selectbox("Select Value Column", df.select_dtypes(include=[np.number]).columns)
                    
                    if st.button("Analyze Time Series"):
                        with st.spinner("Analyzing time series data..."):
                            decomp_fig, ma_fig, arima_fig = analyzer.time_series_analysis(date_column, value_column)
                            
                            if decomp_fig and ma_fig and arima_fig:
                                st.subheader("Time Series Decomposition")
                                st.plotly_chart(decomp_fig, use_container_width=True)
                                st.write("""
                                This plot shows the decomposition of your time series into:
                                - Observed: The original data
                                - Trend: The long-term progression
                                - Seasonal: Repeating patterns
                                - Residual: The noise/random variation
                                """)
                                
                                st.subheader("Moving Averages")
                                st.plotly_chart(ma_fig, use_container_width=True)
                                st.write("""
                                Moving averages help smooth out short-term fluctuations:
                                - 7-day MA: Shows weekly trends
                                - 30-day MA: Shows monthly trends
                                """)
                                
                                st.subheader("ARIMA Forecast")
                                st.plotly_chart(arima_fig, use_container_width=True)
                                st.write("""
                                The ARIMA model provides a 30-day forecast based on historical patterns.
                                Blue line shows historical data, while the orange line shows the predicted values.
                                """)
                            else:
                                st.error("Could not perform time series analysis. Please check your data format.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def text_analysis_page(plot_config):
    st.title("📝 Text Analysis")
    
    # Initialize text analyzer
    text_analyzer = TextAnalysis()
    
    # Input method selection
    input_method = st.radio("Choose input method", ["Direct Input", "File Upload", "Real-time Data"])
    
    text_input = None
    
    if input_method == "Direct Input":
        text_input = st.text_area("Enter text to analyze", height=200)
        
    elif input_method == "File Upload":
        uploaded_file = st.file_uploader("Upload text file", type=['txt', 'csv', 'pdf', 'docx'])
        if uploaded_file:
            try:
                # Read file content based on file type
                file_type = uploaded_file.name.split('.')[-1].lower()
                if file_type == 'txt':
                    text_input = uploaded_file.getvalue().decode('utf-8')
                elif file_type == 'csv':
                    df = pd.read_csv(uploaded_file)
                    text_input = ' '.join(df.astype(str).values.flatten())
                elif file_type == 'pdf':
                    st.error("PDF support coming soon!")
                elif file_type == 'docx':
                    st.error("DOCX support coming soon!")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                
    elif input_method == "Real-time Data":
        source = st.selectbox("Select data source", ["Twitter", "News API"])
        if source == "Twitter":
            keyword = st.text_input("Enter keyword to track")
            if keyword:
                text_input = text_analyzer.get_twitter_data(keyword)
        else:
            keyword = st.text_input("Enter news keyword")
            if keyword:
                text_input = text_analyzer.get_news_data(keyword)
    
    if text_input:
        # Analysis type selection
        analysis_type = st.multiselect(
            "Select analyses to perform",
            ["Basic Statistics", "Sentiment Analysis", "Topic Modeling", "Word Cloud", "Word Frequencies"]
        )
        
        if analysis_type:
            try:
                if "Basic Statistics" in analysis_type:
                    st.subheader("📊 Basic Statistics")
                    stats = text_analyzer.basic_stats(text_input)
                    if isinstance(stats, pd.DataFrame):
                        st.dataframe(stats)
                    else:
                        st.write(stats)
                
                if "Sentiment Analysis" in analysis_type:
                    st.subheader("😊 Sentiment Analysis")
                    sentiment_results = text_analyzer.sentiment_analysis(text_input)
                    
                    if isinstance(sentiment_results, dict):
                        # Display overall sentiment
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Polarity", f"{sentiment_results.get('polarity', 0):.2f}")
                        with col2:
                            st.metric("Subjectivity", f"{sentiment_results.get('subjectivity', 0):.2f}")
                        
                        # Display detailed results if available
                        if 'detailed_results' in sentiment_results:
                            st.write("Detailed Analysis:")
                            st.dataframe(sentiment_results['detailed_results'])
                    else:
                        st.write(sentiment_results)
                
                if "Topic Modeling" in analysis_type:
                    st.subheader("📑 Topic Modeling")
                    topics = text_analyzer.topic_modeling([text_input])
                    if isinstance(topics, pd.DataFrame):
                        st.dataframe(topics)
                    else:
                        st.write(topics)
                
                if "Word Cloud" in analysis_type:
                    st.subheader("☁️ Word Cloud")
                    wordcloud = text_analyzer.generate_wordcloud(text_input)
                    if wordcloud is not None:
                        st.image(wordcloud.to_array())
                
                if "Word Frequencies" in analysis_type:
                    st.subheader("📈 Word Frequencies")
                    freq_fig = text_analyzer.plot_word_frequencies(text_input)
                    if freq_fig is not None:
                        st.plotly_chart(freq_fig)
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                st.error("Please try with a different text or analysis option")

def image_analysis_page(plot_config):
    st.title("Image Analysis")
    
    # Initialize image analyzer
    image_analyzer = ImageAnalysis()
    
    # Mode selection
    analysis_mode = st.radio(
        "Choose Analysis Mode",
        ["Single Image", "Image Folder", "Train Custom Model"]
    )
    
    if analysis_mode == "Single Image":
        uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Analysis options
                analysis_options = st.multiselect(
                    "Select Analysis Options",
                    ["Basic Stats", "Color Analysis", "Custom Model Prediction"]
                )
                
                if analysis_options:
                    # Perform selected analyses
                    if "Basic Stats" in analysis_options:
                        st.subheader("Basic Statistics")
                        stats = image_analyzer.get_basic_stats(image)
                        st.write(stats)
                    
                    if "Color Analysis" in analysis_options:
                        st.subheader("Color Analysis")
                        color_analysis = image_analyzer.analyze_colors(image)
                        st.write(color_analysis)
                    
                    if "Custom Model Prediction" in analysis_options:
                        st.subheader("Custom Model Prediction")
                        # Try to load existing model
                        if image_analyzer.load_model('best_model.pth'):
                            prediction = image_analyzer.predict_image(image)
                            if prediction:
                                st.write(f"Predicted Class: {prediction['class']}")
                                st.write(f"Confidence: {prediction['confidence']:.2f}")
                        else:
                            st.warning("No trained model found. Please train a model first.")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif analysis_mode == "Image Folder":
        folder_path = st.text_input("Enter folder path containing images:")
        if folder_path and os.path.isdir(folder_path):
            # Process all images in the folder
            results = image_analyzer.process_image_folder(folder_path)
            
            if results:
                st.subheader("Image Analysis Results")
                
                # Display summary statistics
                total_images = len(results)
                st.write(f"Total images processed: {total_images}")
                
                # Show individual image results
                for result in results:
                    with st.expander(f"{result['filename']}"):
                        st.write("Image Information:")
                        st.write(f"- Size: {result['size']}")
                        st.write(f"- Mode: {result['mode']}")
                        st.write(f"- Format: {result['format']}")
                        
                        st.write("\nBasic Statistics:")
                        st.write(result['basic_stats'])
                        
                        st.write("\nColor Analysis:")
                        st.write(result['color_analysis'])
                        
                        if 'prediction' in result and result['prediction']:
                            st.write("\nModel Prediction:")
                            st.write(f"- Class: {result['prediction']['class']}")
                            st.write(f"- Confidence: {result['prediction']['confidence']:.2f}")
            else:
                st.warning("No images found in the specified folder.")
        else:
            st.warning("Please enter a valid folder path.")
    
    else:  # Train Custom Model
        st.subheader("Train Custom Model")
        
        # Training data folder
        training_folder = st.text_input(
            "Enter training data folder path (should contain subfolders for each class):"
        )
        
        if training_folder:
            # Convert to Path object and normalize
            training_path = pathlib.Path(training_folder).resolve()
            
            if training_path.is_dir():
                # Training parameters
                epochs = st.slider("Number of epochs", min_value=1, max_value=50, value=10)
                batch_size = st.slider("Batch size", min_value=4, max_value=64, value=32)
                learning_rate = st.select_slider(
                    "Learning rate",
                    options=[0.0001, 0.001, 0.01, 0.1],
                    value=0.001,
                    format_func=lambda x: f"{x:.4f}"
                )
                
                if st.button("Start Training"):
                    try:
                        with st.spinner("Training model..."):
                            # Train the model
                            progress = image_analyzer.train_model(
                                str(training_path),
                                epochs=epochs,
                                batch_size=batch_size,
                                learning_rate=learning_rate
                            )
                            
                            if progress:
                                st.success("Training completed successfully!")
                                
                                # Plot training progress
                                st.subheader("Training Progress")
                                image_analyzer.plot_training_progress(progress)
                                
                                # Save the model
                                image_analyzer.save_model()
                                st.success("Model saved successfully!")
                            else:
                                st.error("Training failed. Please check your dataset.")
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
            else:
                st.error("Please enter a valid directory path")

def realtime_analysis_page(plot_config):
    st.title("Real-time Analysis")
    
    # Initialize analyzer
    analyzer = RealTimeAnalysis()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Stock Analysis", "News Analysis"])
    
    with tab1:
        st.subheader("Stock Analysis")
        
        # Stock symbol input with tooltip
        symbol = st.text_input(
            "Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL)",
            help="Enter the stock symbol of the company you want to analyze. For example, AAPL for Apple Inc."
        )
        
        # Analysis period selection with tooltip
        period = st.selectbox(
            "Select Analysis Period",
            ['1mo', '3mo', '6mo', '1y', '2y', '5y'],
            help="Choose the historical time period for analysis. Longer periods provide more context but may be less relevant for short-term trading."
        )
        
        if symbol:
            with st.spinner("Analyzing stock data..."):
                data = analyzer.get_stock_data(symbol, period)
                
                if data and 'historical' in data and 'forecast' in data:
                    # Display current stock info
                    info = data['info']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Current Price",
                            f"${info.get('currentPrice', 'N/A')}",
                            help="The latest trading price of the stock"
                        )
                    with col2:
                        st.metric(
                            "Market Cap",
                            f"${info.get('marketCap', 'N/A'):,}",
                            help="Total market value of the company's shares"
                        )
                    with col3:
                        st.metric(
                            "52 Week Range",
                            f"${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}",
                            help="Lowest and highest prices in the past year"
                        )
                    
                    # Technical Analysis Section
                    st.subheader("Technical Analysis", help="Analysis based on historical price and volume data")
                    
                    # Plot historical data and predictions
                    fig = go.Figure()
                    
                    # Historical prices
                    fig.add_trace(go.Scatter(
                        x=data['historical'].index,
                        y=data['historical']['Close'],
                        name="Historical Price",
                        line=dict(color='blue')
                    ))
                    
                    # Add technical indicators
                    if 'SMA_20' in data['historical'].columns:
                        fig.add_trace(go.Scatter(
                            x=data['historical'].index,
                            y=data['historical']['SMA_20'],
                            name="20-day SMA",
                            line=dict(color='orange', dash='dash')
                        ))
                    
                    # Add Bollinger Bands
                    if all(col in data['historical'].columns for col in ['BB_upper', 'BB_lower']):
                        fig.add_trace(go.Scatter(
                            x=data['historical'].index,
                            y=data['historical']['BB_upper'],
                            name="Upper BB",
                            line=dict(color='gray', dash='dot')
                        ))
                        fig.add_trace(go.Scatter(
                            x=data['historical'].index,
                            y=data['historical']['BB_lower'],
                            name="Lower BB",
                            line=dict(color='gray', dash='dot'),
                            fill='tonexty'
                        ))
                    
                    # Add future predictions
                    future_dates = pd.to_datetime(data['forecast']['ds'])
                    future_prices = data['forecast']['yhat']
                    future_lower = data['forecast']['yhat_lower']
                    future_upper = data['forecast']['yhat_upper']
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_prices,
                        name="Predicted Price",
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_upper,
                        name="Prediction Upper Bound",
                        line=dict(color='rgba(255,0,0,0.2)', dash='dot')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_lower,
                        name="Prediction Lower Bound",
                        line=dict(color='rgba(255,0,0,0.2)', dash='dot'),
                        fill='tonexty'
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} Stock Price Analysis and Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display predictions for tomorrow
                    if len(data['forecast']) >= 1:
                        tomorrow_pred = data['forecast'].iloc[-1]
                        st.subheader("Tomorrow's Prediction", help="AI-powered price prediction for tomorrow")
                        pred_col1, pred_col2, pred_col3 = st.columns(3)
                        with pred_col1:
                            st.metric(
                                "Predicted Price",
                                f"${tomorrow_pred['yhat']:.2f}",
                                help="Most likely price prediction"
                            )
                        with pred_col2:
                            st.metric(
                                "Upper Bound",
                                f"${tomorrow_pred['yhat_upper']:.2f}",
                                help="95% confidence interval upper bound"
                            )
                        with pred_col3:
                            st.metric(
                                "Lower Bound",
                                f"${tomorrow_pred['yhat_lower']:.2f}",
                                help="95% confidence interval lower bound"
                            )
                    
                    # Analysis Recommendations
                    if 'analysis' in data:
                        st.subheader("Analysis and Recommendations", help="Comprehensive analysis and recommendations")
                        
                        analysis = data['analysis']
                        
                        # Technical Analysis
                        with st.expander("Technical Analysis", expanded=True):
                            for point in analysis.get('Technical_Analysis', []):
                                st.info(point)
                                
                        # Fundamental Analysis
                        with st.expander("Fundamental Analysis", expanded=True):
                            for point in analysis.get('Fundamental_Analysis', []):
                                st.info(point)
                                
                        # Risk Analysis
                        with st.expander("Risk Analysis", expanded=True):
                            for point in analysis.get('Risk_Analysis', []):
                                st.warning(point)
                                
                        # Trading Recommendations
                        with st.expander("Trading Recommendations", expanded=True):
                            for rec in analysis.get('Recommendations', []):
                                st.success(rec)
                        
                        # Technical Indicators Explanation
                        with st.expander("Technical Indicators Explanation"):
                            for indicator, description in analyzer.stock_indicators.items():
                                st.markdown(f"**{indicator}**: {description}")
                                
    with tab2:
        st.subheader("📰 News Analysis")
        
        # News category selection
        category = st.selectbox(
            "Select News Category",
            ["business", "technology", "entertainment", "general", "health", "science", "sports"],
            help="Category for news articles. Select a category to get top headlines."
        )
        
        # Optional keyword filter
        keyword = st.text_input(
            "Filter by Keyword (optional)",
            help="Enter keywords to search for specific news. Leave empty to get general news in the selected category."
        )
        
        if st.button("Fetch News", help="Click to fetch and analyze latest news"):
            with st.spinner("Fetching news..."):
                # Fetch news data
                news_data = analyzer.get_news_data(query=keyword, category=category)
                
                if news_data:
                    # Analyze sentiment
                    sentiment_df = analyzer.analyze_news_sentiment(news_data)
                    
                    # Display sentiment trends
                    if not sentiment_df.empty:
                        st.subheader("📊 Sentiment Analysis")
                        fig = analyzer.plot_sentiment_trends(sentiment_df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Display news articles
                    st.subheader("📰 Latest News")
                    for article in news_data:
                        with st.expander(f"📄 {article['title']}"):
                            st.markdown(f"**Description:** {article['description']}")
                            st.markdown(f"**Source:** {article['source']}")
                            st.markdown(f"**Published:** {article['published_at']}")
                            st.markdown(f"[Read full article]({article['url']})")
                    
if __name__ == "__main__":
    main()
