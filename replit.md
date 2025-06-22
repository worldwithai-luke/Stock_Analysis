# Stock Analysis Dashboard

## Overview

This is a Streamlit-based web application for stock market analysis and prediction. The application fetches real-time stock data from Yahoo Finance, performs technical analysis, and provides price predictions using machine learning. It features an interactive dashboard with charts, technical indicators, and forecasting capabilities.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web UI
- **Visualization**: Plotly for interactive charts and graphs
- **Layout**: Wide layout with sidebar navigation for user inputs
- **Session State Management**: Streamlit session state for data persistence across interactions

### Backend Architecture
- **Data Processing**: Pandas for data manipulation and analysis
- **Machine Learning**: Scikit-learn for price prediction models
- **API Integration**: yfinance library for Yahoo Finance data access
- **Modular Design**: Utility modules for data fetching and predictions

### Technology Stack
- **Python 3.11**: Core runtime environment
- **Streamlit**: Web application framework
- **Plotly**: Interactive data visualization
- **yfinance**: Stock data API wrapper
- **Pandas**: Data analysis and manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Central application entry point and UI orchestration
- **Features**: Dashboard layout, user interaction handling, chart rendering
- **Session Management**: Maintains stock data, info, and predictions in session state

### 2. Stock Data Fetcher (`utils/stock_data.py`)
- **Purpose**: Yahoo Finance API integration and data retrieval
- **Caching**: 5-minute TTL caching for API responses
- **Error Handling**: Comprehensive error handling for API failures
- **Data Validation**: Ensures data completeness and quality

### 3. Stock Predictor (`utils/predictions.py`)
- **Purpose**: Machine learning-based price prediction
- **Algorithm**: Linear Regression with technical indicators
- **Features**: Moving averages, trend analysis, feature engineering
- **Scalability**: Handles both short-term and long-term predictions

### 4. Configuration Files
- **pyproject.toml**: Project dependencies and metadata
- **.replit**: Deployment and runtime configuration
- **uv.lock**: Dependency lock file for reproducible builds

## Data Flow

1. **User Input**: Stock symbol and time period selection via sidebar
2. **Data Fetching**: StockDataFetcher retrieves historical data from Yahoo Finance
3. **Data Processing**: Raw data is cleaned and technical indicators are calculated
4. **Visualization**: Plotly charts display price history and technical analysis
5. **Prediction**: StockPredictor generates future price forecasts
6. **Display**: Results are rendered in the Streamlit dashboard

## External Dependencies

### Core Dependencies
- **yfinance**: Yahoo Finance API for stock data
- **streamlit**: Web application framework
- **plotly**: Interactive visualization library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms

### Data Sources
- **Yahoo Finance**: Primary source for stock market data
- **Real-time Data**: Live stock prices and historical data
- **Market Information**: Company info, financial metrics, and market data

## Deployment Strategy

### Platform
- **Replit**: Cloud-based development and deployment platform
- **Autoscale Deployment**: Automatic scaling based on traffic

### Runtime Configuration
- **Python 3.11**: Specified runtime environment
- **Port Configuration**: Application runs on port 5000, exposed on port 80
- **Streamlit Server**: Configured for production deployment

### Dependency Management
- **UV Package Manager**: Used for dependency installation and management
- **Lock File**: uv.lock ensures reproducible dependency resolution
- **Automatic Installation**: Dependencies installed via workflow automation

### Workflows
- **Parallel Execution**: Main project workflow runs tasks in parallel
- **Dependency Installation**: Automated installation of required packages
- **Application Startup**: Streamlit server starts automatically on deployment

## Local Development Support

### Local Setup Files
- **README.md**: Complete local development guide
- **local_requirements.txt**: Python dependencies for local installation
- **run_local.py**: Interactive setup and runner script
- **start_local.py**: Quick start script for local development
- **setup_local.sh**: Automated setup for Linux/macOS
- **setup_local.bat**: Automated setup for Windows
- **.streamlit/config.toml**: Streamlit configuration for both local and cloud deployment

### Local Development Features
- Automatic dependency checking and installation
- Virtual environment setup scripts
- Cross-platform compatibility (Windows, macOS, Linux)
- Browser auto-launch functionality
- Error handling and troubleshooting guidance

## Changelog

- June 22, 2025. Initial setup
- June 22, 2025. Added comprehensive local development support with setup scripts and documentation

## User Preferences

Preferred communication style: Simple, everyday language.