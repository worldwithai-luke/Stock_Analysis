# Stock Analysis Dashboard

A Streamlit-based web application for stock market analysis and prediction.

## Features

- Real-time stock data from Yahoo Finance
- Interactive price charts with technical indicators
- Machine learning-based price predictions
- Technical analysis with RSI, MACD, and Bollinger Bands
- Volume analysis and trend visualization

## Local Development Setup

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

1. Clone or download this project to your local computer

2. Open a terminal/command prompt and navigate to the project folder:
   ```bash
   cd path/to/your/project
   ```

3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Option 1: Quick Start (Easiest)
```bash
python run_local.py
```
This script will check your setup and start the application automatically.

#### Option 2: Manual Setup

1. Make sure your virtual environment is activated

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your web browser and go to: `http://localhost:8501`

#### Option 3: Automated Setup Scripts

**For Windows:**
```bash
setup_local.bat
```

**For Linux/macOS:**
```bash
./setup_local.sh
```

These scripts will create a virtual environment and install all dependencies automatically.

### Using the Application

1. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT) in the sidebar
2. Select a time period for historical data
3. View interactive charts showing:
   - Price history with moving averages
   - Volume analysis
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Price predictions

### Troubleshooting

- If you get import errors, make sure all packages are installed: `pip install -r requirements.txt`
- If the app won't start, check that you're using Python 3.11 or higher: `python --version`
- If charts don't load, check your internet connection (required for Yahoo Finance data)

### Project Structure

```
├── app.py                 # Main Streamlit application
├── utils/
│   ├── stock_data.py     # Yahoo Finance data fetching
│   └── predictions.py    # Machine learning predictions
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Dependencies

- **streamlit**: Web application framework
- **yfinance**: Yahoo Finance API for stock data
- **plotly**: Interactive data visualization
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms