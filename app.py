from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import tensorflow as tf
import os
import json
import sys

# Definiranje i registriranje ManualStandardScaler klase radi uspješne deserializacije pod Gunicornom
class ManualStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        if isinstance(self.scale_, pd.Series):
            self.scale_[self.scale_ == 0] = 1.0
        else:
            self.scale_[self.scale_ == 0] = 1.0
        return self
        
    def transform(self, X):
        return (X - self.mean_) / self.scale_
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)

import __main__
__main__.ManualStandardScaler = ManualStandardScaler
sys.modules['__main__'].ManualStandardScaler = ManualStandardScaler

# Smanjivanje tensorflow logova
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
OUTPUT_DIR = Path('outputs_v2')

# Globalni cache za učitane modele i skalere radi bržeg rada i uštede memorije na poslužitelju
MODEL_CACHE = {}

def load_cached_object(path, loader_func):
    path_str = str(path)
    if path_str not in MODEL_CACHE:
        # Ograničavamo cache kako bismo spriječili preveliku potrošnju RAM-a
        if len(MODEL_CACHE) >= 40:
            stari_kljuc = next(iter(MODEL_CACHE))
            del MODEL_CACHE[stari_kljuc]
        MODEL_CACHE[path_str] = loader_func(path)
    return MODEL_CACHE[path_str]

def izracunaj_live_znacajke(ticker):
    # Dohvati podatke (potrebno nam je barem 100 dana za pouzdano računanje 50-dnevnih pomičnih prosjeka i 10-dnevnog LSTM prozora)
    df = yf.download(ticker, period='100d', interval='1d', progress=False, auto_adjust=True)
    sp500 = yf.download('^GSPC', period='100d', interval='1d', progress=False, auto_adjust=True)
    vix = yf.download('^VIX', period='100d', interval='1d', progress=False, auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if isinstance(sp500.columns, pd.MultiIndex): sp500.columns = sp500.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)
    
    d = df.copy()
    
    # Prinosi
    for n in [1, 5, 10, 20]: 
        d[f'return_{n}d'] = d['Close'].pct_change(n)
        
    # Odnos prema pomičnim prosjecima
    for n in [5, 10, 20, 50]: 
        d[f'close_to_sma_{n}'] = d['Close'] / d['Close'].rolling(n).mean() - 1

    # MACD
    ema_12 = d['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = d['Close'].ewm(span=26, adjust=False).mean()
    d['macd'] = ema_12 - ema_26
    d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()
    d['macd_diff'] = d['macd'] - d['macd_signal']
    d['macd_norm'] = d['macd'] / d['Close']

    # RSI
    gain = d['Close'].diff().where(d['Close'].diff() > 0, 0.0).rolling(14).mean()
    loss = (-d['Close'].diff().where(d['Close'].diff() < 0, 0.0)).rolling(14).mean()
    d['rsi_14'] = 100 - 100 / (1 + (gain / loss.replace(0, np.nan)))

    # Bollinger Bands
    sma_20 = d['Close'].rolling(20).mean()
    std_20 = d['Close'].rolling(20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    d['bb_position'] = (d['Close'] - bb_lower) / (bb_upper - bb_lower)
    d['bb_width'] = (bb_upper - bb_lower) / sma_20 

    # ATR
    prev_close = d['Close'].shift(1)
    true_range = pd.concat([d['High'] - d['Low'], (d['High'] - prev_close).abs(), (d['Low'] - prev_close).abs()], axis=1).max(axis=1)
    d['atr_14'] = true_range.rolling(14).mean()
    d['atr_pct'] = d['atr_14'] / d['Close'] 
    
    # Volatilnost, volumen i momentum
    d['volatility_10d'] = d['return_1d'].rolling(10).std()
    d['volatility_30d'] = d['return_1d'].rolling(30).std()
    d['volume_change'] = d['Volume'].pct_change()
    d['volume_ratio'] = d['Volume'] / d['Volume'].rolling(10).mean()
    d['hl_range'] = (d['High'] - d['Low']) / d['Close']
    d['momentum_5'] = d['Close'] - d['Close'].shift(5)
    d['momentum_5_norm'] = d['momentum_5'] / d['Close'].shift(5)
    
    # Makroekonomski indikatori
    macro_df = pd.DataFrame({'sp500_return': sp500['Close'].pct_change(), 'vix_close': vix['Close'], 'vix_return': vix['Close'].pct_change()})
    d = d.join(macro_df, how='left').ffill().bfill()
    
    kolone = [
        'return_1d', 'return_5d', 'return_10d', 'return_20d', 
        'close_to_sma_5', 'close_to_sma_10', 'close_to_sma_20', 'close_to_sma_50', 
        'macd', 'macd_signal', 'macd_diff', 'macd_norm', 
        'rsi_14', 'bb_position', 'bb_width', 
        'volatility_10d', 'volatility_30d', 'atr_pct', 
        'volume_change', 'volume_ratio', 'hl_range', 'momentum_5_norm', 
        'sp500_return', 'vix_close', 'vix_return'
    ]
    
    # Dohvaćamo zadnjih 10 redova kako bismo imali dovoljno povijesti za LSTM (lookback=10)
    zadnji_redovi = d[kolone].tail(10)
    trenutna_cijena = float(round(d['Close'].iloc[-1], 2)) 
    
    return zadnji_redovi, trenutna_cijena

@app.route('/')
def index():
    return render_template('index.html')

# API za "Market Overview" traku na vrhu stranice
@app.route('/api/market_overview')
def market_overview():
    tickers = {'S&P 500': '^GSPC', 'Nasdaq': '^IXIC', 'Dow Jones': '^DJI', 'VIX': '^VIX', 'Bitcoin': 'BTC-USD'}
    results = []
    for name, t in tickers.items():
        try:
            df = yf.download(t, period='5d', interval='1d', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=['Close'])
            if len(df) >= 2:
                close_today = float(df['Close'].iloc[-1])
                close_yest = float(df['Close'].iloc[-2])
                pct_change = ((close_today - close_yest) / close_yest) * 100
                results.append({"name": name, "price": round(close_today, 2), "change": round(pct_change, 2)})
        except Exception as e:
            print(f"Error loading {name}: {e}")
    return jsonify(results)

# API za TradingView Chart
@app.route('/api/chart/<ticker>')
def get_chart_data(ticker):
    df = yf.download(ticker, period='6mo', interval='1d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=['Close', 'Open', 'High', 'Low'])
    
    chart_data = []
    for index, row in df.iterrows():
        chart_data.append({
            "time": index.strftime('%Y-%m-%d'),
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
            "close": float(row['Close'])
        })
    return jsonify(chart_data)

# API za dohvaćanje popisa svih 54 dionica grupiranih po sektorima
@app.route('/api/tickers')
def get_tickers():
    sektori = {
        "Tehnologija": [
            {"ticker": "AAPL", "naziv": "Apple Inc. (AAPL)"},
            {"ticker": "MSFT", "naziv": "Microsoft Corp. (MSFT)"},
            {"ticker": "GOOGL", "naziv": "Alphabet Inc. (GOOGL)"},
            {"ticker": "AMZN", "naziv": "Amazon.com Inc. (AMZN)"},
            {"ticker": "META", "naziv": "Meta Platforms Inc. (META)"},
            {"ticker": "NVDA", "naziv": "NVIDIA Corp. (NVDA)"},
            {"ticker": "AVGO", "naziv": "Broadcom Inc. (AVGO)"},
            {"ticker": "CSCO", "naziv": "Cisco Systems Inc. (CSCO)"},
            {"ticker": "ADBE", "naziv": "Adobe Inc. (ADBE)"},
            {"ticker": "CRM", "naziv": "Salesforce Inc. (CRM)"},
            {"ticker": "AMD", "naziv": "Advanced Micro Devices (AMD)"},
            {"ticker": "QCOM", "naziv": "Qualcomm Inc. (QCOM)"},
            {"ticker": "INTC", "naziv": "Intel Corp. (INTC)"},
            {"ticker": "TXN", "naziv": "Texas Instruments (TXN)"}
        ],
        "Financije": [
            {"ticker": "JPM", "naziv": "JPMorgan Chase (JPM)"},
            {"ticker": "BAC", "naziv": "Bank of America (BAC)"},
            {"ticker": "MS", "naziv": "Morgan Stanley (MS)"},
            {"ticker": "GS", "naziv": "Goldman Sachs (GS)"},
            {"ticker": "V", "naziv": "Visa Inc. (V)"},
            {"ticker": "MA", "naziv": "Mastercard Inc. (MA)"},
            {"ticker": "AXP", "naziv": "American Express (AXP)"}
        ],
        "Zdravstvo": [
            {"ticker": "JNJ", "naziv": "Johnson & Johnson (JNJ)"},
            {"ticker": "LLY", "naziv": "Eli Lilly & Co. (LLY)"},
            {"ticker": "UNH", "naziv": "UnitedHealth Group (UNH)"},
            {"ticker": "ABBV", "naziv": "AbbVie Inc. (ABBV)"},
            {"ticker": "MRK", "naziv": "Merck & Co. (MRK)"},
            {"ticker": "PFE", "naziv": "Pfizer Inc. (PFE)"},
            {"ticker": "TMO", "naziv": "Thermo Fisher (TMO)"},
            {"ticker": "ABT", "naziv": "Abbott Laboratories (ABT)"},
            {"ticker": "CVS", "naziv": "CVS Health Corp. (CVS)"}
        ],
        "Potrošačka dobra": [
            {"ticker": "TSLA", "naziv": "Tesla, Inc. (TSLA)"},
            {"ticker": "HD", "naziv": "Home Depot (HD)"},
            {"ticker": "MCD", "naziv": "McDonald's Corp. (MCD)"},
            {"ticker": "NKE", "naziv": "Nike, Inc. (NKE)"},
            {"ticker": "SBUX", "naziv": "Starbucks Corp. (SBUX)"},
            {"ticker": "WMT", "naziv": "Walmart Inc. (WMT)"},
            {"ticker": "PG", "naziv": "Procter & Gamble (PG)"},
            {"ticker": "KO", "naziv": "Coca-Cola Co. (KO)"},
            {"ticker": "PEP", "naziv": "PepsiCo, Inc. (PEP)"},
            {"ticker": "EL", "naziv": "Estée Lauder Cos. (EL)"}
        ],
        "Energija & Industrija": [
            {"ticker": "XOM", "naziv": "Exxon Mobil (XOM)"},
            {"ticker": "CVX", "naziv": "Chevron Corp. (CVX)"},
            {"ticker": "CAT", "naziv": "Caterpillar Inc. (CAT)"},
            {"ticker": "GE", "naziv": "General Electric (GE)"},
            {"ticker": "HON", "naziv": "Honeywell (HON)"},
            {"ticker": "UNP", "naziv": "Union Pacific (UNP)"},
            {"ticker": "UPS", "naziv": "United Parcel Service (UPS)"}
        ],
        "Komunikacije, usluge & ostalo": [
            {"ticker": "NFLX", "naziv": "Netflix Inc. (NFLX)"},
            {"ticker": "DIS", "naziv": "Walt Disney (DIS)"},
            {"ticker": "T", "naziv": "AT&T Inc. (T)"},
            {"ticker": "VZ", "naziv": "Verizon (VZ)"},
            {"ticker": "NEE", "naziv": "NextEra Energy (NEE)"},
            {"ticker": "PLTR", "naziv": "Palantir Technologies (PLTR)"},
            {"ticker": "AMT", "naziv": "American Tower (AMT)"}
        ]
    }
    return jsonify(sektori)

# API za dohvaćanje metrika performansi iz modela
@app.route('/api/performance/<ticker>')
def get_performance(ticker):
    try:
        csv_path = OUTPUT_DIR / 'validation_metrics.csv'
        if not csv_path.exists():
            return jsonify({"error": "Metrike performansi još nisu generirane."}), 404
            
        df = pd.read_csv(csv_path)
        df_ticker = df[df['Ticker'] == ticker]
        
        results = []
        for _, row in df_ticker.iterrows():
            results.append({
                "model": row['Model'],
                "accuracy": float(row['Accuracy']),
                "precision": float(row['Precision']),
                "recall": float(row['Recall']),
                "f1": float(row['F1']),
                "auc": float(row['ROC-AUC'])
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API za dohvaćanje rezultata simulacije trgovanja
@app.route('/api/simulation/<ticker>')
def get_simulation(ticker):
    try:
        csv_path = OUTPUT_DIR / 'trading_simulation.csv'
        if not csv_path.exists():
            return jsonify({"error": "Simulacijske metrike još nisu generirane."}), 404
            
        df = pd.read_csv(csv_path)
        df_ticker = df[df['Ticker'] == ticker]
        if len(df_ticker) == 0:
            return jsonify({"error": f"Nema simulacije za dionicu {ticker}."}), 404
            
        row = df_ticker.iloc[0]
        return jsonify({
            "ticker": ticker,
            "model": row['Model'],
            "buy_hold_return": float(row['Buy & Hold (%)']),
            "model_return": float(row['Model (%)']),
            "outperformance": float(row['Razlika (pp)'])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API za ML Predikcije (STVARNE, bez simulacije)
@app.route('/api/predict/<ticker>/<model_name>')
def predict_live(ticker, model_name):
    try:
        X_live, danas_cijena = izracunaj_live_znacajke(ticker)
        
        # Datotečno ime modela
        model_filename_part = model_name.replace(' ', '_')
        
        # Učitavanje optimalnog praga za klasifikaciju (threshold tuning)
        threshold = 0.50
        thresholds_path = OUTPUT_DIR / 'thresholds.json'
        if thresholds_path.exists():
            try:
                with open(thresholds_path, 'r') as f:
                    thresh_dict = json.load(f)
                if ticker in thresh_dict and model_name in thresh_dict[ticker]:
                    threshold = float(thresh_dict[ticker][model_name])
            except Exception as e:
                print(f"Greška pri čitanju pragova: {e}")
        
        # LSTM učitavanje i predikcija
        if model_name == "LSTM":
            model_path = OUTPUT_DIR / f'{ticker}_LSTM.keras'
            scaler_path = OUTPUT_DIR / f'{ticker}_scaler.joblib'
            
            if not model_path.exists() or not scaler_path.exists():
                return jsonify({"error": f"LSTM Model ili skaler za {ticker} ne postoji."}), 404
                
            model = load_cached_object(model_path, tf.keras.models.load_model)
            scaler = load_cached_object(scaler_path, joblib.load)
            
            # Skaliranje i dimenzioniranje za LSTM (batch_size, timesteps, features)
            # LSTM model očekuje sekvencu od 10 koraka (1, 10, 25)
            X_live_scaled = scaler.transform(X_live)
            X_live_scaled_np = np.array(X_live_scaled)
            X_live_lstm = X_live_scaled_np.reshape((1, 10, X_live_scaled_np.shape[1]))
            
            proba = float(model.predict(X_live_lstm, verbose=0)[0, 0])
            
        # Klasični modeli
        else:
            model_path = OUTPUT_DIR / f'{ticker}_{model_filename_part}.joblib'
            if not model_path.exists():
                return jsonify({"error": f"Model {model_name} za {ticker} nije pronađen."}), 404
                
            model = load_cached_object(model_path, joblib.load)
            
            # Za klasične modele koristimo samo zadnji red (posljednji trgovački dan)
            X_live_last = X_live.iloc[[-1]]
            
            # Logistička regresija zahtijeva skaliranje
            if model_name == "Logistic Regression":
                scaler_path = OUTPUT_DIR / f'{ticker}_scaler.joblib'
                if not scaler_path.exists():
                    return jsonify({"error": f"Skaler za {ticker} nije pronađen."}), 404
                scaler = load_cached_object(scaler_path, joblib.load)
                X_live_in = scaler.transform(X_live_last)
            else:
                X_live_in = X_live_last.values
                
            proba = float(model.predict_proba(X_live_in)[0, 1])

        # Klasifikacija na temelju optimalnog praga
        pred = "RAST" if proba >= threshold else "PAD"
        
        # Ekstrakcija pokazatelja za UI
        last_row = X_live.iloc[-1]
        rsi_val = float(last_row['rsi_14'])
        macd_diff_val = float(last_row['macd_diff'])
        bb_pos_val = float(last_row['bb_position'])
        vix_val = float(last_row['vix_close'])
        daily_change = float(last_row['return_1d']) * 100
        
        # Provjera NaN vrijednosti
        def clean_val(val, default):
            return default if np.isnan(val) or np.isinf(val) else val
            
        return jsonify({
            "Ticker": ticker,
            "CijenaDanas": danas_cijena,
            "Predikcija": pred,
            "Vjerojatnost": round(proba * 100, 2),
            "OdabraniModel": model_name,
            "OptimalniPrag": round(threshold * 100, 2),
            "IndicatorRSI": round(clean_val(rsi_val, 50.0), 2),
            "IndicatorMACD": round(clean_val(macd_diff_val, 0.0), 4),
            "IndicatorBB": round(clean_val(bb_pos_val, 0.5) * 100, 2),
            "IndicatorVIX": round(clean_val(vix_val, 15.0), 2),
            "DailyChangePct": round(clean_val(daily_change, 0.0), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)