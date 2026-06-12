import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Smanji TensorFlow logove

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

class ManualStandardScaler:
    """
    Ručna implementacija standardizacije podataka za demonstraciju razumijevanja algoritma.
    """
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


# Postavljanje sjemena za ponovljivost
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

TICKERS = [
    # Tehnologija
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM', 'AMD', 'QCOM', 'INTC', 'TXN',
    # Financije
    'JPM', 'BAC', 'MS', 'GS', 'V', 'MA', 'AXP',
    # Zdravstvo
    'JNJ', 'LLY', 'UNH', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'CVS',
    # Potrošačka dobra
    'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'WMT', 'PG', 'KO', 'PEP', 'EL',
    # Energija i industrija
    'XOM', 'CVX', 'CAT', 'GE', 'HON', 'UNP', 'UPS',
    # Komunikacije, usluge i ostalo
    'NFLX', 'DIS', 'T', 'VZ', 'NEE', 'PLTR', 'AMT'
]

START_DATE = '2015-01-01'
END_DATE = '2025-01-01'
OUTPUT_DIR = Path('outputs_v2')
OUTPUT_DIR.mkdir(exist_ok=True)

# Definiramo duljinu povijesnog prozora (lookback) za sekvencijalni LSTM model
LOOKBACK = 10 

# Ručne implementacije evaluacijskih metrika
def manual_confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def manual_accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

def manual_precision(y_true, y_pred):
    tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
    fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def manual_recall(y_true, y_pred):
    tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
    fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def manual_f1_score(y_true, y_pred):
    p = manual_precision(y_true, y_pred)
    r = manual_recall(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

def manual_precision_class0(y_true, y_pred):
    tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
    fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0

def manual_recall_class0(y_true, y_pred):
    tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
    fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def manual_f1_score_class0(y_true, y_pred):
    p0 = manual_precision_class0(y_true, y_pred)
    r0 = manual_recall_class0(y_true, y_pred)
    return 2 * (p0 * r0) / (p0 + r0) if (p0 + r0) > 0 else 0.0

def manual_macro_f1_score(y_true, y_pred):
    f1_1 = manual_f1_score(y_true, y_pred)
    f1_0 = manual_f1_score_class0(y_true, y_pred)
    return (f1_1 + f1_0) / 2.0

# Ručno kreiranje sekvenci za LSTM (trodimenzionalni ulaz)
def create_sequences(X, y, lookback=10):
    X_arr = np.array(X)
    y_arr = np.array(y)
    Xs, ys = [], []
    for i in range(len(X_arr) - lookback):
        Xs.append(X_arr[i:(i + lookback)])
        ys.append(y_arr[i + lookback])
    return np.array(Xs), np.array(ys)

# Ručna optimizacija klasifikacijskog praga (threshold tuning) na validacijskom skupu pomoću Macro F1 metrike
def optimize_threshold(y_true, y_proba):
    best_thresh = 0.50
    best_macro = 0.0
    for thresh in np.arange(0.35, 0.66, 0.01):
        preds = (y_proba >= thresh).astype(int)
        macro = manual_macro_f1_score(y_true, preds)
        if macro > best_macro:
            best_macro = macro
            best_thresh = thresh
    return float(best_thresh)

# Preuzimanje makroekonomskih indikatora
print("Preuzimanje makroekonomskih indikatora (S&P 500 i VIX)...")
sp500 = yf.download('^GSPC', start=START_DATE, end=END_DATE, interval='1d', progress=False, auto_adjust=True)
vix = yf.download('^VIX', start=START_DATE, end=END_DATE, interval='1d', progress=False, auto_adjust=True)

if isinstance(sp500.columns, pd.MultiIndex): sp500.columns = sp500.columns.get_level_values(0)
if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

macro_df = pd.DataFrame({
    'sp500_return': sp500['Close'].pct_change(),
    'vix_close': vix['Close'],
    'vix_return': vix['Close'].pct_change()
})

def compute_features(df):
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
    
    # Prinosi
    for n in [1, 5, 10, 20]:
        d[f'return_{n}d'] = d['Close'].pct_change(n)
        
    # Odnos prema pomičnim prosjecima
    for n in [5, 10, 20, 50]:
        sma = d['Close'].rolling(n).mean()
        d[f'close_to_sma_{n}'] = d['Close'] / sma - 1

    # MACD
    ema_12 = d['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = d['Close'].ewm(span=26, adjust=False).mean()
    d['macd'] = ema_12 - ema_26
    d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()
    d['macd_diff'] = d['macd'] - d['macd_signal']
    d['macd_norm'] = d['macd'] / d['Close']

    # RSI
    delta = d['Close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d['rsi_14'] = 100 - 100 / (1 + rs)

    # Bollinger Bands
    sma_20 = d['Close'].rolling(20).mean()
    std_20 = d['Close'].rolling(20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    d['bb_position'] = (d['Close'] - bb_lower) / (bb_upper - bb_lower)
    d['bb_width'] = (bb_upper - bb_lower) / sma_20

    # ATR
    prev_close = d['Close'].shift(1)
    tr1 = d['High'] - d['Low']
    tr2 = (d['High'] - prev_close).abs()
    tr3 = (d['Low'] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
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

    # Spajanje makroekonomskih značajki
    d = d.join(macro_df, how='left')
    d['sp500_return'] = d['sp500_return'].ffill().bfill()
    d['vix_close'] = d['vix_close'].ffill().bfill()
    d['vix_return'] = d['vix_return'].ffill().bfill()

    return d

FEATURE_COLUMNS = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'close_to_sma_5', 'close_to_sma_10', 'close_to_sma_20', 'close_to_sma_50',
    'macd', 'macd_signal', 'macd_diff', 'macd_norm',
    'rsi_14', 'bb_position', 'bb_width',
    'volatility_10d', 'volatility_30d',
    'atr_pct', 'volume_change', 'volume_ratio',
    'hl_range', 'momentum_5_norm',
    'sp500_return', 'vix_close', 'vix_return'
]

def prepare_target_and_clean(df):
    d = df.copy()
    d['target'] = (d['Close'].shift(-1) > d['Close']).astype(int)
    d = d.dropna(subset=FEATURE_COLUMNS + ['target']).iloc[:-1]
    return d

def chronological_split(df, train_end='2022-12-31', val_end='2023-12-31'):
    X = df[FEATURE_COLUMNS]
    y = df['target']
    X_train, y_train = X.loc[:train_end], y.loc[:train_end]
    
    X_val_raw = X.loc[train_end:val_end]
    y_val_raw = y.loc[train_end:val_end]
    if train_end in X_val_raw.index:
        X_val = X_val_raw.iloc[1:]
        y_val = y_val_raw.iloc[1:]
    else:
        X_val = X_val_raw
        y_val = y_val_raw
        
    X_test_raw = X.loc[val_end:]
    y_test_raw = y.loc[val_end:]
    if val_end in X_test_raw.index:
        X_test = X_test_raw.iloc[1:]
        y_test = y_test_raw.iloc[1:]
    else:
        X_test = X_test_raw
        y_test = y_test_raw
        
    return X_train, y_train, X_val, y_val, X_test, y_test

# Liste za spremanje metrika
all_val_metrics = []
all_test_metrics = []
all_sim_results = []
feature_importances_list = []
ticker_thresholds = {}

# Petlja po svim dionicama
total_tickers = len(TICKERS)
for idx, ticker in enumerate(TICKERS, 1):
    print(f"[{idx}/{total_tickers}] Obrada dionice {ticker}...")
    try:
        # 1. Učitavanje i obrada podataka
        df_raw = yf.download(ticker, start=START_DATE, end=END_DATE, interval='1d', progress=False, auto_adjust=True)
        if len(df_raw) < 100:
            print(f"  Nedovoljno podataka za {ticker}, preskačem.")
            continue
            
        df_feat = compute_features(df_raw)
        df_clean = prepare_target_and_clean(df_feat)
        
        # 2. Split
        X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(df_clean)
        
        # Izračunavanje težina klasa
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        total_count = len(y_train)
        w0 = total_count / (2.0 * neg_count)
        w1 = total_count / (2.0 * pos_count)
        class_weight_dict = {0: float(w0), 1: float(w1)}
        
        # 3. Skaliranje (StandardScaler) i spremanje skalera
        scaler = ManualStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        joblib.dump(scaler, OUTPUT_DIR / f'{ticker}_scaler.joblib')
        
        # 4. Obuka modela
        # Logistic Regression
        lr = LogisticRegression(C=1.0, penalty='l2', class_weight='balanced', solver='liblinear', random_state=SEED)
        lr.fit(X_train_scaled, y_train)
        joblib.dump(lr, OUTPUT_DIR / f'{ticker}_Logistic_Regression.joblib')
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=50, class_weight='balanced', random_state=SEED, n_jobs=-1)
        rf.fit(X_train.values, y_train)
        joblib.dump(rf, OUTPUT_DIR / f'{ticker}_Random_Forest.joblib')
        
        # XGBoost
        scale_pos_weight = neg_count / pos_count
        xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_state=SEED, eval_metric='logloss', n_jobs=-1)
        xgb.fit(X_train.values, y_train)
        joblib.dump(xgb, OUTPUT_DIR / f'{ticker}_XGBoost.joblib')
        
        # LSTM - KREIRANJE SEKVENCI (Prava vremenska serija s pogledom unazad od 10 dana)
        X_train_lstm, y_train_lstm = create_sequences(X_train_scaled, y_train, LOOKBACK)
        X_val_lstm, y_val_lstm = create_sequences(X_val_scaled, y_val, LOOKBACK)
        X_test_lstm, y_test_lstm = create_sequences(X_test_scaled, y_test, LOOKBACK)
        
        # LSTM s tanh aktivacijom (standard za sprečavanje gradijentnih eksplozija i monotone predikcije klasa)
        lstm = Sequential([
            LSTM(32, input_shape=(LOOKBACK, X_train_scaled.shape[1]), activation="tanh", kernel_regularizer=l2(0.005)),
            Dropout(0.3),
            Dense(16, activation="relu", kernel_regularizer=l2(0.005)),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ])
        lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        
        # Rano zaustavljanje prati validacijski gubitak sekvencijalnih podataka
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        lstm.fit(
            X_train_lstm, y_train_lstm, 
            epochs=20, batch_size=64, 
            validation_data=(X_val_lstm, y_val_lstm),
            class_weight=class_weight_dict,
            callbacks=[early_stopping],
            verbose=0
        )
        lstm.save(OUTPUT_DIR / f'{ticker}_LSTM.keras')
        
        # 5. Dobivanje sirovih vjerojatnosti na validacijskom skupu
        y_val_prob_lr = lr.predict_proba(X_val_scaled)[:, 1]
        y_val_prob_rf = rf.predict_proba(X_val.values)[:, 1]
        y_val_prob_xgb = xgb.predict_proba(X_val.values)[:, 1]
        y_val_prob_lstm = lstm.predict(X_val_lstm, verbose=0).flatten()
        
        # Optimiziranje pragova na validacijskom skupu (za LSTM koristimo y_val_lstm!)
        t_lr = optimize_threshold(y_val, y_val_prob_lr)
        t_rf = optimize_threshold(y_val, y_val_prob_rf)
        t_xgb = optimize_threshold(y_val, y_val_prob_xgb)
        t_lstm = optimize_threshold(y_val_lstm, y_val_prob_lstm)
        
        ticker_thresholds[ticker] = {
            'Logistic Regression': t_lr,
            'Random Forest': t_rf,
            'XGBoost': t_xgb,
            'LSTM': t_lstm
        }
        
        y_val_pred_lr = (y_val_prob_lr >= t_lr).astype(int)
        y_val_pred_rf = (y_val_prob_rf >= t_rf).astype(int)
        y_val_pred_xgb = (y_val_prob_xgb >= t_xgb).astype(int)
        y_val_pred_lstm = (y_val_prob_lstm >= t_lstm).astype(int)
        
        models_preds = {
            'Logistic Regression': (y_val_pred_lr, y_val_prob_lr, t_lr, y_val),
            'Random Forest': (y_val_pred_rf, y_val_prob_rf, t_rf, y_val),
            'XGBoost': (y_val_pred_xgb, y_val_prob_xgb, t_xgb, y_val),
            'LSTM': (y_val_pred_lstm, y_val_prob_lstm, t_lstm, y_val_lstm)
        }
        
        # Izračun metrika za val skup pomoću optimiziranih pragova
        for model_name, (preds, probs, thresh, y_true_eval) in models_preds.items():
            acc = manual_accuracy(y_true_eval, preds)
            prec = manual_precision(y_true_eval, preds)
            rec = manual_recall(y_true_eval, preds)
            f1 = manual_f1_score(y_true_eval, preds)
            
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_true_eval, probs)
            except:
                auc = 0.50
                
            all_val_metrics.append({
                'Model': model_name,
                'Ticker': ticker,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1': f1,
                'ROC-AUC': auc,
                'Best params': f"Threshold: {thresh:.2f}"
            })
            
        # Odabir najboljeg modela na temelju optimiziranog F1 rezultata na validaciji
        val_f1s = {
            'Logistic Regression': manual_f1_score(y_val, y_val_pred_lr),
            'Random Forest': manual_f1_score(y_val, y_val_pred_rf),
            'XGBoost': manual_f1_score(y_val, y_val_pred_xgb),
            'LSTM': manual_f1_score(y_val_lstm, y_val_pred_lstm)
        }
        best_model_name = max(val_f1s, key=val_f1s.get)
        best_thresh = ticker_thresholds[ticker][best_model_name]
        
        # 6. Evaluacija na Test setu (2024) s najboljim modelom i pragom
        if best_model_name == 'Logistic Regression':
            y_test_prob = lr.predict_proba(X_test_scaled)[:, 1]
            y_test_eval = y_test
        elif best_model_name == 'Random Forest':
            y_test_prob = rf.predict_proba(X_test.values)[:, 1]
            y_test_eval = y_test
        elif best_model_name == 'XGBoost':
            y_test_prob = xgb.predict_proba(X_test.values)[:, 1]
            y_test_eval = y_test
        else: # LSTM
            y_test_prob = lstm.predict(X_test_lstm, verbose=0).flatten()
            y_test_eval = y_test_lstm
            
        y_test_pred = (y_test_prob >= best_thresh).astype(int)
        
        t_acc = manual_accuracy(y_test_eval, y_test_pred)
        t_prec = manual_precision(y_test_eval, y_test_pred)
        t_rec = manual_recall(y_test_eval, y_test_pred)
        t_f1 = manual_f1_score(y_test_eval, y_test_pred)
        try:
            from sklearn.metrics import roc_auc_score
            t_auc = roc_auc_score(y_test_eval, y_test_prob)
        except:
            t_auc = 0.50
            
        all_test_metrics.append({
            'Ticker': ticker,
            'Best model': best_model_name,
            'Accuracy': t_acc,
            'Precision': t_prec,
            'Recall': t_rec,
            'F1': t_f1,
            'ROC-AUC': t_auc,
            'N test': len(y_test_eval)
        })
        
        # 7. Trading simulacija na test setu (2024)
        test_dates = X_test.index
        if best_model_name == 'LSTM':
            test_dates_eval = test_dates[LOOKBACK:]
        else:
            test_dates_eval = test_dates
            
        close_prices = df_clean.loc[test_dates_eval, 'Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]
            
        next_returns = close_prices.pct_change().shift(-1).fillna(0)
        
        bh_equity = (1 + next_returns).cumprod()
        model_returns = next_returns * y_test_pred
        model_equity = (1 + model_returns).cumprod()
        
        bh_total = (bh_equity.iloc[-1] - 1) * 100
        model_total = (model_equity.iloc[-1] - 1) * 100
        
        all_sim_results.append({
            'Ticker': ticker,
            'Model': best_model_name,
            'Buy & Hold (%)': round(bh_total, 2),
            'Model (%)': round(model_total, 2),
            'Razlika (pp)': round(model_total - bh_total, 2)
        })
        
        feature_importances_list.append(xgb.feature_importances_)
        
        print(f"  Najbolji model: {best_model_name} | Prag: {best_thresh:.2f} | Test F1: {t_f1:.4f}")
    except Exception as e:
        print(f"  Greška kod dionice {ticker}: {str(e)}")

# Spremanje svih rezultata u CSV
print("\nSpremanje svih rezultata...")
validation_df = pd.DataFrame(all_val_metrics)
validation_df.to_csv(OUTPUT_DIR / 'validation_metrics.csv', index=False)

test_df = pd.DataFrame(all_test_metrics)
test_df.to_csv(OUTPUT_DIR / 'test_metrics_2024.csv', index=False)

sim_df = pd.DataFrame(all_sim_results)
sim_df.to_csv(OUTPUT_DIR / 'trading_simulation.csv', index=False)

# Spremanje optimalnih pragova u JSON datoteku
with open(OUTPUT_DIR / 'thresholds.json', 'w') as f:
    json.dump(ticker_thresholds, f, indent=4)

if feature_importances_list:
    mean_importances = np.mean(feature_importances_list, axis=0)
    imp_df = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Importance': mean_importances
    }).sort_values('Importance', ascending=False)
    imp_df.to_csv(OUTPUT_DIR / 'top_features.csv', index=False)

print("Sve modelske datoteke uspješno generirane s optimiziranim sekvencijalnim LSTM modelom!")
