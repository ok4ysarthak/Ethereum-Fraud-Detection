# backend/app.py
import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
import logging
from datetime import datetime
import threading
import time
from flask_cors import CORS

# Import your other modules (make sure these modules exist in backend/)
from data_fetcher import EthereumDataFetcher
from wallet_scorer import WalletScorer
from batch_processor import BatchProcessor
from config import Config

app = Flask(__name__)
CORS(app)

# --- Setup Logging ---
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        try:
            handler.stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

logger = logging.getLogger(__name__)

# --- Model Loading ---
MODEL_PATH = Config.MODEL_PATH

model = None
detector = None
data_fetcher = None
wallet_scorer = None
batch_processor = None
batch_thread = None

def load_model():
    """Load the model and initialize components."""
    global model, detector, data_fetcher, wallet_scorer, batch_processor
    try:
        detector = joblib.load(MODEL_PATH)
        model = detector.model if hasattr(detector, 'model') else detector

        # If the sklearn model has feature_names_in_, log it for debugging
        try:
            if hasattr(model, 'feature_names_in_'):
                logger.info(f"Model expects features: {list(model.feature_names_in_)}")
        except Exception:
            pass

        # Initialize other components
        sepolia_rpc = os.getenv("SEPOLIA_RPC_URL") or "https://eth-sepolia.g.alchemy.com/v2/5bURjldvKPu4glB_tFxWt"
        data_fetcher = EthereumDataFetcher(sepolia_rpc)
        wallet_scorer = WalletScorer(detector)
        batch_processor = BatchProcessor(data_fetcher, detector, wallet_scorer)

        logger.info(f"✅ Model and components loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        logger.error(f"❌ Error: Model file not found at {MODEL_PATH}")
        logger.error("Please make sure you have trained the model first.")
        exit(1)
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        exit(1)

# ---------------------------
# Helpers
# ---------------------------
def get_risk_level(risk_score):
    if risk_score >= 0.8:
        return "Very High"
    elif risk_score >= 0.6:
        return "High"
    elif risk_score >= 0.4:
        return "Medium"
    elif risk_score >= 0.2:
        return "Low"
    else:
        return "Very Low"

def get_trust_level(trust_score):
    if trust_score >= 0.8:
        return "Very High"
    elif trust_score >= 0.6:
        return "High"
    elif trust_score >= 0.4:
        return "Medium"
    elif trust_score >= 0.2:
        return "Low"
    else:
        return "Very Low"

def extract_features_from_tx(raw_tx):
    """
    Defensive extraction of the 9 raw features your model expects.
    Returns a dict with keys:
      Transaction_Value, Transaction_Fees, Number_of_Inputs,
      Number_of_Outputs, Gas_Price, Wallet_Age_Days, Wallet_Balance,
      Transaction_Velocity, Exchange_Rate
    """
    defaults = {
        "Transaction_Value": 0.0,
        "Transaction_Fees": 0.0,
        "Number_of_Inputs": 0.0,
        "Number_of_Outputs": 0.0,
        "Gas_Price": 0.0,
        "Wallet_Age_Days": 0.0,
        "Wallet_Balance": 0.0,
        "Transaction_Velocity": 0.0,
        "Exchange_Rate": 0.0
    }
    if not raw_tx:
        return defaults.copy()

    f = {}
    for k in defaults.keys():
        if k in raw_tx:
            try:
                f[k] = float(raw_tx.get(k) or 0.0)
            except:
                f[k] = defaults[k]
        else:
            lk = k.lower()
            if lk in raw_tx:
                try:
                    f[k] = float(raw_tx.get(lk) or 0.0)
                except:
                    f[k] = defaults[k]
            else:
                # try common aliases
                alt = {
                    "Transaction_Value": ["value", "value_eth", "amount"],
                    "Transaction_Fees": ["fee", "fees", "transaction_fee"],
                    "Number_of_Inputs": ["num_inputs", "inputs"],
                    "Number_of_Outputs": ["num_outputs", "outputs"],
                    "Gas_Price": ["gasPrice", "gas_price"],
                    "Wallet_Age_Days": ["wallet_age_days", "wallet_age"],
                    "Wallet_Balance": ["wallet_balance", "balance"],
                    "Transaction_Velocity": ["tx_velocity", "transaction_velocity"],
                    "Exchange_Rate": ["exchange_rate", "eth_price"]
                }.get(k, [])
                found = False
                for a in alt:
                    if a in raw_tx:
                        try:
                            f[k] = float(raw_tx.get(a) or 0.0)
                            found = True
                            break
                        except:
                            continue
                if not found:
                    f[k] = defaults[k]
    return f

def build_complete_feature_vector(raw_features_dict):
    """
    Given the 9 raw features as dict, compute engineered features and return numeric vector of length 13.
    Order (important): raw 9 in this exact order, then engineered features:
      - value_to_fee_ratio
      - input_output_ratio
      - gas_efficiency
      - balance_turnover
    """
    feature_order = [
        'Transaction_Value', 'Transaction_Fees', 'Number_of_Inputs',
        'Number_of_Outputs', 'Gas_Price', 'Wallet_Age_Days',
        'Wallet_Balance', 'Transaction_Velocity', 'Exchange_Rate'
    ]
    arr = []
    for k in feature_order:
        arr.append(float(raw_features_dict.get(k, 0.0) or 0.0))

    tx_value = arr[0]
    tx_fees = arr[1]
    num_inputs = arr[2]
    num_outputs = arr[3]
    gas_price = arr[4]
    wallet_balance = arr[6]

    # engineered features (use small epsilon to avoid divide-by-zero)
    eps = 1e-8
    value_to_fee_ratio = tx_value / (tx_fees + eps)
    input_output_ratio = num_inputs / (num_outputs + eps)
    gas_efficiency = tx_value / (gas_price + eps)
    balance_turnover = tx_value / (wallet_balance + eps)

    arr_extended = arr + [
        value_to_fee_ratio, input_output_ratio, gas_efficiency, balance_turnover
    ]
    return arr_extended, feature_order + [
        "value_to_fee_ratio", "input_output_ratio", "gas_efficiency", "balance_turnover"
    ]

# ---------------------------
# POST /predict/transaction
# ---------------------------
@app.route('/predict/transaction', methods=['POST'])
def predict_transaction_risk():
    """
    Expect JSON: { "features": { ... } }
    We'll build the full 13-feature numeric vector and call the predictor.
    """
    if not detector:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        features_dict = data.get('features') or data.get('input_features') or data

        # Defensive extraction if user posted an entire transaction object
        if not isinstance(features_dict, dict):
            return jsonify({"error": "Invalid features payload"}), 400

        # Extract the 9 raw features (with fallbacks)
        raw_feats = extract_features_from_tx(features_dict)

        # Build 13-length numeric vector
        complete_vector, full_feature_names = build_complete_feature_vector(raw_feats)

        # Convert to numpy array for sklearn
        X = np.array([complete_vector], dtype=float)

        # Try to call detector in several ways:
        risk_probability = None
        # 1) If detector has a convenience method that accepts dicts
        if hasattr(detector, 'predict_fraud_risk'):
            try:
                # Try dict call first (some wrappers accept dict)
                risk_probability = float(detector.predict_fraud_risk(raw_feats))
            except Exception as e:
                logger.debug(f"detector.predict_fraud_risk(dict) failed: {e}. Will try numeric model call.")

        # 2) If risk_probability still None, try sklearn-style predict_proba on model (or detector)
        if risk_probability is None:
            try:
                # Prefer model if present
                m = model if model is not None else detector
                if hasattr(m, "predict_proba"):
                    prob = m.predict_proba(X)
                    # assume binary classifier, positive class is index 1
                    risk_probability = float(prob[0][1])
                elif hasattr(m, "predict"):
                    out = m.predict(X)
                    # If predict returns class label, try to map or cast
                    try:
                        risk_probability = float(out[0])
                    except:
                        # fallback 0/1 mapping
                        risk_probability = 1.0 if out[0] == 1 else 0.0
                else:
                    # last resort: call detector.predict_fraud_risk with the dict (again)
                    if hasattr(detector, 'predict_fraud_risk'):
                        risk_probability = float(detector.predict_fraud_risk(raw_feats))
            except Exception as e:
                logger.error(f"Error calling numeric model: {e}")
                return jsonify({"error": f"Model prediction failed: {str(e)}"}), 400

        # prepare response
        input_features_return = raw_feats.copy()
        # attach engineered features as well
        engineered = dict(zip(full_feature_names[-4:], complete_vector[-4:]))
        input_features_return.update(engineered)

        risk_probability = float(risk_probability)
        return jsonify({
            "risk_probability": risk_probability,
            "risk_level": get_risk_level(risk_probability),
            "is_high_risk": risk_probability > Config.HIGH_RISK_THRESHOLD,
            "input_features": input_features_return
        }), 200

    except Exception as e:
        logger.error(f"❌ Error during transaction prediction: {e}")
        return jsonify({"error": str(e)}), 400

# ---------------------------
# GET /transactions
# ---------------------------
@app.route('/transactions', methods=['GET'])
def get_recent_transactions():
    try:
        tx_list = []
        # allow client to request number via ?n=50
        try:
            n = int(request.args.get('n') or 50)
        except:
            n = 50

        if data_fetcher and hasattr(data_fetcher, 'get_latest_transactions'):
            tx_list = data_fetcher.get_latest_transactions(n) or []
        else:
            if wallet_scorer and hasattr(wallet_scorer, 'wallet_history'):
                hist = getattr(wallet_scorer, 'wallet_history', {}) or {}
                for addr, txs in hist.items():
                    for tx in txs:
                        tx_list.append(tx)

        normalized = []
        for tx in tx_list:
            txd = tx if isinstance(tx, dict) else getattr(tx, '__dict__', {})
            mapped = {
                "id": txd.get("hash") or txd.get("transaction_hash") or txd.get("tx_hash") or txd.get("id") or "",
                "from": txd.get("from_address") or txd.get("from") or txd.get("sender") or "",
                "to": txd.get("to_address") or txd.get("to") or txd.get("receiver") or "",
                "value": (txd.get("Transaction_Value") or txd.get("value") or txd.get("value_eth") or txd.get("amount") or "0 ETH"),
                "riskScore": float(txd.get("risk_probability") or txd.get("fraud_risk") or txd.get("risk_score") or 0.0),
                "walletScore": float(txd.get("wallet_score") or txd.get("trust_score") or 0.0),
                "timestamp": txd.get("timestamp") or txd.get("time") or txd.get("created_at") or "",
                "status": txd.get("status") or ("flagged" if float(txd.get("risk_probability", 0) or 0) > Config.HIGH_RISK_THRESHOLD else "processed")
            }
            normalized.append(mapped)

        return jsonify(normalized), 200
    except Exception as e:
        logger.error(f"❌ Error in /transactions endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------
# GET /transaction/<tx_hash>
# ---------------------------
@app.route('/transaction/<tx_hash>', methods=['GET'])
def lookup_transaction(tx_hash):
    try:
        raw_tx = None
        # 1) search wallet_history
        if wallet_scorer and hasattr(wallet_scorer, 'wallet_history'):
            history = getattr(wallet_scorer, 'wallet_history', {}) or {}
            found = False
            for addr, txs in history.items():
                for tx in txs:
                    tx_hash_val = (tx.get('hash') if isinstance(tx, dict) else getattr(tx, 'hash', None))
                    if tx_hash_val and str(tx_hash_val).lower() == tx_hash.lower():
                        raw_tx = tx if isinstance(tx, dict) else getattr(tx, '__dict__', tx)
                        found = True
                        break
                if found:
                    break

        # 2) try data_fetcher
        if not raw_tx and data_fetcher and hasattr(data_fetcher, 'get_transaction_details'):
            try:
                raw = data_fetcher.get_transaction_details(tx_hash)
                if raw:
                    raw_tx = raw
            except Exception as e:
                logger.debug(f"data_fetcher.get_transaction_details failed: {e}")

        if not raw_tx:
            return jsonify({"error": "Transaction not found"}), 404

        features = extract_features_from_tx(raw_tx)
        # build full vector for model if needed
        try:
            Xvec, names = build_complete_feature_vector(features)
        except:
            Xvec = None

        risk_probability = 0.0
        try:
            if detector and hasattr(detector, 'predict_fraud_risk'):
                risk_probability = float(detector.predict_fraud_risk(features))
            else:
                m = model if model is not None else detector
                if hasattr(m, "predict_proba") and Xvec is not None:
                    prob = m.predict_proba(np.array([Xvec], dtype=float))
                    risk_probability = float(prob[0][1])
                elif hasattr(m, "predict") and Xvec is not None:
                    out = m.predict(np.array([Xvec], dtype=float))
                    try:
                        risk_probability = float(out[0])
                    except:
                        risk_probability = 1.0 if out[0] == 1 else 0.0
        except Exception as e:
            logger.error(f"Error while predicting risk for tx {tx_hash}: {e}")
            risk_probability = 0.0

        from_addr = raw_tx.get("from_address") or raw_tx.get("from") or None
        to_addr = raw_tx.get("to_address") or raw_tx.get("to") or None
        from_score = None
        to_score = None
        try:
            if wallet_scorer and hasattr(wallet_scorer, 'get_wallet_score'):
                if from_addr:
                    from_score = wallet_scorer.get_wallet_score(from_addr)
                if to_addr:
                    to_score = wallet_scorer.get_wallet_score(to_addr)
        except Exception as e:
            logger.debug(f"Error getting wallet scores: {e}")

        result = {
            "transaction_hash": tx_hash,
            "raw_transaction": raw_tx,
            "features": features,
            "risk_probability": float(risk_probability),
            "risk_level": get_risk_level(float(risk_probability)),
            "is_high_risk": float(risk_probability) > Config.HIGH_RISK_THRESHOLD,
            "from_address": from_addr,
            "to_address": to_addr,
            "from_wallet_score": float(from_score) if from_score is not None else None,
            "to_wallet_score": float(to_score) if to_score is not None else None,
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"❌ Error in /transaction/{tx_hash} lookup: {e}")
        return jsonify({"error": str(e)}), 500

# --- Main Execution ---
if __name__ == '__main__':
    load_model()
    logger.info("🚀 Ethereum Fraud Detection API starting...")
    app.run(host='0.0.0.0', port=5000, debug=False)
