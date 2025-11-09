import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import logging
from datetime import datetime
import threading
import time
from flask_cors import CORS
from web3 import Web3 
from flask_socketio import SocketIO, emit
from config import Config

# Tell Flask where the 'static' folder is
app = Flask(__name__, static_folder='static') 
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

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
        try: handler.stream.reconfigure(encoding='utf-8')
        except Exception: pass
logger = logging.getLogger(__name__)

# --- Model Loading (for /predict/transaction) ---
MODEL_PATH = Config.MODEL_PATH
model = None
detector = None

def load_model():
    """Load *only* the ML model for the /predict/transaction endpoint."""
    global model, detector
    try:
        detector = joblib.load(MODEL_PATH)
        model = detector.model if hasattr(detector, 'model') else detector
        try:
            if hasattr(model, 'feature_names_in_'):
                logger.info(f"Model expects features: {list(model.feature_names_in_)}")
        except Exception: pass
        logger.info(f"✅ ML Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        logger.error(f"❌ Error: Model file not found at {MODEL_PATH}")
        exit(1)
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        exit(1)

# --- In-Memory Cache ---
PROCESSED_TX_CACHE = {} # Stores the FULL payload from oracle.py
CACHE_MAX_SIZE = 1000
cache_lock = threading.Lock()

# ---------------------------
# Helpers
# ---------------------------
def get_risk_level(risk_score):
    if risk_score >= 0.8: return "Very High"
    elif risk_score >= 0.6: return "High"
    elif risk_score >= 0.4: return "Medium"
    elif risk_score >= 0.2: return "Low"
    else: return "Very Low"

def get_trust_level(trust_score):
    if trust_score >= 0.8: return "Very High"
    elif trust_score >= 0.6: return "High"
    elif trust_score >= 0.4: return "Medium"
    elif trust_score >= 0.2: return "Low"
    else: return "Very Low"

def extract_features_from_tx(raw_tx):
    defaults = {
        "Transaction_Value": 0.0, "Transaction_Fees": 0.0,
        "Number_of_Inputs": 0.0, "Number_of_Outputs": 0.0,
        "Gas_Price": 0.0, "Wallet_Age_Days": 0.0, "Wallet_Balance": 0.0,
        "Transaction_Velocity": 0.0, "Exchange_Rate": 0.0
    }
    if not raw_tx: return defaults.copy()
    f = {}
    for k in defaults.keys():
        if k in raw_tx:
            try: f[k] = float(raw_tx.get(k) or 0.0)
            except: f[k] = defaults[k]
        else:
            lk = k.lower()
            if lk in raw_tx:
                try: f[k] = float(raw_tx.get(lk) or 0.0)
                except: f[k] = defaults[k]
            else:
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
                        except: continue
                if not found: f[k] = defaults[k]
    return f

def build_complete_feature_vector(raw_features_dict):
    feature_order = [
        'Transaction_Value', 'Transaction_Fees', 'Number_of_Inputs',
        'Number_of_Outputs', 'Gas_Price', 'Wallet_Age_Days',
        'Wallet_Balance', 'Transaction_Velocity', 'Exchange_Rate'
    ]
    arr = [float(raw_features_dict.get(k, 0.0) or 0.0) for k in feature_order]
    tx_value = arr[0]; tx_fees = arr[1]; num_inputs = arr[2]
    num_outputs = arr[3]; gas_price = arr[4]; wallet_balance = arr[6]
    eps = 1e-8
    value_to_fee_ratio = tx_value / (tx_fees + eps)
    input_output_ratio = num_inputs / (num_outputs + eps)
    gas_efficiency = tx_value / (gas_price + eps)
    balance_turnover = tx_value / (wallet_balance + eps)
    arr_extended = arr + [value_to_fee_ratio, input_output_ratio, gas_efficiency, balance_turnover]
    return arr_extended, feature_order + ["value_to_fee_ratio", "input_output_ratio", "gas_efficiency", "balance_turnover"]

# ---------------------------
# --- Frontend Static Routes ---
# ---------------------------
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    try:
        return send_from_directory(app.static_folder, path)
    except:
        return send_from_directory(app.static_folder, 'index.html')

# ---------------------------
# API: POST /predict/transaction
# (This is used by your oracle.py)
# ---------------------------
@app.route('/predict/transaction', methods=['POST'])
def predict_transaction_risk():
    if not detector: return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.get_json(force=True)
        features_dict = data.get('features') or data.get('input_features') or data
        if not isinstance(features_dict, dict):
            return jsonify({"error": "Invalid features payload"}), 400
        raw_feats = extract_features_from_tx(features_dict)
        complete_vector, full_feature_names = build_complete_feature_vector(raw_feats)
        X = np.array([complete_vector], dtype=float)
        risk_probability = None
        if hasattr(detector, 'predict_fraud_risk'):
            try: risk_probability = float(detector.predict_fraud_risk(raw_feats))
            except Exception as e: logger.debug(f"detector.predict_fraud_risk(dict) failed: {e}.")
        if risk_probability is None:
            try:
                m = model if model is not None else detector
                if hasattr(m, "predict_proba"):
                    prob = m.predict_proba(X)
                    risk_probability = float(prob[0][1])
                elif hasattr(m, "predict"):
                    out = m.predict(X)
                    try: risk_probability = float(out[0])
                    except: risk_probability = 1.0 if out[0] == 1 else 0.0
                elif hasattr(detector, 'predict_fraud_risk'):
                    risk_probability = float(detector.predict_fraud_risk(raw_feats))
            except Exception as e:
                logger.error(f"Error calling numeric model: {e}")
                return jsonify({"error": f"Model prediction failed: {str(e)}"}), 400
        input_features_return = raw_feats.copy()
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
# API: GET /wallets
# (Reads from the cache populated by the oracle)
# ---------------------------
@app.route('/wallets', methods=['GET'])
def get_wallets():
    with cache_lock:
        wallet_scores = {}
        # Iterate in reverse chronological order
        for tx in sorted(PROCESSED_TX_CACHE.values(), key=lambda x: x.get("timestamp", 0), reverse=True):
            from_addr = tx.get("from_address")
            if from_addr and from_addr not in wallet_scores:
                wallet_scores[from_addr] = tx.get("wallet_trust_score") 
    
    limit = int(request.args.get('limit', 100))
    wallet_list = [{"address": addr, "score": score} for addr, score in wallet_scores.items() if score is not None]
    wallet_list.sort(key=lambda x: x.get('score', 1.0)) # Sort by score, lowest trust first
    return jsonify(wallet_list[:limit]), 200

# ---------------------------
# API: GET /wallet/<address>
# (Reads from the cache populated by the oracle)
# ---------------------------
@app.route('/wallet/<address>', methods=['GET'])
def get_wallet(address):
    latest_score = None
    history = []
    try:
        checksum_addr = Web3.to_checksum_address(address)
    except:
        return jsonify({"error": "Invalid wallet address format"}), 400
    with cache_lock:
        # Find all txs from this wallet and its latest score
        for tx in sorted(PROCESSED_TX_CACHE.values(), key=lambda x: x.get("timestamp", 0), reverse=True):
            from_addr = tx.get("from_address")
            if from_addr and Web3.to_checksum_address(from_addr) == checksum_addr:
                if latest_score is None:
                    latest_score = tx.get("wallet_trust_score") # Get latest score
                
                history.append({
                    "hash": tx.get("transaction_hash"),
                    "value": tx.get("Transaction_Value"),
                    "risk_probability": tx.get("risk_probability"),
                    "timestamp": tx.get("timestamp")
                })
    
    if latest_score is None:
        return jsonify({"error": "Wallet not found in cache"}), 404

    return jsonify({
        "address": checksum_addr,
        "trust_score": float(latest_score),
        "trust_level": get_trust_level(float(latest_score)),
        "history": history[:20] # Return 20 most recent
    }), 200

# ---------------------------
# API: GET /transactions
# (Reads from the cache populated by the oracle)
# ---------------------------
@app.route('/transactions', methods=['GET'])
def get_transactions():
    try:
        n = int(request.args.get('n') or request.args.get('limit') or 50)
        
        with cache_lock:
            all_processed_full = list(PROCESSED_TX_CACHE.values())
            
        all_processed_full.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        # --- FIX: Normalize on the fly ---
        # The cache contains the FULL payload, but the tables need the SIMPLE payload.
        normalized_list = []
        for tx_data in all_processed_full[:n]:
            risk_score = float(tx_data.get("risk_probability", 0.0))
            wallet_score = float(tx_data.get("wallet_trust_score", 0.0))
            mapped = {
                "id": tx_data.get("transaction_hash") or "",
                "from": tx_data.get("from_address") or "",
                "to": tx_data.get("to_address") or "",
                "value": tx_data.get("Transaction_Value") or 0,
                "riskScore": risk_score,
                "walletScore": wallet_score,
                "timestamp": tx_data.get("timestamp") or "",
                "status": ("flagged" if risk_score > Config.HIGH_RISK_THRESHOLD else "processed")
            }
            normalized_list.append(mapped)
        return jsonify(normalized_list), 200
    except Exception as e:
        logger.error(f"❌ Error in /transactions GET endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------
# --- FIX: POST /transactions ---
# (This is the endpoint your oracle.py is trying to hit)
# ---------------------------
@app.route('/transactions', methods=['POST'])
def add_transaction():
    """
    Receives an enriched transaction payload from oracle.py,
    caches it, and broadcasts a simplified version to all connected clients.
    """
    try:
        data = request.get_json(force=True) # This is the full, rich payload from oracle.py
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400
        tx_hash = data.get("transaction_hash")
        if not tx_hash:
            return jsonify({"error": "Missing 'transaction_hash'"}), 400

        # --- FIX: Store the FULL data payload ---
        with cache_lock:
            PROCESSED_TX_CACHE[tx_hash] = data # Store the whole thing
            
            # Prune cache if it gets too big
            if len(PROCESSED_TX_CACHE) > CACHE_MAX_SIZE:
                sorted_items = sorted(PROCESSED_TX_CACHE.items(), key=lambda item: item[1].get("timestamp", 0))
                num_to_remove = len(PROCESSED_TX_CACHE) - CACHE_MAX_SIZE
                for i in range(num_to_remove):
                    del PROCESSED_TX_CACHE[sorted_items[i][0]]
        
        # --- Create the simple version ONLY for the broadcast ---
        risk_score = float(data.get("risk_probability", 0.0))
        wallet_score = float(data.get("wallet_trust_score", 0.0))
        mapped_tx_for_broadcast = {
            "id": data.get("transaction_hash") or "",
            "from": data.get("from_address") or "",
            "to": data.get("to_address") or "",
            "value": data.get("Transaction_Value") or 0,
            "riskScore": risk_score,
            "walletScore": wallet_score,
            "timestamp": data.get("timestamp") or int(time.time()),
            "status": ("flagged" if risk_score > Config.HIGH_RISK_THRESHOLD else "processed")
        }
        
        socketio.emit('new_transaction', mapped_tx_for_broadcast) # Emit the simple version
        
        logger.info(f"Cached full oracle payload for: {tx_hash}")
        return jsonify({"status": "ok", "cached_hash": tx_hash}), 201
    
    except Exception as e:
        logger.error(f"❌ Error in /transactions POST endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------
# API: GET /transaction/<tx_hash>
# (Reads from the cache populated by the oracle)
# ---------------------------
@app.route('/transaction/<tx_hash>', methods=['GET'])
def lookup_transaction(tx_hash):
    try:
        norm_hash = tx_hash
        if not tx_hash.startswith("0x"):
            norm_hash = "0x" + tx_hash
        
        with cache_lock:
            tx_data = PROCESSED_TX_CACHE.get(norm_hash)

        if not tx_data:
             # Try to find it even if the case is different
            for key, val in PROCESSED_TX_CACHE.items():
                if key.lower() == norm_hash.lower():
                    tx_data = val
                    break
            if not tx_data:
                return jsonify({"error": "Transaction not found in cache"}), 404

        # --- FIX: Return the FULL data payload ---
        # The frontend modal will look for 'raw_transaction'
        risk_score = float(tx_data.get("risk_probability", 0.0))
        result = {
            "transaction_hash": tx_data.get("transaction_hash"),
            "raw_transaction": tx_data.get("raw_details") or tx_data, # This is the full object
            "features": tx_data.get("raw_details") or tx_data, # Also put it here
            
            "risk_probability": risk_score,
            "risk_level": get_risk_level(risk_score),
            "is_high_risk": risk_score > Config.HIGH_RISK_THRESHOLD,
            "from_address": tx_data.get("from_address"),
            "to_address": tx_data.get("to_address"),
            "from_wallet_score": tx_data.get("wallet_trust_score"),
            "to_wallet_score": None, # We don't track receiver score in this model
            "timestamp": tx_data.get("timestamp"),
            "value": tx_data.get("Transaction_Value")
        }
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"❌ Error in /transaction/{tx_hash} lookup: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------
# --- Socket.IO Events ---
# ---------------------------
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

# --- Main Execution ---
if __name__ == '__main__':
    load_model()
    
    logger.info("🚀 Ethereum Fraud Detection API starting...")
    logger.info("This server *receives* data from oracle.py.")
    logger.info("Run 'python oracle.py' in a separate terminal to start data processing.")
    
    # --- Run with SocketIO ---
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)