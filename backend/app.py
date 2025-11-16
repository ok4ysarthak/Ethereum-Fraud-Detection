# app.py
import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import logging
from datetime import datetime
import threading
import json # <-- ADD THIS
from data_fetcher import EthereumDataFetcher# <-- ADD THIS
import time
from flask_cors import CORS
from web3 import Web3 
from flask_socketio import SocketIO, emit
from wallet_updater import WalletScoreUpdater, INITIAL_SCORE
from config import Config
from db import init_db, SessionLocal
import crud
from models_db import Transaction, Wallet

# Tell Flask where the 'static' folder is
app = Flask(__name__, static_folder='static')
CORS(app)

# create socketio with the created app
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Setup Logging (after app created) ---
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# make sure app.logger uses the same config
app.logger.handlers = logging.getLogger().handlers
app.logger.setLevel(getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# --- Initialize DB inside application context (FIX) ---
# previously calling init_db() at import time triggered context issues.
with app.app_context():
    try:
        app.logger.info("📌 Creating database tables if not exist...")
        init_db()
        app.logger.info("✅ Database initialized!")
    except Exception as e:
        app.logger.exception("DB init failed: %s", e)
        # continue; endpoints will fail later if DB is unreachable

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
        except Exception:
            pass
        logger.info(f"✅ ML Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        logger.error(f"❌ Error: Model file not found at {MODEL_PATH}")
        # don't exit immediately; allow server to start for debugging
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")

# --- In-Memory Cache ---
PROCESSED_TX_CACHE = {} # Stores the FULL payload from oracle.py
CACHE_MAX_SIZE = 1000
cache_lock = threading.Lock()

# ---------------------------
# Helpers to map DB rows -> frontend JSON
# ---------------------------
def tx_to_front(tx_row):
    if not tx_row:
        return None
    ts = None
    try:
        ts = int(tx_row.timestamp.timestamp())
    except Exception:
        ts = None

    return {
        "id": tx_row.tx_hash,
        "transaction_hash": tx_row.tx_hash,
        "from": (tx_row.from_address or "").lower(),
        "to": (tx_row.to_address or "").lower(),
        "value": float(tx_row.amount_eth or 0.0),
        "riskScore": float(tx_row.risk_score or 0.0),
        "walletScore": float(tx_row.wallet_trust_score or 0.0),
        "timestamp": ts,
        "status": tx_row.status,
        "saved_to_chain": bool(tx_row.saved_to_chain or False),
        "onchain_record_txhash": tx_row.onchain_record_txhash,
        "raw_transaction": tx_row.raw_payload or (tx_row.metadata_json or {}),
    }

def wallet_to_front(wallet_row):
    if not wallet_row:
        return None
    return {
        "address": (wallet_row.address or "").lower(),
        "first_seen": int(wallet_row.first_seen.timestamp()) if wallet_row.first_seen else None,
        "last_seen": int(wallet_row.last_seen.timestamp()) if wallet_row.last_seen else None,
        "age_days": wallet_row.age_days,
        "score": float(wallet_row.trust_score or 0.0) if wallet_row.trust_score is not None else None,
        "avg_risk": float(wallet_row.avg_risk or 0.0) if wallet_row.avg_risk is not None else None,
        "labels": wallet_row.labels or {},
        "metadata": wallet_row.metadata_json or {},
    }

# small helpers
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

# feature extraction helpers (kept from your original)
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
# ROUTES (unchanged logic, just ensure DB calls are inside sessions)
# ---------------------------

@app.route('/transactions', methods=['GET'])
def api_get_transactions():
    try:
        limit = min(int(request.args.get('limit', 100)), 1000)
        offset = int(request.args.get('offset', 0))
        min_risk = request.args.get('min_risk', None)
        from_addr = request.args.get('from_address', None)
        to_addr = request.args.get('to_address', None)
        sort = request.args.get('sort', 'time')

        session = SessionLocal()
        q = session.query(Transaction)

        if min_risk is not None:
            try:
                min_risk_val = float(min_risk)
                q = q.filter(Transaction.risk_score >= min_risk_val)
            except:
                pass
        if from_addr:
            q = q.filter(Transaction.from_address == from_addr.lower())
        if to_addr:
            q = q.filter(Transaction.to_address == to_addr.lower())

        if sort == 'risk':
            q = q.order_by(Transaction.risk_score.desc())
        else:
            q = q.order_by(Transaction.timestamp.desc())

        rows = q.offset(offset).limit(limit).all()
        session.close()

        result = [tx_to_front(r) for r in rows]
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error in /transactions: %s", e)
        return jsonify({"error": "internal server error"}), 500

# Frontend static routes
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/activity')
def serve_activity_page():
    return send_from_directory(app.static_folder, 'activity.html')

@app.route('/live')
def serve_live_page():
    return send_from_directory(app.static_folder, 'live.html')

@app.route('/<path:path>')
def serve_static_files(path):
    try:
        return send_from_directory(app.static_folder, path)
    except:
        return send_from_directory(app.static_folder, 'index.html')

# POST /predict/transaction - keeps your existing behavior
@app.route('/predict/transaction', methods=['POST'])
def predict_transaction_risk():
    if not detector:
        return jsonify({"error": "Model not loaded"}), 500
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
            try:
                risk_probability = float(detector.predict_fraud_risk(raw_feats))
            except Exception as e:
                logger.debug(f"detector.predict_fraud_risk(dict) failed: {e}.")
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

# Wallets endpoints that read from the in-memory cache

@app.route('/wallets_db', methods=['GET'])
def get_wallets_db():
    try:
        limit = int(request.args.get('limit', 100))
        session = SessionLocal()
        from models_db import Wallet
        rows = session.query(Wallet).order_by(Wallet.trust_score.asc().nulls_last()).limit(limit).all()
        session.close()
        res = [{"address": r.address, "score": float(r.trust_score) if r.trust_score is not None else None,
                "first_seen": r.first_seen and int(r.first_seen.timestamp()), "last_seen": r.last_seen and int(r.last_seen.timestamp())}
               for r in rows]
        return jsonify(res), 200
    except Exception as e:
        app.logger.exception("Error fetching wallets from DB: %s", e)
        return jsonify([]), 500


@app.route('/wallets', methods=['GET'])
def get_wallets():
    try:
        limit = min(int(request.args.get('limit', 100)), 1000)
        sort_by = request.args.get('sort', 'score') # 'score' or 'last_seen'
        
        session = SessionLocal()
        q = session.query(Wallet)
        
        if sort_by == 'last_seen':
             q = q.order_by(Wallet.last_seen.desc().nulls_last())
        else:
             # Sort by score, lowest trust first (nulls last)
             q = q.order_by(Wallet.trust_score.asc().nulls_last())

        rows = q.limit(limit).all()
        session.close()
        
        # Use your existing wallet_to_front helper!
        result = [wallet_to_front(r) for r in rows]
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.exception("Error in /wallets: %s", e)
        return jsonify({"error": "internal server error"}), 500
    
@app.route('/wallet/<address>', methods=['GET'])
def get_wallet(address):
    try:
        norm_addr = crud._normalize_address(address)
        if not norm_addr:
             return jsonify({"error": "Invalid wallet address format"}), 400
             
        session = SessionLocal()
        
        # 1. Get the wallet data
        wallet_row = crud.get_wallet_by_address(session, norm_addr)
        
        if not wallet_row:
            session.close()
            return jsonify({"error": "Wallet not found in database"}), 404
            
        # 2. Get recent transaction history for this wallet
        history_rows = session.query(Transaction)\
            .filter(Transaction.from_address == norm_addr)\
            .order_by(Transaction.timestamp.desc())\
            .limit(20)\
            .all()
            
        session.close()
        
        # 3. Format the data
        wallet_data = wallet_to_front(wallet_row)
        
        history_list = [{
            "hash": tx.tx_hash,
            "value": tx.amount_eth,
            "risk_probability": tx.risk_score,
            "timestamp": int(tx.timestamp.timestamp()) if tx.timestamp else None
        } for tx in history_rows]

        # Combine for the final response
        wallet_data["trust_score"] = wallet_data.get("score")
        wallet_data["trust_level"] = get_trust_level(wallet_data.get("score", 0.0))
        wallet_data["history"] = history_list

        return jsonify(wallet_data), 200

    except Exception as e:
        app.logger.exception("Error in /wallet/<address>: %s", e)
        return jsonify({"error": "internal server error"}), 500
    
# GET /transactions (cache)
@app.route('/transactions', methods=['GET'])
def get_transactions():
    try:
        n = int(request.args.get('n') or request.args.get('limit') or 50)
        with cache_lock:
            all_processed_full = list(PROCESSED_TX_CACHE.values())
        all_processed_full.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        normalized_list = []
        for tx_data in all_processed_full[:n]:
            risk_score = float(tx_data.get("risk_probability", 0.0))
            wallet_score = tx_data.get("wallet_trust_score")
            try: wallet_score = float(wallet_score) if wallet_score is not None else None
            except: wallet_score = None
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

# POST /transactions (endpoint used by oracle.py)
@app.route('/transactions', methods=['POST'])
def add_transaction():
    try:
        data = request.get_json(force=True)
        if not data or not isinstance(data, dict):
            app.logger.warning("add_transaction: invalid payload")
            return jsonify({"error": "Invalid JSON payload"}), 400

        # normalize tx hash
        tx_hash = data.get("transaction_hash") or data.get("transactionHash") or data.get("hash")
        if not tx_hash:
            return jsonify({"error": "Missing 'transaction_hash'"}), 400
        tx_hash = tx_hash if tx_hash.startswith("0x") else "0x" + tx_hash

        # ensure timestamp present if not provided
        if "timestamp" not in data or not data.get("timestamp"):
            data["timestamp"] = int(time.time())

        # --- BEGIN NEW WALLET SCORE LOGIC ---
        # We will calculate the wallet score here, before saving
        
        new_wallet_score = None
        session = None
        try:
            from_addr = data.get("from_address") or data.get("from")
            tx_risk = data.get("risk_probability")
            tx_value = data.get("Transaction_Value") or data.get("value") or 0.0

            if from_addr and tx_risk is not None:
                session = SessionLocal()
                
                # 1. Get the wallet's current score
                current_wallet = crud.get_wallet_by_address(session, from_addr)
                
                if current_wallet and current_wallet.trust_score is not None:
                    current_score = float(current_wallet.trust_score)
                else:
                    current_score = INITIAL_SCORE # Use the initial score from wallet_updater
                
                # 2. Calculate the new score
                updater = WalletScoreUpdater()
                new_wallet_score = updater.calculate_new_score(
                    current_score=current_score,
                    tx_risk_probability=float(tx_risk),
                    tx_value_usd=float(tx_value) # Assuming value is in USD, adjust if not
                )
                
                # 3. Inject the new score into the data payload
                data["wallet_trust_score"] = new_wallet_score
                app.logger.info(f"Calculated new score for {from_addr}: {new_wallet_score:.4f}")

            else:
                app.logger.debug("Skipping wallet score update (missing from_addr or risk_score)")

        except Exception as e:
            app.logger.exception("Error during wallet score calculation: %s", e)
            # Don't fail the transaction, just log the error
        finally:
            if session:
                session.close()
        # --- END NEW WALLET SCORE LOGIC ---


        # Save to in-memory cache (so frontend still sees it immediately)
        with cache_lock:
            PROCESSED_TX_CACHE[tx_hash] = data
            if len(PROCESSED_TX_CACHE) > CACHE_MAX_SIZE:
                sorted_items = sorted(PROCESSED_TX_CACHE.items(), key=lambda item: item[1].get("timestamp", 0))
                num_to_remove = len(PROCESSED_TX_CACHE) - CACHE_MAX_SIZE
                for i in range(num_to_remove):
                    del PROCESSED_TX_CACHE[sorted_items[i][0]]

        # Broadcast to websocket clients (unchanged)
        try:
            risk_score = float(data.get("risk_probability", 0.0))
        except Exception:
            risk_score = 0.0
        
        # Use the score we just calculated for the broadcast
        wallet_score_for_broadcast = new_wallet_score if new_wallet_score is not None else data.get("wallet_trust_score")
        
        try:
            wallet_score = float(wallet_score_for_broadcast) if wallet_score_for_broadcast is not None else None
        except Exception:
            wallet_score = None

        mapped_tx_for_broadcast = {
            "id": tx_hash,
            "from": data.get("from_address") or data.get("from") or "",
            "to": data.get("to_address") or data.get("to") or "",
            "value": data.get("Transaction_Value") or data.get("value") or 0,
            "riskScore": risk_score,
            "walletScore": wallet_score, # Use the new score
            "timestamp": data.get("timestamp"),
            "status": ("flagged" if risk_score > Config.HIGH_RISK_THRESHOLD else "processed")
        }
        socketio.emit('new_transaction', mapped_tx_for_broadcast)

        # Persist to Postgres (DEFINITE logging + fail-safe)
        # This will now work, because crud.create_transaction will find
        # the "wallet_trust_score" we just added to the 'data' dict.
        try:
            session = SessionLocal()
            tx_row = None
            try:
                tx_row = crud.create_transaction(session, data)
                session.commit() # The commit is now handled in crud.py, but this is a safeguard
                app.logger.info(f"Persisted TX to DB: {getattr(tx_row, 'tx_hash', tx_hash)}")
            except Exception as e:
                session.rollback()
                app.logger.exception("DB persistence failed inside create_transaction: %s", e)
                # do not raise; still return 201 so oracle doesn't resend, but log heavily
            finally:
                session.close()
        except Exception as e:
            app.logger.exception("Failed to create DB session or persist tx: %s", e)

        return jsonify({"status": "ok", "cached_hash": tx_hash}), 201

    except Exception as e:
        app.logger.exception("Unexpected error in add_transaction: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/admin/backfill_cache_to_db', methods=['POST'])
def backfill_cache_to_db():
    """
    One-off: persist all entries currently in PROCESSED_TX_CACHE to DB.
    Call via: curl -X POST http://127.0.0.1:5000/admin/backfill_cache_to_db
    """
    try:
        count = 0
        with cache_lock:
            items = list(PROCESSED_TX_CACHE.items())
        session = SessionLocal()
        try:
            for tx_hash, data in items:
                try:
                    crud.create_transaction(session, data)
                    count += 1
                except Exception as e:
                    app.logger.exception("Backfill: failed to persist %s: %s", tx_hash, e)
            session.commit()
        except Exception as e:
            session.rollback()
            app.logger.exception("Backfill: error during commit: %s", e)
            return jsonify({"status": "error", "detail": str(e)}), 500
        finally:
            session.close()
        return jsonify({"status": "ok", "persisted": count}), 200
    except Exception as e:
        app.logger.exception("Backfill: unexpected error: %s", e)
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route('/transactions/<tx_hash>/onchain', methods=['POST'])
def mark_transaction_onchain(tx_hash):
    try:
        data = request.get_json(force=True)
        onchain_tx = data.get("onchain_tx_hash") or data.get("onchain_txhash") or data.get("onchain")
        if not onchain_tx:
            return jsonify({"error": "missing onchain_tx_hash"}), 400
        try:
            with app.app_context():
                session = SessionLocal()
                updated = crud.mark_tx_onchain(session, tx_hash, onchain_tx)
                session.close()
        except Exception as e:
            logger.exception("DB update failed: %s", e)
            return jsonify({"error": "db update failed"}), 500
        if not updated:
            return jsonify({"error": "tx not found"}), 404
        return jsonify({"status": "ok", "tx_hash": tx_hash, "onchain_tx": onchain_tx}), 200
    except Exception as e:
        logger.exception("Error in mark_transaction_onchain: %s", e)
        return jsonify({"error": str(e)}), 500

# @app.route('/transaction/<tx_hash>', methods=['GET'])
# def lookup_transaction(tx_hash):
#     try:
#         norm_hash = crud._normalize_address(tx_hash) # Use the crud helper
#         if not norm_hash:
#             return jsonify({"error": "Invalid tx_hash format"}), 400

#         tx_data_formatted = None
        
#         # 1. Check cache first for speed
#         with cache_lock:
#             cached_data = PROCESSED_TX_CACHE.get(norm_hash)
        
#         if cached_data:
#             app.logger.debug(f"TX {norm_hash} found in cache.")
#             # Format the cached data (which is a raw dict)
#             risk_score = float(cached_data.get("risk_probability", 0.0))
#             tx_data_formatted = {
#                 "transaction_hash": cached_data.get("transaction_hash"),
#                 "raw_transaction": cached_data.get("raw_details") or cached_data,
#                 "features": cached_data.get("raw_details") or cached_data,
#                 "risk_probability": risk_score,
#                 "risk_level": get_risk_level(risk_score),
#                 "is_high_risk": risk_score > Config.HIGH_RISK_THRESHOLD,
#                 "from_address": cached_data.get("from_address"),
#                 "to_address": cached_data.get("to_address"),
#                 "from_wallet_score": cached_data.get("wallet_trust_score"),
#                 "to_wallet_score": None, # We don't track this
#                 "timestamp": cached_data.get("timestamp"),
#                 "value": cached_data.get("Transaction_Value") or cached_data.get("value")
#             }

#         # 2. If not in cache, check the database
#         if not tx_data_formatted:
#             app.logger.debug(f"TX {norm_hash} not in cache, checking DB.")
#             session = SessionLocal()
#             try:
#                 tx_row = session.query(Transaction).filter(Transaction.tx_hash == norm_hash).one_or_none()
                
#                 if tx_row:
#                     app.logger.debug(f"TX {norm_hash} found in DB.")
#                     # Use our existing tx_to_front helper
#                     db_data = tx_to_front(tx_row)
                    
#                     # Format the database data
#                     risk_score = float(db_data.get("riskScore", 0.0))
#                     tx_data_formatted = {
#                         "transaction_hash": db_data.get("transaction_hash"),
#                         "raw_transaction": db_data.get("raw_transaction"), # This has the payload
#                         "features": db_data.get("raw_transaction"),
#                         "risk_probability": risk_score,
#                         "risk_level": get_risk_level(risk_score),
#                         "is_high_risk": risk_score > Config.HIGH_RISK_THRESHOLD,
#                         "from_address": db_data.get("from"),
#                         "to_address": db_data.get("to"),
#                         "from_wallet_score": db_data.get("walletScore"),
#                         "to_wallet_score": None,
#                         "timestamp": db_data.get("timestamp"),
#                         "value": db_data.get("value")
#                     }
#                 else:
#                     app.logger.warning(f"TX {norm_hash} not found in cache OR DB.")
#             except Exception as e:
#                 app.logger.exception(f"DB lookup failed for {norm_hash}: {e}")
#             finally:
#                 session.close()

#         # 3. If still not found, 404
#         if not tx_data_formatted:
#             return jsonify({"error": "Transaction not found"}), 404

#         # 4. Return the formatted data
#         return jsonify(tx_data_formatted), 200

#     except Exception as e:
#         app.logger.exception(f"❌ Error in /transaction/{tx_hash} lookup: {e}")
#         return jsonify({"error": str(e)}), 500

@app.route('/transaction/<tx_hash>', methods=['GET'])
def lookup_transaction(tx_hash):
    try:
        norm_hash = crud._normalize_address(tx_hash) # Use the crud helper
        if not norm_hash:
            return jsonify({"error": "Invalid tx_hash format"}), 400

        tx_data_formatted = None

        # 1. Check cache first for speed
        with cache_lock:
            cached_data = PROCESSED_TX_CACHE.get(norm_hash)

        if cached_data:
            app.logger.debug(f"TX {norm_hash} found in cache.")
            risk_score = float(cached_data.get("risk_probability", 0.0))
            tx_data_formatted = {
                "transaction_hash": cached_data.get("transaction_hash"),
                "raw_transaction": cached_data.get("raw_details") or cached_data,
                "features": cached_data.get("raw_details") or cached_data,
                "risk_probability": risk_score,
                "risk_level": get_risk_level(risk_score),
                "is_high_risk": risk_score > Config.HIGH_RISK_THRESHOLD,
                "from_address": cached_data.get("from_address"),
                "to_address": cached_data.get("to_address"),
                "from_wallet_score": cached_data.get("wallet_trust_score"),
                "to_wallet_score": None,
                "timestamp": cached_data.get("timestamp"),
                "value": cached_data.get("Transaction_Value") or cached_data.get("value")
            }

        # 2. If not in cache, check the database
        if not tx_data_formatted:
            app.logger.debug(f"TX {norm_hash} not in cache, checking DB.")
            session = SessionLocal()
            try:
                tx_row = session.query(Transaction).filter(Transaction.tx_hash == norm_hash).one_or_none()

                if tx_row:
                    app.logger.debug(f"TX {norm_hash} found in DB.")
                    db_data = tx_to_front(tx_row)
                    risk_score = float(db_data.get("riskScore", 0.0))
                    tx_data_formatted = {
                        "transaction_hash": db_data.get("transaction_hash"),
                        "raw_transaction": db_data.get("raw_transaction"),
                        "features": db_data.get("raw_transaction"),
                        "risk_probability": risk_score,
                        "risk_level": get_risk_level(risk_score),
                        "is_high_risk": risk_score > Config.HIGH_RISK_THRESHOLD,
                        "from_address": db_data.get("from"),
                        "to_address": db_data.get("to"),
                        "from_wallet_score": db_data.get("walletScore"),
                        "to_wallet_score": None,
                        "timestamp": db_data.get("timestamp"),
                        "value": db_data.get("value")
                    }
            except Exception as e:
                app.logger.exception(f"DB lookup failed for {norm_hash}: {e}")
            finally:
                session.close()

        # 3. --- NEW: If not found, attempt live fetch from blockchain ---
        if not tx_data_formatted:
            app.logger.debug(f"TX {norm_hash} not in local DB, attempting live fetch.")
            try:
                data_fetcher = EthereumDataFetcher()
                tx = data_fetcher.w3.eth.get_transaction(norm_hash)

                if tx:
                    app.logger.debug(f"TX {norm_hash} found live on-chain.")
                    block = data_fetcher.w3.eth.get_block(tx.blockNumber)
                    tx_value_eth = Web3.from_wei(tx.value, 'ether')

                    # Serialize the 'AttributeDict' from web3 to a standard dict
                    raw_payload = json.loads(Web3.to_json(tx))
                    tx_timestamp = int(time.time()) # Default to now for pending tx
                    
                    if tx.blockNumber is None:
                        # Transaction is PENDING
                        app.logger.warning(f"TX {norm_hash} is PENDING (blockNumber is None).")
                        raw_payload["message"] = "This transaction is PENDING and not yet in a block. Etherscan may not find it yet."
                    else:
                        # Transaction is CONFIRMED
                        block = data_fetcher.w3.eth.get_block(tx.blockNumber)
                        if block:
                            tx_timestamp = block.timestamp
                        raw_payload["message"] = "Fetched live from blockchain (confirmed). Risk scores are not available for this transaction."
                    tx_data_formatted = {
                        "transaction_hash": tx.hash.hex(),
                        "raw_transaction": raw_payload,
                        "features": raw_payload,
                        "risk_probability": None, # No risk score, not processed
                        "risk_level": "Unknown",
                        "is_high_risk": False,
                        "from_address": tx["from"],
                        "to_address": tx.to,
                        "from_wallet_score": None, # No wallet score
                        "to_wallet_score": None,
                        "timestamp": tx_timestamp,
                        "value": float(tx_value_eth)
                    }
                else:
                    app.logger.warning(f"TX {norm_hash} not found anywhere (cache, DB, or live).")
            except Exception as e:
                app.logger.exception(f"Live fetch failed for {norm_hash}: {e}")
                # Don't error out, just means we couldn't find it
                pass

        # 4. If still not found, 404
        if not tx_data_formatted:
            return jsonify({"error": "Transaction not found"}), 404

        # 5. Return the formatted data
        return jsonify(tx_data_formatted), 200

    except Exception as e:
        app.logger.exception(f"❌ Error in /transaction/{tx_hash} lookup: {e}")
        return jsonify({"error": str(e)}), 500

# Socket.IO events
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
    # Run with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
