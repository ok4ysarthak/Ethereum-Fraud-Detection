# backend/oracle.py
import os
import json
import time
import logging
import requests
from web3 import Web3
from dotenv import load_dotenv
# Local imports
from wallet_updater import WalletScoreUpdater
from data_fetcher import EthereumDataFetcher

# Load env
load_dotenv(dotenv_path='../.env')
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL")
ORACLE_PRIVATE_KEY = os.getenv("ORACLE_PRIVATE_KEY")
API_URL = os.getenv("ML_API_URL", "http://127.0.0.1:5000")
# We will prefer GET /transaction/<hash> first, then fallback to POST /predict/transaction

POLL_INTERVAL_SECONDS = int(os.getenv("ORACLE_POLL_INTERVAL", "15"))
TXS_PER_POLL = int(os.getenv("ORACLE_TXS_PER_POLL", "20"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if not all([CONTRACT_ADDRESS, SEPOLIA_RPC_URL, ORACLE_PRIVATE_KEY]):
    raise EnvironmentError("Missing CONTRACT_ADDRESS, SEPOLIA_RPC_URL or ORACLE_PRIVATE_KEY in .env")

# Connect to web3
logger.info("Connecting to Sepolia RPC...")
w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC_URL))
if not w3.is_connected():
    raise ConnectionError("Failed to connect to Sepolia RPC")

oracle_account = w3.eth.account.from_key(ORACLE_PRIVATE_KEY)
w3.eth.default_account = oracle_account.address
logger.info(f"Oracle signer: {oracle_account.address}")

# Load contract ABI
artifact_path = '../artifacts/contracts/TrustScore.sol/FraudDetection.json'
try:
    with open(artifact_path, 'r') as f:
        cj = json.load(f)
        contract_abi = cj.get('abi', [])
except FileNotFoundError:
    logger.error(f"Artifact not found at {artifact_path}")
    raise

fraud_detection_contract = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=contract_abi)
logger.info(f"Connected to contract at: {CONTRACT_ADDRESS}")

# Helpers
score_updater = WalletScoreUpdater()
fetcher = EthereumDataFetcher(SEPOLIA_RPC_URL)

processed_tx_hashes = set()

def try_backend_lookup(tx_hash_hex):
    """
    Try several GET variants for /transaction/<tx_hash>:
      - tx_hash_hex (as-is)
      - '0x' + tx_hash_hex (if missing)
      - URL encoded
    Return parsed JSON on success or None on failure.
    """
    import urllib.parse

    base = API_URL.rstrip('/')
    variants = [tx_hash_hex]

    # add 0x prefix if missing
    if not tx_hash_hex.startswith('0x'):
        variants.append('0x' + tx_hash_hex)

    # add urlencoded versions (defensive)
    for v in list(variants):
        variants.append(urllib.parse.quote_plus(v))

    # ensure unique while preserving order
    seen = set()
    variants_unique = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            variants_unique.append(v)

    for v in variants_unique:
        url = f"{base}/transaction/{v}"
        try:
            logger.debug(f"Trying backend GET: {url}")
            r = requests.get(url, timeout=10)
            # log status for debugging
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception as e:
                    logger.error(f"GET {url} returned 200 but JSON parse failed: {e}")
                    return None
            else:
                logger.debug(f"GET {url} returned {r.status_code}: {r.text[:300]}")
        except Exception as e:
            logger.debug(f"GET {url} request failed: {e}")

    # none of the GET variants succeeded
    return None

def post_predict_transaction(features):
    """
    Fallback to POST /predict/transaction if direct lookup not available.
    Expects backend to accept {"features": {...}} and return {"risk_probability":...}
    """
    try:
        url = f"{API_URL.rstrip('/')}/predict/transaction"
        r = requests.post(url, json={"features": features}, timeout=12)
        if r.status_code == 200:
            return r.json()
        else:
            logger.error(f"ML API error {r.status_code}: {r.text}")
            return None
    except Exception as e:
        logger.exception("Error calling ML API POST /predict/transaction")
        return None

def call_ml_for_tx(tx_hash_hex, features):
    """
    Try GET transaction lookup first; else POST features.
    Return risk_probability (float) or None.
    """
    # 1. Try backend lookup which may already compute risk (preferred)
    got = try_backend_lookup(tx_hash_hex)
    if got and ("risk_probability" in got or "risk_score" in got or "risk" in got):
        # normalize
        val = got.get("risk_probability") or got.get("risk_score") or got.get("risk") or 0.0
        try:
            return float(val)
        except:
            return None

    # 2. Fallback to POST /predict/transaction
    res = post_predict_transaction(features)
    if res and ("risk_probability" in res or "risk_score" in res or "risk" in res):
        val = res.get("risk_probability") or res.get("risk_score") or res.get("risk") or 0.0
        try:
            return float(val)
        except:
            return None

    return None

def send_update_tx(tx_hash_bytes32, from_addr, to_addr, value_eth, risk_probability):
    try:
        risk_int = int(risk_probability * 100)
        nonce = w3.eth.get_transaction_count(oracle_account.address)
        tx = fraud_detection_contract.functions.updateTransactionRisk(
            tx_hash_bytes32,
            from_addr,
            to_addr,
            int(value_eth * 10**18),
            risk_int
        ).build_transaction({
            'from': oracle_account.address,
            'nonce': nonce,
            'gas': 300000,
            'gasPrice': w3.eth.gas_price
        })
        signed = w3.eth.account.sign_transaction(tx, ORACLE_PRIVATE_KEY)
        txh = w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info(f"Sent updateTransactionRisk tx: {txh.hex()}")
        receipt = w3.eth.wait_for_transaction_receipt(txh, timeout=180)
        logger.info(f"updateTransactionRisk mined in block {receipt.blockNumber}")
        return txh.hex()
    except Exception as e:
        logger.exception(f"Failed to send updateTransactionRisk tx: {e}")
        return None

def send_update_wallet_score(wallet_address, new_score_float):
    try:
        score_int = int(new_score_float * 100)
        nonce = w3.eth.get_transaction_count(oracle_account.address)
        tx = fraud_detection_contract.functions.updateWalletScore(
            wallet_address,
            score_int
        ).build_transaction({
            'from': oracle_account.address,
            'nonce': nonce,
            'gas': 200000,
            'gasPrice': w3.eth.gas_price
        })
        signed = w3.eth.account.sign_transaction(tx, ORACLE_PRIVATE_KEY)
        txh = w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info(f"Sent updateWalletScore tx: {txh.hex()}")
        receipt = w3.eth.wait_for_transaction_receipt(txh, timeout=180)
        logger.info(f"updateWalletScore mined in block {receipt.blockNumber}")
        return txh.hex()
    except Exception as e:
        logger.exception(f"Failed to send updateWalletScore tx: {e}")
        return None

def normalize_tx_dict(tx_hash_hex, details):
    """
    Build a normalized tx dict that includes:
      - hash (hex)
      - from_address/from
      - to_address/to
      - Transaction_Value, Transaction_Fees, Gas_Price, etc. (from details)
    details is expected to be the dict returned by data_fetcher.get_transaction_details
    """
    out = {}
    out['hash'] = tx_hash_hex
    if not details:
        return out
    # prefer strongly named keys from fetcher
    out['from_address'] = details.get('from_address') or details.get('from') or details.get('sender')
    out['to_address'] = details.get('to_address') or details.get('to') or details.get('receiver')
    # bring feature names in
    for k in ['Transaction_Value','Transaction_Fees','Number_of_Inputs','Number_of_Outputs','Gas_Price','Wallet_Age_Days','Wallet_Balance','Transaction_Velocity','Exchange_Rate','timestamp']:
        if k in details:
            out[k] = details[k]
    # fallback for some alternatives
    out['value'] = details.get('Transaction_Value') or details.get('value') or details.get('value_eth') or 0.0
    return out

def process_transaction(tx_hash_hex):
    """
    For a given real tx hash:
      - get enriched features from data_fetcher.get_transaction_details
      - call model via backend (GET /transaction/hash OR POST /predict/transaction)
      - update on-chain (updateTransactionRisk, updateWalletScore)
    """
    if tx_hash_hex in processed_tx_hashes:
        return False

    # get details via fetcher (this calls web3 under the hood)
    details = None
    try:
        details = fetcher.get_transaction_details(tx_hash_hex)
    except Exception as e:
        logger.debug(f"fetcher.get_transaction_details failed for {tx_hash_hex}: {e}")
        # still continue with minimal info
        details = {}

    tx_obj = normalize_tx_dict(tx_hash_hex, details)
    from_addr = tx_obj.get('from_address') or tx_obj.get('from') or None
    to_addr = tx_obj.get('to_address') or tx_obj.get('to') or None
    value_eth = float(tx_obj.get('Transaction_Value') or tx_obj.get('value') or 0.0)

    logger.info(f"Processing tx {tx_hash_hex} from {from_addr} -> {to_addr} value={value_eth} ETH")

    # Prepare features for ML
    features = {
        "Transaction_Value": float(tx_obj.get("Transaction_Value", value_eth)),
        "Transaction_Fees": float(tx_obj.get("Transaction_Fees", 0.0)),
        "Number_of_Inputs": float(tx_obj.get("Number_of_Inputs", 1.0)),
        "Number_of_Outputs": float(tx_obj.get("Number_of_Outputs", 1.0)),
        "Gas_Price": float(tx_obj.get("Gas_Price", 0.0)),
        "Wallet_Age_Days": float(tx_obj.get("Wallet_Age_Days", 0.0)),
        "Wallet_Balance": float(tx_obj.get("Wallet_Balance", 0.0)),
        "Transaction_Velocity": float(tx_obj.get("Transaction_Velocity", 0.0)),
        "Exchange_Rate": float(tx_obj.get("Exchange_Rate", 0.0))
    }

    # Call ML (prefer GET lookup which may include predictions)
    risk = call_ml_for_tx(tx_hash_hex, features)
    if risk is None:
        logger.warning("Skipping tx due to ML API failure")
        return False

    logger.info(f"ML risk for {tx_hash_hex}: {risk:.4f}")

    # Compute new wallet score (use chain current score)
    try:
        cur_int = fraud_detection_contract.functions.getWalletTrustScore(from_addr).call() if from_addr else 0
        current_score = float(cur_int) / 100.0 if cur_int is not None else score_updater.get_initial_score()
    except Exception:
        current_score = score_updater.get_initial_score()

    if current_score == 0:
        current_score = score_updater.get_initial_score()

    tx_value_usd = features["Transaction_Value"] * features["Exchange_Rate"]
    new_score = score_updater.calculate_new_score(current_score, float(risk), tx_value_usd)

    # Convert tx hash to bytes32 (use binary form)
    try:
        tx_bytes32 = Web3.toBytes(hexstr=tx_hash_hex)
    except Exception:
        tx_bytes32 = Web3.keccak(text=tx_hash_hex)[:32]

    # send updateTransactionRisk
    try:
        send_update_tx(tx_bytes32,
                       Web3.to_checksum_address(from_addr) if from_addr else oracle_account.address,
                       Web3.to_checksum_address(to_addr) if to_addr else oracle_account.address,
                       float(value_eth),
                       float(risk))
    except Exception as e:
        logger.exception(f"Failed to send updateTransactionRisk for {tx_hash_hex}: {e}")

    # update wallet score if changed
    try:
        cur_chain_int = 0
        try:
            cur_chain_int = fraud_detection_contract.functions.getWalletTrustScore(from_addr).call()
        except:
            cur_chain_int = int(current_score * 100)
        if int(new_score * 100) != int(cur_chain_int):
            send_update_wallet_score(Web3.to_checksum_address(from_addr), new_score)
    except Exception as e:
        logger.exception(f"Failed to update wallet score for {from_addr}: {e}")

    processed_tx_hashes.add(tx_hash_hex)
    return True

def get_recent_tx_hashes(limit=20):
    """
    Read the latest blocks and collect unique transaction hashes until we have 'limit' txs.
    This avoids relying on a possibly feature-only list returned by the fetcher.
    """
    hashes = []
    latest_block = w3.eth.get_block('latest')
    block_number = latest_block.number
    blocks_checked = 0
    max_blocks = 50
    while len(hashes) < limit and block_number > 0 and blocks_checked < max_blocks:
        try:
            block = w3.eth.get_block(block_number, full_transactions=True)
            for tx in block.transactions:
                h = tx.hash.hex() if hasattr(tx.hash, 'hex') else str(tx.hash)
                if h not in hashes:
                    hashes.append(h)
                    if len(hashes) >= limit:
                        break
            block_number -= 1
            blocks_checked += 1
        except Exception as e:
            logger.debug(f"Error reading block {block_number}: {e}")
            block_number -= 1
            blocks_checked += 1
            continue
    return hashes

# ---------- Replace or add these helper functions in backend/oracle.py ----------



logger = logging.getLogger(__name__)

API_PREDICT_URL = API_URL  # your existing constant: "http://127.0.0.1:5000/predict/transaction"

def post_features_to_ml_api(features, tx_hash=None, max_retries=3, timeout=12):
    """
    Post the features dict to the ML backend POST /predict/transaction
    Retries on transient errors. Returns parsed JSON on success, or None on failure.
    """
    if not isinstance(features, dict):
        logger.warning("post_features_to_ml_api: features is not a dict; skipping.")
        return None

    payload = {"features": features}
    headers = {"Content-Type": "application/json"}

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"Posting features to ML API (attempt {attempt}) for tx {tx_hash}")
            r = requests.post(API_PREDICT_URL, json=payload, headers=headers, timeout=timeout)
            logger.debug(f"ML API returned status {r.status_code} for tx {tx_hash}")
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception as e:
                    logger.error(f"Failed to parse JSON from ML API for tx {tx_hash}: {e}")
                    return None
            else:
                # log body for debugging (truncate)
                logger.error(f"ML API error {r.status_code}: {r.text[:400]}")
        except requests.exceptions.Timeout:
            logger.warning(f"ML API timeout on attempt {attempt} for tx {tx_hash}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"ML API connection error on attempt {attempt} for tx {tx_hash}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error calling ML API on attempt {attempt} for tx {tx_hash}: {e}")

        # small backoff
        time.sleep(1.5 * attempt)

    logger.error(f"All attempts failed to call ML API for tx {tx_hash}")
    return None


def process_transaction_dict(tx_dict):
    """
    tx_dict is the dict returned by data_fetcher.get_transaction_details()
    This function:
      - extracts features
      - calls ML backend (POST /predict/transaction)
      - returns (risk_probability, ml_response) or (None, None) on failure
    """
    try:
        if not isinstance(tx_dict, dict):
            logger.debug("process_transaction_dict: input is not a dict; skipping")
            return None, None

        # tx metadata
        tx_hash = tx_dict.get("hash") or tx_dict.get("tx_hash") or tx_dict.get("transaction_hash") or None
        from_addr = tx_dict.get("from_address") or tx_dict.get("from") or None
        to_addr = tx_dict.get("to_address") or tx_dict.get("to") or None
        value = float(tx_dict.get("Transaction_Value") or tx_dict.get("value") or 0.0)

        # extract features — prefer the exact names your model expects
        features = {}
        # key list must match what your model expects (defensive)
        expected_keys = [
            "Transaction_Value", "Transaction_Fees", "Number_of_Inputs",
            "Number_of_Outputs", "Gas_Price", "Wallet_Age_Days",
            "Wallet_Balance", "Transaction_Velocity", "Exchange_Rate"
        ]
        for k in expected_keys:
            # attempt several possible fallback key names
            if k in tx_dict:
                features[k] = tx_dict.get(k)
            else:
                # try some common alternatives
                alt = {
                    "Transaction_Value": ["value", "value_eth", "amount"],
                    "Transaction_Fees": ["fee", "fees", "transaction_fee"],
                    "Number_of_Inputs": ["num_inputs", "inputs"],
                    "Number_of_Outputs": ["num_outputs", "outputs"],
                    "Gas_Price": ["gasPrice", "gas_price", "Gas_Price"],
                    "Wallet_Age_Days": ["wallet_age_days", "wallet_age"],
                    "Wallet_Balance": ["wallet_balance", "balance"],
                    "Transaction_Velocity": ["tx_velocity", "transaction_velocity"],
                    "Exchange_Rate": ["exchange_rate", "eth_price"]
                }.get(k, [])
                found = False
                for a in alt:
                    if a in tx_dict:
                        features[k] = tx_dict.get(a)
                        found = True
                        break
                if not found:
                    features[k] = 0.0

        # Convert numeric fields to floats (defensive)
        for kk in features:
            try:
                features[kk] = float(features[kk] or 0.0)
            except:
                features[kk] = 0.0

        # Now call the ML API via POST with the features
        ml_response = post_features_to_ml_api(features, tx_hash=tx_hash)
        if not ml_response:
            logger.warning(f"Skipping tx {tx_hash} because ML response was not available")
            return None, None

        # ML response should contain risk_probability or similar
        risk = ml_response.get("risk_probability") or ml_response.get("risk_score") or ml_response.get("risk") or None
        if risk is None:
            # try other shapes
            if "result" in ml_response and isinstance(ml_response["result"], dict):
                risk = ml_response["result"].get("risk_probability") or ml_response["result"].get("risk_score")
        try:
            risk_probability = float(risk)
        except:
            risk_probability = None

        return risk_probability, ml_response

    except Exception as e:
        logger.exception(f"Error in process_transaction_dict: {e}")
        return None, None

# ----------------- Example usage inside your main loop -----------------
# Replace existing processing logic that used try_backend_lookup/call_ml_for_tx
# with something like this:

def main_loop():
    logger.info("\n--- Oracle Service Started: Processing real transactions from Sepolia ---")
    # initialize fetcher (if not already), e.g.:
    from data_fetcher import EthereumDataFetcher
    fetcher = EthereumDataFetcher(SEPOLIA_RPC_URL)
    while True:
        try:
            # fetch some recent txs (already returns feature dicts)
            txs = fetcher.get_latest_transactions(20)
            logger.info(f"Collected {len(txs)} tx hashes to inspect")
            for tx in txs:
                # tx is a dict from data_fetcher.get_transaction_details()
                tx_hash = tx.get("hash") or tx.get("tx_hash") or tx.get("transaction_hash") or None
                from_addr = tx.get("from_address") or tx.get("from")
                to_addr = tx.get("to_address") or tx.get("to")
                value = tx.get("Transaction_Value") or tx.get("value") or 0.0

                # ensure we have features — if not, skip
                risk_probability, ml_response = process_transaction_dict(tx)
                if risk_probability is None:
                    logger.warning(f"Skipping tx {tx_hash} due to ML failure")
                    continue

                logger.info(f"Got ML risk: {risk_probability:.4f} for tx {tx_hash}")

                # now update chain with results (same code as you already have)
                # update_transaction_risk_on_chain expects bytes32 for tx_hash, ensure you pass a 32-byte value
                try:
                    # if tx_hash is hex string (0x...), convert to bytes32 appropriate for your contract.
                    # Some contracts accept bytes32 hex strings directly; else use Web3.toBytes(hexstr=tx_hash)
                    tx_hash_bytes = tx_hash
                    # adjust depending on your contract signature:
                    update_transaction_risk_on_chain(tx_hash_bytes, from_addr, to_addr, float(value), risk_probability)
                except Exception as e:
                    logger.error(f"Error sending updateTransactionRisk for {tx_hash}: {e}")

            logger.info("Loop done - sleeping 60s")
            time.sleep(60)

        except Exception as e:
            logger.exception(f"Main loop error: {e}")
            time.sleep(10)

# End of replacement code


def main_loop():
    logger.info("Oracle started — processing real transactions from Sepolia.")
    while True:
        try:
            tx_hashes = get_recent_tx_hashes(TXS_PER_POLL)
            logger.info(f"Collected {len(tx_hashes)} tx hashes to inspect")
            processed = 0
            for th in tx_hashes:
                try:
                    ok = process_transaction(th)
                    if ok:
                        processed += 1
                except Exception as e:
                    logger.exception(f"Error processing tx {th}: {e}")
            logger.info(f"Iteration complete — processed {processed} tx(s). Sleeping {POLL_INTERVAL_SECONDS}s")
        except Exception as e:
            logger.exception(f"Unhandled error in the oracle loop: {e}")
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    main_loop()
