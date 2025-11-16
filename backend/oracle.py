# backend/oracle.py
import os
import json
import time
import logging
import requests
import traceback
from web3 import Web3
from dotenv import load_dotenv

# Local imports (make sure these modules exist & are importable)
from wallet_updater import WalletScoreUpdater
from data_fetcher import EthereumDataFetcher

# Load env (project root assumed; adjust path if needed)
load_dotenv(dotenv_path='../.env')
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL")
ORACLE_PRIVATE_KEY = os.getenv("ORACLE_PRIVATE_KEY")
API_URL = os.getenv("ML_API_URL", "http://127.0.0.1:5000")

POLL_INTERVAL_SECONDS = int(os.getenv("ORACLE_POLL_INTERVAL", "15"))
TXS_PER_POLL = int(os.getenv("ORACLE_TXS_PER_POLL", "5"))

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

# Instantiate helpers
score_updater = WalletScoreUpdater()
fetcher = EthereumDataFetcher(SEPOLIA_RPC_URL)

# caches
processed_tx_hashes = set()
wallet_score_cache = {}  # address -> last known int-score-on-chain (0-100)
ml_get_cache = {}        # optional small cache for Backend GET /transaction/<hash>

# ---------- ML API helpers ----------
API_BASE = API_URL.rstrip('/')
API_PREDICT_URL = f"{API_BASE}/predict/transaction"

def try_backend_lookup(tx_hash_hex):
    """
    Try GET /transaction/<tx_hash> on the ML backend.
    Returns parsed JSON if success or None.
    Tries several variants (with/without 0x; urlencoded).
    """
    import urllib.parse
    variants = [tx_hash_hex]
    if not tx_hash_hex.startswith("0x"):
        variants.insert(0, "0x" + tx_hash_hex)
    # add encoded forms
    variants += [urllib.parse.quote_plus(v) for v in list(variants)]

    seen = set()
    for v in variants:
        if v in seen:
            continue
        seen.add(v)
        url = f"{API_BASE}/transaction/{v}"
        try:
            logger.debug(f"Trying backend GET: {url}")
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception as e:
                    logger.error(f"GET {url} returned 200 but JSON parse failed: {e}")
                    return None
            else:
                logger.debug(f"GET {url} returned {r.status_code}")
        except Exception as e:
            logger.debug(f"GET request failed for {url}: {e}")
    return None

def post_features_to_ml_api(features, tx_hash=None, max_retries=3, timeout=12):
    """
    POST features to /predict/transaction. Retries on transient errors.
    Returns parsed JSON or None.
    """
    if not isinstance(features, dict):
        logger.warning("post_features_to_ml_api: features is not a dict; skipping.")
        return None

    payload = {"features": features}
    headers = {"Content-Type": "application/json"}
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"POSTing features to ML API (attempt {attempt}) for tx {tx_hash}")
            r = requests.post(API_PREDICT_URL, json=payload, headers=headers, timeout=timeout)
            logger.debug(f"ML API returned {r.status_code} for tx {tx_hash}")
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception as e:
                    logger.error(f"Failed to parse ML JSON for tx {tx_hash}: {e}")
                    return None
            else:
                logger.error(f"ML API error {r.status_code}: {r.text[:400]}")
        except requests.exceptions.Timeout:
            logger.warning(f"ML API timeout on attempt {attempt} for tx {tx_hash}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"ML API connection error on attempt {attempt} for tx {tx_hash}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error calling ML API on attempt {attempt} for tx {tx_hash}: {e}")
        time.sleep(1.2 * attempt)
    logger.error(f"All attempts failed to call ML API for tx {tx_hash}")
    return None


API_URL = "http://127.0.0.1:5000/transactions"  # change if different host/port
MAX_RETRIES = 5
RETRY_BASE = 0.5  # seconds

def post_transaction_to_api(tx_payload):
    """
    tx_payload: dict containing transaction data (must include `transaction_hash`).
    Performs retries with exponential backoff.
    """
    headers = {"Content-Type": "application/json"}
    body = json.dumps(tx_payload)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(API_URL, data=body, headers=headers, timeout=6)
            if resp.status_code in (200, 201):
                return resp.json()
            else:
                # log and maybe retry
                print(f"[oracle->api] Unexpected status {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            print(f"[oracle->api] Attempt {attempt} failed: {e}")
        # exponential backoff
        sleep_for = RETRY_BASE * (2 ** (attempt - 1))
        time.sleep(sleep_for)
    # after retries failed
    print("[oracle->api] All retries failed, saving to local queue (optional).")
    # Optionally persist to local disk/queue for later resend
    return None


def call_ml_for_tx(tx_hash_hex, features):
    """
    Try GET /transaction/<hash> first (fast, cached), else call POST /predict/transaction.
    Returns float risk_probability or None.
    """
    # check cache
    if tx_hash_hex in ml_get_cache:
        cached = ml_get_cache[tx_hash_hex]
        val = cached.get("risk_probability") or cached.get("risk_score") or cached.get("risk")
        try:
            return float(val)
        except:
            pass

    # 1) try GET lookup
    got = try_backend_lookup(tx_hash_hex)
    if got:
        ml_get_cache[tx_hash_hex] = got
        val = got.get("risk_probability") or got.get("risk_score") or got.get("risk")
        try:
            return float(val)
        except:
            pass

    # 2) fallback to POST features
    res = post_features_to_ml_api(features, tx_hash=tx_hash_hex)
    if not res:
        return None
    val = res.get("risk_probability") or res.get("risk_score") or res.get("risk")
    try:
        return float(val)
    except:
        # try nested shapes
        if "result" in res and isinstance(res["result"], dict):
            val2 = res["result"].get("risk_probability") or res["result"].get("risk_score")
            try:
                return float(val2)
            except:
                pass
    return None

def post_enriched_tx_to_backend(api_base, enriched_tx, max_retries=2, timeout=8):
    """
    Send enriched_tx (dict) to backend POST /transactions
    """
    if not api_base:
        logger.warning("No API base provided for posting enriched tx.")
        return False

    url = api_base.rstrip('/') + '/transactions'
    headers = {"Content-Type": "application/json"}
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, json=enriched_tx, headers=headers, timeout=timeout)
            if r.status_code in (200, 201):
                logger.info(f"Posted tx to backend: {enriched_tx.get('transaction_hash')[:12]} -> {r.status_code}")
                return True
            else:
                logger.warning(f"Backend POST /transactions returned {r.status_code}: {r.text[:200]}")
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed to post to backend: {e}")
            logger.debug(traceback.format_exc())
        time.sleep(0.8 * attempt)
    logger.error(f"All attempts failed to POST tx {enriched_tx.get('transaction_hash')}")
    return False

# ---------- On-chain update helpers ----------
def send_update_tx(tx_hash_bytes32, from_addr, to_addr, value_eth, risk_probability):
    """
    Build & send updateTransactionRisk tx on-chain.
    tx_hash_bytes32: bytes or bytes32 hex; we will attempt to normalize.
    """
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
            'gas': 350000,
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
    """
    Send on-chain updateWalletScore if necessary.
    """
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

# ---------- Normalizers & feature extractor ----------
def normalize_tx_dict(details):
    """
    Accept details returned by data_fetcher.get_transaction_details (or a raw tx dict).
    Return a normalized feature dict and canonical fields.
    """
    out = {}
    # hash field - prefer explicit transaction_hash
    tx_hash = details.get('transaction_hash') or details.get('hash') or details.get('tx_hash') or details.get('transactionHash')
    if tx_hash and not str(tx_hash).startswith("0x"):
        tx_hash = "0x" + str(tx_hash)
    out['hash'] = tx_hash

    out['from_address'] = details.get('from_address') or details.get('from') or details.get('sender')
    out['to_address'] = details.get('to_address') or details.get('to') or details.get('receiver')

    # numeric features
    out['Transaction_Value'] = float(details.get('Transaction_Value') or details.get('value') or 0.0)
    out['Transaction_Fees'] = float(details.get('Transaction_Fees') or details.get('Transaction_Fee') or 0.0)
    out['Number_of_Inputs'] = float(details.get('Number_of_Inputs') or 1)
    out['Number_of_Outputs'] = float(details.get('Number_of_Outputs') or 1)
    out['Gas_Price'] = float(details.get('Gas_Price') or 0.0)
    out['Wallet_Age_Days'] = float(details.get('Wallet_Age_Days') or 0.0)
    out['Wallet_Balance'] = float(details.get('Wallet_Balance') or 0.0)
    # velocity naming - data_fetcher returns tx_per_day_lifetime etc; collapse into Transaction_Velocity
    tx_per_day = details.get('tx_per_day_lifetime') or details.get('tx_per_day') or details.get('Transaction_Velocity') or 0.0
    out['Transaction_Velocity'] = float(tx_per_day or 0.0)
    out['Exchange_Rate'] = float(details.get('Exchange_Rate') or 0.0)
    out['timestamp'] = int(details.get('timestamp') or 0)
    return out

# ---------- Main processing flow ----------
def process_and_update(details):
    """
    Take a transaction 'details' dict from fetcher.get_transaction_details,
    call ML, compute new wallet score, and write updates on-chain.
    Returns True if processed & updated (or at least ML computed), False otherwise.
    """
    if not isinstance(details, dict):
        logger.debug("process_and_update: details is not a dict; skipping.")
        return False

    tx_norm = normalize_tx_dict(details)
    tx_hash = tx_norm.get('hash')
    if not tx_hash:
        logger.warning("Transaction missing hash; skipping")
        return False

    # avoid reprocessing
    if tx_hash in processed_tx_hashes:
        logger.debug(f"Skipping already processed tx {tx_hash}")
        return False

    from_addr = tx_norm.get('from_address') or oracle_account.address
    to_addr = tx_norm.get('to_address') or oracle_account.address
    value_eth = float(tx_norm.get('Transaction_Value') or 0.0)

    # call ML (GET or POST)
    risk = call_ml_for_tx(tx_hash, tx_norm)
    if risk is None:
        logger.warning(f"No ML risk for {tx_hash}; skipping further processing")
        return False

    logger.info(f"ML risk for {tx_hash}: {risk:.4f}")

    # compute wallet score
    # try reading on-chain value first (contract stores score as int 0-100 maybe scaled by 100 in your contract)
    chain_score_int = None
    try:
        chain_score_int = fraud_detection_contract.functions.getWalletTrustScore(from_addr).call()
        # assume contract returns an int (e.g., 4234 -> 42.34). We'll normalize later by dividing by 100
    except Exception as e:
        logger.debug(f"Could not read wallet score from chain for {from_addr}: {e}")

    if chain_score_int is not None:
        chain_score = float(chain_score_int) / 100.0
        wallet_score_cache[from_addr] = chain_score_int
    else:
        chain_score = score_updater.get_initial_score()
        wallet_score_cache[from_addr] = int(chain_score * 100)

    # compute new score using your WalletScoreUpdater
    tx_value_usd = tx_norm.get('Transaction_Value', 0.0) * tx_norm.get('Exchange_Rate', 0.0)
    new_score = score_updater.calculate_new_score(chain_score, float(risk), float(tx_value_usd))

    logger.info(f"Wallet {from_addr} current_score={chain_score:.2f} -> new_score={new_score:.2f}")

    # prepare tx hash bytes32 for contract call - try convert from hex
    try:
        tx_hash_bytes = Web3.toBytes(hexstr=tx_hash)
        # ensure 32 bytes
        if len(tx_hash_bytes) < 32:
            tx_hash_bytes = tx_hash_bytes.rjust(32, b'\x00')
        elif len(tx_hash_bytes) > 32:
            tx_hash_bytes = tx_hash_bytes[:32]
    except Exception:
        tx_hash_bytes = Web3.keccak(text=tx_hash)[:32]

    # send updateTransactionRisk and capture its tx hash
    tx_update_hash = None
    try:
        tx_update_hash = send_update_tx(
            tx_hash_bytes,
            Web3.to_checksum_address(from_addr),
            Web3.to_checksum_address(to_addr),
            float(value_eth),
            float(risk)
        )
    except Exception as e:
        logger.exception(f"Failed to send updateTransactionRisk for {tx_hash}: {e}")
    if tx_update_hash:
        try:
            requests.post(f"{API_BASE}/transactions/{tx_hash}/onchain", json={"onchain_tx_hash": tx_update_hash}, timeout=8)
        except Exception:
            logger.debug("Failed to notify backend about onchain tx")
    # send updateWalletScore only if changed and capture the tx hash
    wallet_update_tx = None
    try:
        current_chain_int = wallet_score_cache.get(from_addr)
        new_chain_int = int(new_score * 100)
        if current_chain_int is None:
            # attempt to read from chain again
            try:
                current_chain_int = fraud_detection_contract.functions.getWalletTrustScore(from_addr).call()
            except:
                current_chain_int = None

        if int(new_chain_int) != int(current_chain_int or 0):
            logger.info(f"Updating wallet score on-chain for {from_addr}: {current_chain_int} -> {new_chain_int}")
            wallet_update_tx = send_update_wallet_score(Web3.to_checksum_address(from_addr), new_score)
            wallet_score_cache[from_addr] = new_chain_int
        else:
            logger.debug(f"No wallet score change for {from_addr}: {current_chain_int} == {new_chain_int}")
    except Exception as e:
        logger.exception(f"Failed while deciding/updating wallet score for {from_addr}: {e}")

    # Build payload to POST to backend ingest endpoint
    try:
        payload = {
            "transaction_hash": tx_hash,
            "from_address": Web3.to_checksum_address(from_addr) if from_addr else None,
            "to_address": Web3.to_checksum_address(to_addr) if to_addr else None,
            "Transaction_Value": float(tx_norm.get('Transaction_Value', 0.0)),
            "Transaction_Fees": float(tx_norm.get('Transaction_Fees', 0.0)) if tx_norm.get('Transaction_Fees') is not None else float(details.get('Transaction_Fees', 0.0)),
            "Gas_Price": float(tx_norm.get('Gas_Price', 0.0)),
            "Wallet_Age_Days": float(tx_norm.get('Wallet_Age_Days', 0.0)),
            "Wallet_Balance": float(tx_norm.get('Wallet_Balance', 0.0)),
            "Transaction_Velocity": float(tx_norm.get('Transaction_Velocity', 0.0)),
            "Exchange_Rate": float(tx_norm.get('Exchange_Rate', 0.0)),
            "timestamp": int(tx_norm.get('timestamp') or details.get('timestamp') or int(time.time())),
            "risk_probability": float(risk),
            "wallet_trust_before": chain_score,
            "wallet_trust_score": float(new_score),
            "onchain_tx_update": tx_update_hash,
            "onchain_wallet_update": wallet_update_tx,
            "raw_details": details
        }
        # Non-blocking post to backend; log failure but do not crash oracle
        posted = post_enriched_tx_to_backend(API_BASE, payload)
        if not posted:
            logger.warning(f"Posting enriched tx to backend failed for {tx_hash} (continuing).")
    except Exception as e:
        logger.exception(f"Failed to build/post enriched payload for {tx_hash}: {e}")

    processed_tx_hashes.add(tx_hash)
    return True

# ---------- Helpers to gather latest txs ----------
def get_recent_tx_details(limit=5):
    """
    Use your fetcher to get latest transaction details (enriched dicts).
    This is preferred to iterating raw tx hashes because fetcher already enriches and handles block parsing.
    """
    try:
        txs = fetcher.get_latest_transactions(limit)
        return txs
    except Exception as e:
        logger.exception(f"Failed to fetch latest transactions via fetcher: {e}")
        return []

# ---------- Main loop ----------
def main_loop():
    logger.info("Oracle started — processing real transactions from Sepolia.")
    while True:
        try:
            tx_details_list = get_recent_tx_details(TXS_PER_POLL)
            logger.info(f"Collected {len(tx_details_list)} transactions to inspect")
            processed = 0
            for details in tx_details_list:
                try:
                    ok = process_and_update(details)
                    if ok:
                        processed += 1
                except Exception as e:
                    logger.exception(f"Error processing tx details: {e}")
            logger.info(f"Iteration complete — processed {processed}/{len(tx_details_list)} tx(s). Sleeping {POLL_INTERVAL_SECONDS}s")
        except Exception as e:
            logger.exception(f"Unhandled error in the oracle loop: {e}")
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    main_loop()
