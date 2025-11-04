# backend/batch_processor.py
import time
import logging
from datetime import datetime
from config import Config

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, data_fetcher, fraud_detector, wallet_scorer):
        self.data_fetcher = data_fetcher
        self.fraud_detector = fraud_detector
        self.wallet_scorer = wallet_scorer

        # Keep a dict of tx_hash -> processed record (most recent first)
        self.processed_transactions = {}  # {tx_hash: record}
        self.max_records = 1000  # keep up to 1000 recent processed txs

    def _add_processed_record(self, record):
        """Add or update a processed tx record, keep bounded size."""
        tx_hash = record.get("id") or record.get("transaction_hash")
        if not tx_hash:
            return
        self.processed_transactions[tx_hash] = record
        # If too big, remove oldest (by timestamp)
        if len(self.processed_transactions) > self.max_records:
            # remove the earliest timestamp record
            items = sorted(self.processed_transactions.items(), key=lambda kv: kv[1].get("timestamp", 0))
            oldest_key = items[0][0]
            del self.processed_transactions[oldest_key]

    def process_new_transactions(self):
        """Process new transactions and update wallet scores for contract-related txs."""
        logger.info("Starting batch processing...")

        try:
            # Prefer contract-specific fetch if available
            limit = Config.BATCH_FETCH_LIMIT
            txs = []
            if hasattr(self.data_fetcher, "get_contract_transactions"):
                txs = self.data_fetcher.get_contract_transactions(
                    contract_address=Config.CONTRACT_ADDRESS,
                    limit=limit
                )
            else:
                txs = self.data_fetcher.get_latest_transactions(limit)

            logger.info(f"Fetched {len(txs)} transactions for processing")

            for raw in txs:
                try:
                    # raw should have from_address, to_address, timestamp and feature fields
                    tx_hash = raw.get("transaction_hash") or raw.get("hash") or raw.get("id")
                    # Skip if we already processed this tx_hash and recorded it
                    if tx_hash and tx_hash in self.processed_transactions:
                        continue

                    # run model to predict fraud risk
                    fraud_risk = 0.0
                    try:
                        # assume the detector can accept dict features (as in your app)
                        fraud_risk = float(self.fraud_detector.predict_fraud_risk(raw))
                    except Exception:
                        # fallback if detector requires numeric vectors
                        try:
                            # attempt sklearn style
                            import numpy as np
                            arr = np.array([[ raw.get('Transaction_Value',0),
                                               raw.get('Transaction_Fees',0),
                                               raw.get('Number_of_Inputs',0),
                                               raw.get('Number_of_Outputs',0),
                                               raw.get('Gas_Price',0),
                                               raw.get('Wallet_Age_Days',0),
                                               raw.get('Wallet_Balance',0),
                                               raw.get('Transaction_Velocity',0),
                                               raw.get('Exchange_Rate',0)
                                             ]], dtype=float)
                            if hasattr(self.fraud_detector, 'predict_proba'):
                                prob = self.fraud_detector.predict_proba(arr)
                                fraud_risk = float(prob[0][1])
                            else:
                                pred = self.fraud_detector.predict(arr)
                                fraud_risk = float(pred[0])
                        except Exception as e:
                            logger.warning(f"Model fallback failed: {e}")
                            fraud_risk = 0.0

                    # update wallet history and scores for sender
                    from_addr = raw.get('from_address') or raw.get('from')
                    if from_addr and hasattr(self.wallet_scorer, 'update_wallet_history'):
                        try:
                            self.wallet_scorer.update_wallet_history(from_addr, raw)
                            wallet_history = self.wallet_scorer.wallet_history.get(from_addr, [])
                            # calculate wallet score if method exists
                            if hasattr(self.wallet_scorer, 'calculate_wallet_score'):
                                wallet_score = self.wallet_scorer.calculate_wallet_score(from_addr, wallet_history)
                            elif hasattr(self.wallet_scorer, 'get_wallet_score'):
                                wallet_score = self.wallet_scorer.get_wallet_score(from_addr)
                            else:
                                wallet_score = None
                            if wallet_score is not None:
                                # store latest wallet score
                                if not hasattr(self.wallet_scorer, 'wallet_scores'):
                                    self.wallet_scorer.wallet_scores = {}
                                self.wallet_scorer.wallet_scores[from_addr] = wallet_score
                        except Exception as e:
                            logger.debug(f"Error updating wallet history/score: {e}")
                            wallet_score = None
                    else:
                        wallet_score = None

                    # normalized record for frontend
                    record = {
                        "id": tx_hash,
                        "transaction_hash": tx_hash,
                        "from": from_addr,
                        "to": raw.get('to_address') or raw.get('to'),
                        "value": f"{raw.get('Transaction_Value', 0)} ETH",
                        "riskScore": float(fraud_risk),
                        "walletScore": float(wallet_score) if wallet_score is not None else None,
                        "timestamp": raw.get('timestamp') or int(time.time()),
                        "status": "flagged" if float(fraud_risk) > Config.HIGH_RISK_THRESHOLD else "processed",
                        "raw": raw
                    }

                    # store processed record
                    self._add_processed_record(record)

                except Exception as e:
                    logger.error(f"Error processing single transaction: {e}")
                    continue

            logger.info(f"Batch processed. Stored processed transaction count: {len(self.processed_transactions)}")
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")

    def start_scheduler(self):
        """Not used if app spawns its own thread. Kept for compatibility."""
        pass
