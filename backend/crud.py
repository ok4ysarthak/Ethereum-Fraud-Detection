# backend/crud.py
from sqlalchemy import select, update, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert
from web3 import Web3
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def _normalize_address(addr):
    """Normalize to a lowercase 0x-prefixed hex string for stable DB keys."""
    if not addr:
        return None
    a = str(addr).strip()
    if not a:
        return None
    if not a.startswith("0x"):
        a = "0x" + a
    return a.lower()

def _to_dt(ts):
    """Convert unix int or iso string to datetime or return None."""
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(int(ts))
        return datetime.fromisoformat(str(ts))
    except Exception:
        try:
            return datetime.fromtimestamp(int(float(ts)))
        except Exception:
            return None

def create_transaction(session, data):
    """
    Insert or update Transaction row, and upsert corresponding Wallet sender row.
    - Will NOT overwrite an existing wallet.trust_score with NULL.
    - Stores normalized (lowercase) addresses for stable uniqueness.
    Returns the Transaction row object (SQLAlchemy instance).
    """
    try:
        # basic validations
        tx_hash = data.get("transaction_hash") or data.get("transactionHash") or data.get("hash")
        if not tx_hash:
            raise ValueError("Missing transaction_hash")

        # normalize tx_hash
        tx_hash = tx_hash if tx_hash.startswith("0x") else "0x" + tx_hash

        # addresses: normalize to lowercase 0x format
        from_addr = data.get("from_address") or data.get("from") or None
        to_addr = data.get("to_address") or data.get("to") or None
        from_addr = _normalize_address(from_addr)
        to_addr = _normalize_address(to_addr)

        # numeric fields
        try:
            risk_score = float(data.get("risk_probability")) if data.get("risk_probability") is not None else None
        except Exception:
            risk_score = None

        try:
            wallet_trust_score = float(data.get("wallet_trust_score")) if data.get("wallet_trust_score") is not None else None
        except Exception:
            wallet_trust_score = None

        ts_dt = _to_dt(data.get("timestamp"))

        # raw payload: prefer structured dict, but store string if needed
        raw_json = data.get("raw_details") or data.get("raw_transaction") or data.get("raw_payload") or data
        if isinstance(raw_json, (dict, list)):
            raw_payload_to_store = raw_json
        else:
            # try to parse string to json, else store as string
            try:
                raw_payload_to_store = json.loads(raw_json)
            except Exception:
                raw_payload_to_store = str(raw_json)

        # lazy import models to avoid circular imports
        from models_db import Transaction, Wallet

        # Build transaction upsert payload
        tx_upsert_values = {
            "tx_hash": tx_hash,
            "risk_score": risk_score,
            "wallet_trust_score": wallet_trust_score,
            "from_address": from_addr,
            "to_address": to_addr,
            "amount_eth": float(data.get("Transaction_Value") or data.get("value") or 0.0),
            "timestamp": ts_dt,
            "status": data.get("status") or ("flagged" if (risk_score is not None and risk_score > 0.6) else "processed"),
            "raw_payload": raw_payload_to_store,
            "onchain_record_txhash": data.get("onchain_tx_update") or data.get("onchain_tx") or None
        }

        # Upsert transaction (on tx_hash unique)
        stmt = insert(Transaction).values(**tx_upsert_values).on_conflict_do_update(
            index_elements=['tx_hash'],
            set_ = {
                "risk_score": tx_upsert_values["risk_score"],
                "wallet_trust_score": tx_upsert_values["wallet_trust_score"],
                "from_address": tx_upsert_values["from_address"],
                "to_address": tx_upsert_values["to_address"],
                "amount_eth": tx_upsert_values["amount_eth"],
                "timestamp": tx_upsert_values["timestamp"],
                "status": tx_upsert_values["status"],
                "raw_payload": tx_upsert_values["raw_payload"],
                "onchain_record_txhash": tx_upsert_values["onchain_record_txhash"]
            }
        )
        session.execute(stmt)

        # Upsert wallet (sender). We will NOT overwrite existing trust_score with NULL.
        if from_addr:
            wallet_vals = {
                "address": from_addr,
                "first_seen": ts_dt or datetime.utcnow(),
                "last_seen": ts_dt or datetime.utcnow(),
            }

            # If we have a non-null trust score, include it in the update set.
            if wallet_trust_score is not None:
                stmt_w = insert(Wallet).values(**{**wallet_vals, "trust_score": wallet_trust_score}).on_conflict_do_update(
                    index_elements=['address'],
                    set_ = {
                        "last_seen": wallet_vals["last_seen"],
                        "trust_score": wallet_trust_score
                    }
                )
            else:
                # Do not set trust_score in update; only update last_seen (preserve existing trust_score)
                stmt_w = insert(Wallet).values(**wallet_vals).on_conflict_do_update(
                    index_elements=['address'],
                    set_ = {
                        "last_seen": wallet_vals["last_seen"]
                    }
                )
            session.execute(stmt_w)

        # At this point, caller may commit (but commit here to be safer if caller forgets)
        try:
            session.commit()
        except Exception as e:
            logger.exception("Commit failed in create_transaction: %s", e)
            try:
                session.rollback()
            except:
                pass
            raise

        # return transaction row
        tx_row = session.query(Transaction).filter(Transaction.tx_hash == tx_hash).one_or_none()
        return tx_row

    except SQLAlchemyError as e:
        logger.exception("SQLAlchemy error in create_transaction: %s", e)
        try:
            session.rollback()
        except:
            pass
        raise
    except Exception as e:
        logger.exception("Error in create_transaction: %s", e)
        try:
            session.rollback()
        except:
            pass
        raise

def get_wallet_by_address(session, address):
    """
    Fetches a single wallet by its (normalized) address.
    """
    from models_db import Wallet
    
    norm_addr = _normalize_address(address)
    if not norm_addr:
        return None
        
    try:
        wallet = session.query(Wallet).filter(Wallet.address == norm_addr).one_or_none()
        return wallet
    except SQLAlchemyError as e:
        logger.exception("SQLAlchemy error in get_wallet_by_address: %s", e)
        session.rollback()
        return None
    except Exception as e:
        logger.exception("Error in get_wallet_by_address: %s", e)
        session.rollback()
        return None
    
def mark_tx_onchain(session, tx_hash, onchain_record_hash):
    """
    Finds a transaction by its original hash and marks it as saved on-chain
    by adding the new on-chain transaction hash.
    Returns the updated Transaction object or None.
    """
    from models_db import Transaction
    
    norm_hash = _normalize_address(tx_hash)
    if not norm_hash:
        logger.warning(f"mark_tx_onchain received invalid tx_hash: {tx_hash}")
        return None
        
    try:
        # Find the transaction
        tx = session.query(Transaction).filter(Transaction.tx_hash == norm_hash).one_or_none()
        
        if tx:
            # Update the transaction fields
            tx.saved_to_chain = True
            tx.onchain_record_txhash = onchain_record_hash
            tx.status = "confirmed_onchain"
            
            session.add(tx)
            session.commit()
            logger.info(f"Marked tx {norm_hash} as on-chain with hash {onchain_record_hash}")
            return tx
        else:
            # Transaction not found
            logger.warning(f"Could not find tx {norm_hash} to mark as on-chain.")
            return None
            
    except SQLAlchemyError as e:
        logger.exception("SQLAlchemy error in mark_tx_onchain: %s", e)
        session.rollback()
        return None
    except Exception as e:
        logger.exception("Error in mark_tx_onchain: %s", e)
        session.rollback()
        return None