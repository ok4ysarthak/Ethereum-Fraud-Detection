# run from project root: python -m backend.backfill_wallet_scores
from db import SessionLocal, engine
from models_db import Wallet, Transaction
from sqlalchemy import desc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backfill")

def backfill():
    session = SessionLocal()
    # get distinct from_addresses that have a wallet_trust_score in transactions
    q = session.query(Transaction.from_address).filter(Transaction.wallet_trust_score != None).distinct()
    addresses = [r[0] for r in q.all()]
    logger.info("Found %d addresses to backfill", len(addresses))

    for addr in addresses:
        # find latest transaction (by id or timestamp) for this address with a wallet_trust_score
        tx = session.query(Transaction).filter(
            Transaction.from_address == addr,
            Transaction.wallet_trust_score != None
        ).order_by(desc(Transaction.id)).first()
        if not tx:
            continue
        score = float(tx.wallet_trust_score)
        w = session.query(Wallet).filter(Wallet.address == addr).first()
        if not w:
            # create a wallet row if missing
            w = Wallet(address=addr, trust_score=score)
            session.add(w)
            session.commit()
            logger.info("Created wallet %s with score %s", addr, score)
        else:
            w.trust_score = score
            session.add(w)
            session.commit()
            logger.info("Updated wallet %s -> score %s", addr, score)

    session.close()
    logger.info("Backfill complete.")

if __name__ == "__main__":
    backfill()
