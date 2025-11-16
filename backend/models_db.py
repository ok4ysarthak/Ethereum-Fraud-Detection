# backend/models_db.py
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON, BigInteger
)
import datetime

Base = declarative_base()

class Wallet(Base):
    __tablename__ = "wallets"
    id = Column(Integer, primary_key=True, index=True)
    address = Column(String(66), unique=True, index=True, nullable=False)
    first_seen = Column(DateTime, default=datetime.datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.datetime.utcnow)
    age_days = Column(Integer, nullable=True)
    trust_score = Column(Float, nullable=True)
    avg_risk = Column(Float, nullable=True)
    labels = Column(JSON, nullable=True)
    metadata_json = Column("metadata_json", JSON, nullable=True)   # renamed

    outgoing = relationship("Transaction", back_populates="from_wallet", foreign_keys="Transaction.from_wallet_id")
    incoming = relationship("Transaction", back_populates="to_wallet", foreign_keys="Transaction.to_wallet_id")


class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    tx_hash = Column(String(66), unique=True, index=True, nullable=False)
    block_number = Column(BigInteger, nullable=True)
    from_wallet_id = Column(Integer, ForeignKey("wallets.id"), nullable=True)
    to_wallet_id = Column(Integer, ForeignKey("wallets.id"), nullable=True)
    from_address = Column(String(66), nullable=True)
    to_address = Column(String(66), nullable=True)
    amount_eth = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=True)
    wallet_trust_score = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    nonce = Column(BigInteger, nullable=True)
    gas = Column(BigInteger, nullable=True)
    status = Column(String(32), nullable=True)
    saved_to_chain = Column(Boolean, default=False)
    onchain_record_txhash = Column(String(66), nullable=True)
    metadata_json = Column("metadata_json", JSON, nullable=True)  # renamed
    raw_payload = Column(JSON, nullable=True)  # store full payload for debugging/search

    from_wallet = relationship("Wallet", foreign_keys=[from_wallet_id], back_populates="outgoing")
    to_wallet = relationship("Wallet", foreign_keys=[to_wallet_id], back_populates="incoming")
