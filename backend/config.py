# backend/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Ethereum Configuration
    ALCHEMY_API_KEY = os.getenv('ALCHEMY_API_KEY')
    ETHEREUM_PROVIDER_URL = f'https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}'
    SEPOLIA_RPC_URL = os.getenv('SEPOLIA_RPC_URL')  # For testnet
    ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')
    
    # Wallet Keys (Keep these secure!)
    SEPOLIA_PRIVATE_KEY = os.getenv('SEPOLIA_PRIVATE_KEY')
    ORACLE_PRIVATE_KEY = os.getenv('ORACLE_PRIVATE_KEY')
    
    # Smart Contract
    CONTRACT_ADDRESS = os.getenv('CONTRACT_ADDRESS')
    
    # Model Configuration
    MODEL_PATH = '../ml_models/saved_models/fraud_detector_model.pkl'
    
    # Processing Configuration
    BATCH_INTERVAL_MINUTES = 10
    TRANSACTIONS_PER_BATCH = 100
    
    # Risk Thresholds
    HIGH_RISK_THRESHOLD = 0.7
    MEDIUM_RISK_THRESHOLD = 0.4
    LOW_TRUST_THRESHOLD = 0.3
    
    # Network Selection
    NETWORK = 'sepolia'  # or 'mainnet'
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'backend/logs/fraud_detection.log'
    
    # API Configuration
    API_HOST = '0.0.0.0'
    API_PORT = 5000
    
    @staticmethod
    def get_rpc_url():
        return Config.SEPOLIA_RPC_URL

# Create logs directory if it doesn't exist
import os
log_dir = os.path.dirname('backend/logs/fraud_detection.log')
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)
