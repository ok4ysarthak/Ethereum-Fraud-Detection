# decode_tx.py
from web3 import Web3
import json

SEPOLIA_RPC_URL = "https://eth-sepolia.g.alchemy.com/v2/5bURjldvKPu4glB_tFxWt"
TX_HASH = "0x81e2b535c9bd5b729915ecd5eb56469d1cd668b497124121fe2f8131e5e2086a"
ABI_PATH = "artifacts/contracts/TrustScore.sol/FraudDetection.json"  # adjust path

w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC_URL))
with open(ABI_PATH, 'r') as f:
    contract_json = json.load(f)
abi = contract_json['abi']

tx = w3.eth.get_transaction(TX_HASH)
print("Signer (tx.from):", tx['from'])
print("To (contract):", tx['to'])
print("Input (raw):", tx['input'][:200], "...")

contract = w3.eth.contract(address=tx['to'], abi=abi)
fn_obj, fn_params = contract.decode_function_input(tx['input'])
print("Function called:", fn_obj.fn_name)
print("Decoded params:", fn_params)
