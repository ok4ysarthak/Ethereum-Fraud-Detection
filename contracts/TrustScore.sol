// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FraudDetection {
    struct TransactionRecord {
        address from;
        address to;
        uint256 value; // in wei
        uint256 timestamp;
        uint256 riskScore; // 0-100 scale
        bool isFlagged;
        bool exists;
    }
    
    struct WalletScore {
        address wallet;
        uint256 trustScore; // 0-100 scale
        uint256 lastUpdated;
        bool isHighRisk;
        bool exists;
    }
    event TransactionRiskUpdated(bytes32 indexed txHash, address indexed fromAddr, address indexed toAddr, uint256 valueWei, uint256 riskScore);
    event WalletScoreUpdated(address indexed wallet, uint256 trustScore);

    function updateTransactionRisk(bytes32 txHash, address fromAddr, address toAddr, uint256 valueWei, uint256 riskScore) public {
        // ... your logic
        emit TransactionRiskUpdated(txHash, fromAddr, toAddr, valueWei, riskScore);
    }

    function updateWalletScore(address wallet, uint256 trustScore) public {
        // ... your logic
        emit WalletScoreUpdated(wallet, trustScore);
    }

    // Storage mappings
    mapping(bytes32 => TransactionRecord) public transactionRecords;
    mapping(address => WalletScore) public walletScores;
    mapping(address => bool) public authorizedOracles;
    
    // Contract owner
    address public owner;
    
    // Configuration
    uint256 public minTrustScore = 30; // Minimum trust score required (0-100 scale)
    uint256 public highRiskThreshold = 70; // Threshold for high risk transactions
    
    // Events
    event TransactionRiskUpdated(
        bytes32 indexed txHash,
        address indexed from,
        address indexed to,
        uint256 riskScore,
        bool isFlagged
    );
    
    event WalletScoreUpdated(
        address indexed wallet,
        uint256 trustScore,
        bool isHighRisk
    );
    
    event OracleAuthorized(address indexed oracle);
    event OracleRevoked(address indexed oracle);
    event ConfigurationUpdated(string parameter, uint256 oldValue, uint256 newValue);
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Not contract owner");
        _;
    }
    
    modifier onlyAuthorizedOracle() {
        require(authorizedOracles[msg.sender], "Not authorized oracle");
        _;
    }
    
    modifier transactionExists(bytes32 txHash) {
        require(transactionRecords[txHash].exists, "Transaction does not exist");
        _;
    }
    
    modifier walletExists(address wallet) {
        require(walletScores[wallet].exists, "Wallet does not exist");
        _;
    }
    
    // Constructor
    constructor() {
        owner = msg.sender;
        authorizedOracles[msg.sender] = true;
        emit OracleAuthorized(msg.sender);
    }
    
    /**
     * @dev Update transaction risk score
     * @param txHash Transaction hash
     * @param from Sender address
     * @param to Receiver address
     * @param value Transaction value in wei
     * @param riskScore Risk score (0-100 scale)
     */
    function updateTransactionRisk(
        bytes32 txHash,
        address from,
        address to,
        uint256 value,
        uint256 riskScore
    ) external onlyAuthorizedOracle {
        require(riskScore <= 100, "Risk score must be between 0-100");
        require(from != address(0), "Invalid from address");
        require(to != address(0), "Invalid to address");
        
        bool isFlagged = riskScore >= highRiskThreshold;
        
        transactionRecords[txHash] = TransactionRecord({
            from: from,
            to: to,
            value: value,
            timestamp: block.timestamp,
            riskScore: riskScore,
            isFlagged: isFlagged,
            exists: true
        });
        
        emit TransactionRiskUpdated(txHash, from, to, riskScore, isFlagged);
    }
    
    /**
     * @dev Update wallet trust score
     * @param wallet Wallet address
     * @param trustScore Trust score (0-100 scale)
     */
    function updateWalletScore(
        address wallet,
        uint256 trustScore
    ) external onlyAuthorizedOracle {
        require(trustScore <= 100, "Trust score must be between 0-100");
        require(wallet != address(0), "Invalid wallet address");
        
        bool isHighRisk = trustScore < minTrustScore;
        
        walletScores[wallet] = WalletScore({
            wallet: wallet,
            trustScore: trustScore,
            lastUpdated: block.timestamp,
            isHighRisk: isHighRisk,
            exists: true
        });
        
        emit WalletScoreUpdated(wallet, trustScore, isHighRisk);
    }
    
    /**
     * @dev Check if a transaction is allowed based on wallet trust score
     * @param from Sender wallet address
     * @param amount Transaction amount (not used in current implementation)
     * @return bool Whether transaction is allowed
     */
    function isTransactionAllowed(
        address from,
        uint256 amount
    ) external view returns (bool) {
        // If wallet doesn't exist, use default logic
        if (!walletScores[from].exists) {
            return true; // Allow new wallets by default
        }
        
        WalletScore memory walletScore = walletScores[from];
        return walletScore.trustScore >= minTrustScore && !walletScore.isHighRisk;
    }
    
    /**
     * @dev Get wallet trust score
     * @param wallet Wallet address
     * @return uint256 Trust score (0-100 scale)
     */
    function getWalletTrustScore(address wallet) external view returns (uint256) {
        if (!walletScores[wallet].exists) {
            return 50; // Return neutral score for unknown wallets
        }
        return walletScores[wallet].trustScore;
    }
    
    /**
     * @dev Get transaction risk score
     * @param txHash Transaction hash
     * @return uint256 Risk score (0-100 scale)
     */
    function getTransactionRisk(bytes32 txHash) external view transactionExists(txHash) returns (uint256) {
        return transactionRecords[txHash].riskScore;
    }
    
    /**
     * @dev Check if transaction is flagged as high risk
     * @param txHash Transaction hash
     * @return bool Whether transaction is flagged
     */
    function isTransactionFlagged(bytes32 txHash) external view transactionExists(txHash) returns (bool) {
        return transactionRecords[txHash].isFlagged;
    }
    
    /**
     * @dev Check if wallet is high risk
     * @param wallet Wallet address
     * @return bool Whether wallet is high risk
     */
    function isWalletHighRisk(address wallet) external view walletExists(wallet) returns (bool) {
        return walletScores[wallet].isHighRisk;
    }
    
    /**
     * @dev Get wallet information
     * @param wallet Wallet address
     * @return WalletScore struct
     */
    function getWalletInfo(address wallet) external view walletExists(wallet) returns (WalletScore memory) {
        return walletScores[wallet];
    }
    
    /**
     * @dev Get transaction information
     * @param txHash Transaction hash
     * @return TransactionRecord struct
     */
    function getTransactionInfo(bytes32 txHash) external view transactionExists(txHash) returns (TransactionRecord memory) {
        return transactionRecords[txHash];
    }
    
    /**
     * @dev Authorize an oracle address
     * @param oracle Oracle address to authorize
     */
    function authorizeOracle(address oracle) external onlyOwner {
        require(oracle != address(0), "Invalid oracle address");
        require(!authorizedOracles[oracle], "Oracle already authorized");
        
        authorizedOracles[oracle] = true;
        emit OracleAuthorized(oracle);
    }
    
    /**
     * @dev Revoke oracle authorization
     * @param oracle Oracle address to revoke
     */
    function revokeOracle(address oracle) external onlyOwner {
        require(oracle != owner, "Cannot revoke owner");
        require(authorizedOracles[oracle], "Oracle not authorized");
        
        authorizedOracles[oracle] = false;
        emit OracleRevoked(oracle);
    }
    
    /**
     * @dev Set minimum trust score threshold
     * @param score New minimum trust score (0-100)
     */
    function setMinTrustScore(uint256 score) external onlyOwner {
        require(score <= 100, "Score must be between 0-100");
        
        uint256 oldScore = minTrustScore;
        minTrustScore = score;
        
        emit ConfigurationUpdated("minTrustScore", oldScore, score);
    }
    
    /**
     * @dev Set high risk threshold
     * @param threshold New high risk threshold (0-100)
     */
    function setHighRiskThreshold(uint256 threshold) external onlyOwner {
        require(threshold <= 100, "Threshold must be between 0-100");
        require(threshold > minTrustScore, "High risk threshold must be above min trust score");
        
        uint256 oldThreshold = highRiskThreshold;
        highRiskThreshold = threshold;
        
        emit ConfigurationUpdated("highRiskThreshold", oldThreshold, threshold);
    }
    
    /**
     * @dev Check if an address is an authorized oracle
     * @param oracle Address to check
     * @return bool Whether address is authorized oracle
     */
    function isAuthorizedOracle(address oracle) external view returns (bool) {
        return authorizedOracles[oracle];
    }
    
    /**
     * @dev Get contract configuration
     * @return minTrustScore, highRiskThreshold
     */
    function getConfiguration() external view returns (uint256, uint256) {
        return (minTrustScore, highRiskThreshold);
    }
    
    /**
     * @dev Get total number of tracked wallets
     * @return uint256 Number of wallets
     */
    function getWalletCount() external view returns (uint256) {
        // This is a simplified count - in practice, you might want to maintain a counter
        // This function is mainly for demonstration
        return 0; // Implementation would require additional storage
    }
}
