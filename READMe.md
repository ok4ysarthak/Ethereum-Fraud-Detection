                                            DeTrust Protocol — Ethereum Risk Analysis Dashboard

**DeTrust Protocol** is a web application designed to monitor **live Ethereum transactions** and provide **real-time risk analysis**.  
It assigns **risk scores** to transactions and **trust scores** to wallets, helping users identify potentially fraudulent or suspicious activity.

---

Core Features

- **🔴 Live Transactions:** View a real-time feed of new Ethereum transactions as they occur.  
- **⚖️ Risk Scoring:** Automatically flags transactions with a calculated risk score (e.g., *High Risk*, *Low Risk*).  
- **🪙 Wallet Scoring:** Provides a trust score for wallets involved in transactions.  
- **🔍 Wallet Lookup:** *(Component in place)* Analyze a specific wallet address.  
- **🔗 Transaction Lookup:** *(Component in place)* Look up the details of a specific transaction hash.  
- **📜 Contract Info:** *(Component in place)* View information about a smart contract.

---

Tech Stack

**Frontend:** React, React Router, React Bootstrap, Axios  
**Backend:** Python (Flask / FastAPI)

---

Getting Started

This project uses a **separate frontend and backend**, which must be run **simultaneously** in two different terminal windows.

---

Prerequisites

Make sure you have the following installed:
- **Node.js** (v16 or higher)  
- **Python** (v3.8 or higher)  
- **npm** *(comes with Node.js)*  
- **pip** *(comes with Python)*

---
🖥️ Setup Instructions

Backend Setup (Terminal 1)

First, set up and run the Python backend server.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


# 2. Navigate to the backend folder (adjust if your folder structure is different)
cd path-to-your-backend-folder

# 3. Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# 4. Install the required Python packages
# (Assuming you have a requirements.txt file)
pip install -r requirements.txt

# 5. Run the backend server
python app.py

# 6. The server should now be running on:
# http://localhost:5000

```

2️⃣ Frontend Setup (Terminal 2)

Now, in a new terminal window, set up and run the React frontend.

```
# 1. Navigate to the frontend folder from the project root
cd frontend/detrust-frontend  # Based on your project structure

# 2. Install the node modules
npm install
```

⚠️ IMPORTANT: Configure the Proxy

Open the package.json file in this folder and ensure the following line is present inside the main JSON object {}:

"proxy": "http://localhost:5000"

This allows the frontend to communicate with the backend.

3. Run the frontend application
npm start

Your browser should automatically open at:
👉 http://localhost:3000

The app will now fetch data from the backend running on port 5000.

How It Works

The React development server at http://localhost:3000 serves the user interface.
The "proxy" setting in package.json forwards unknown API requests (like /transactions) to the backend server at http://localhost:5000

This setup prevents CORS issues — a standard practice for local development.

.

📂 Project Structure (Frontend)
frontend/detrust-frontend
├── public/
├── src/
│   ├── components/
│   │   ├── ContractInfo.js
│   │   ├── Header.js
│   │   ├── Home.js
│   │   ├── TransactionLookup.js
│   │   ├── Transactions.js
│   │   └── WalletLookup.js
│   ├── App.css
│   ├── App.js
│   ├── index.css
│   └── index.js
├── .gitignore
├── package.json
└── README.md


