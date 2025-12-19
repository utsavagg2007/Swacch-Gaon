# Run Swachh Gaon on VS Code (Local)

This project is:
- Frontend: React + Tailwind (CRA + CRACO)
- Backend: FastAPI
- Database: MongoDB

---

## 1) Prerequisites
Install:
- **Python 3.11+**
- **Node.js 18+** (20 recommended)
- **Yarn**
- **MongoDB** (local) OR MongoDB Atlas
- VS Code extensions (recommended):
  - Python
  - ESLint
  - Prettier

---

## 2) Open the project in VS Code
1) Download/clone the code.
2) Open the folder in VS Code.

---

## 3) Backend setup
### A) Create backend virtualenv
```bash
cd backend
python -m venv .venv
```
Activate:
- Windows (PowerShell):
```bash
.\.venv\Scripts\Activate.ps1
```
- macOS/Linux:
```bash
source .venv/bin/activate
```

### B) Install backend dependencies
```bash
pip install -r requirements.txt
```

### C) Configure backend env
Create/edit `backend/.env`:
```env
MONGO_URL="mongodb://localhost:27017"
DB_NAME="swachh_gaon"
CORS_ORIGINS="*"

# Retell
RETELL_API_KEY="<your-retell-api-key>"
RETELL_FROM_NUMBER="+91XXXXXXXXXX"  # REQUIRED for outbound calls

# For scheduled calls + webhook reachability:
# Put a public URL here (ngrok is easiest locally)
PUBLIC_BACKEND_URL="https://<your-ngrok-subdomain>.ngrok-free.app"
```

### D) Run backend
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8001
```

---

## 4) Frontend setup
### A) Install dependencies
```bash
cd ../frontend
yarn
```

### B) Configure frontend env
Create/edit `frontend/.env`:
```env
REACT_APP_BACKEND_URL=http://localhost:8001
```

### C) Run frontend
```bash
yarn start
```
Open:
- http://localhost:3000/home

---

## 5) IMPORTANT for Retell webhook locally
Retell must hit a **public** URL.

### Use ngrok
In a new terminal:
```bash
ngrok http 8001
```
Copy the HTTPS URL and set it as:
```env
PUBLIC_BACKEND_URL="https://xxxx.ngrok-free.app"
```
Restart backend.

---

## 6) Using Retell inside the app
1) Create Retell agents (Morning + Evening) using `backend/retell_prompts.md`.
2) In Swachh Gaon Dashboard → **Retell AI Setup**:
   - Morning agent ID
   - Evening agent ID
   - **From number (required)**
3) Click **Save Retell setup**.
4) Click **Test: Run morning calls now**.

If you see an error, read the message shown inside the Retell card.

---

## Troubleshooting
- If calls fail with “from_number required”: set From number in dashboard or `RETELL_FROM_NUMBER`.
- If scheduled calls don’t run: `PUBLIC_BACKEND_URL` must be set.
- If no routes: the call endpoints will auto-generate routes for today.
