# Rural Waste Optimization – MVP Plan

## 1) Goal (MVP)
Build a full‑stack app for a Panchayat to:
- Sign up (panchayat profile) → add **multiple wards** → add **multiple vehicles/drivers**
- Login and access a **private dashboard**
- Add and view daily waste **logs**
- Configure call **schedule times** (morning route briefing + evening waste collection)
- Run a backend **ML + optimization** pipeline that:
  - Predicts next-day ward-wise waste (RandomForestRegressor + hyperparameter tuning)
  - Produces an optimized vehicle allocation + route plan
  - Exposes **“data to Retell”** (outbound payload) and accepts **“data from Retell”** (webhook) to insert logs and retrain

> Note: Retell is not set up yet. For MVP we implement **Retell-ready endpoints** + sample payloads, without actually calling Retell APIs.

---

## 2) Architecture
### Frontend (React + Tailwind + shadcn/ui)
- Routes:
  - `/` → redirect to `/home`
  - `/home` landing page (navbar: logo + Sign in)
  - `/login` login form + link to `/signup`
  - `/signup` panchayat details + email + password → Next
  - `/signup/wards` add multiple wards (name + address) → Next/Back
  - `/signup/vehicle` add multiple vehicles (driver, phone, vehicle no, capacity) → Finish/Back
  - `/dashboard` private route: add logs + list logs + show latest optimized plan
  - `/dashboard/wards` edit wards
  - `/dashboard/vehicle` edit vehicles

- Auth: JWT stored in `localStorage`, attached as `Authorization: Bearer <token>`.

### Backend (FastAPI + MongoDB)
- Single FastAPI app with `/api` prefix.
- Collections:
  - `panchayats`, `wards`, `vehicles`, `logs`, `routes`, `settings`, `ml_models_meta`
- Key services:
  - Auth/JWT
  - Geocoding (OpenStreetMap Nominatim) for ward + panchayat address → lat/lon
  - ML training + prediction
  - Optimization + route ordering (haversine + heuristic clustering)
  - Retell payload + webhook ingestion

---

## 3) Database Schema (MongoDB)
### Panchayat
```json
{
  "_id": "uuid",
  "email": "string (unique)",
  "name": "string",
  "address": "string",
  "lat": 0.0,
  "lon": 0.0,
  "password_hash": "string",
  "created_at": "ISO"
}
```

### Ward
```json
{
  "_id": "uuid",
  "panchayat_id": "uuid",
  "name": "string",
  "address": "string",
  "lat": 0.0,
  "lon": 0.0,
  "created_at": "ISO"
}
```

### Vehicle
```json
{
  "_id": "uuid",
  "panchayat_id": "uuid",
  "driver_name": "string",
  "driver_phone": "string",
  "vehicle_number": "string",
  "capacity": 0.0,
  "created_at": "ISO"
}
```

### Log
```json
{
  "_id": "uuid",
  "panchayat_id": "uuid",
  "ward_id": "uuid",
  "vehicle_id": "uuid",
  "vehicle_number": "string",
  "waste_collected": 0.0,
  "log_date": "YYYY-MM-DD",
  "source": "manual | retell",
  "created_at": "ISO"
}
```

### Route Plan (next-day)
```json
{
  "_id": "uuid",
  "panchayat_id": "uuid",
  "plan_date": "YYYY-MM-DD",
  "vehicle_id": "uuid",
  "vehicle_number": "string",
  "driver_phone": "string",
  "round_trips": 1,
  "wards": [
    {"ward_id":"uuid","ward_name":"string","lat":0.0,"lon":0.0,"predicted_waste": 0.0}
  ],
  "predicted_total": 0.0,
  "route_order": ["ward_id", "ward_id"],
  "created_at": "ISO"
}
```

### Settings (call schedules)
```json
{
  "_id": "uuid",
  "panchayat_id": "uuid",
  "morning_call_time_ist": "HH:MM",
  "evening_call_time_ist": "HH:MM",
  "updated_at": "ISO"
}
```

---

## 4) API Design (MVP)
### Auth
- `POST /api/auth/signup` → create panchayat + JWT
- `POST /api/auth/login` → JWT
- `GET /api/auth/me` → current panchayat

### Panchayat
- `GET /api/panchayat` → profile
- `PUT /api/panchayat` → update profile

### Wards
- `GET /api/wards`
- `POST /api/wards`
- `PUT /api/wards/{ward_id}`
- `DELETE /api/wards/{ward_id}`

### Vehicles
- `GET /api/vehicles`
- `POST /api/vehicles`
- `PUT /api/vehicles/{vehicle_id}`
- `DELETE /api/vehicles/{vehicle_id}`

### Logs
- `GET /api/logs?limit=200`
- `POST /api/logs`

### Settings
- `GET /api/settings/call-schedule`
- `PUT /api/settings/call-schedule`

### Optimization + Routes
- `POST /api/optimization/run` → trains/tunes model (if enough data), predicts next-day, generates route plans, stores to `routes`
- `GET /api/routes?date=YYYY-MM-DD` → list route plans

### Retell-ready (data OUT / data IN)
- `GET /api/retell/morning/payload?date=YYYY-MM-DD`
  - returns per-driver call payload: phone, vehicle, route, round trips
- `GET /api/retell/evening/payload?date=YYYY-MM-DD`
  - returns per-driver call payload: phone, vehicle, wards list (for confirmation)
- `POST /api/retell/webhook/evening-report`
  - accepts driver report (total waste collected + wards visited)
  - applies **Method A (Proportional Allocation)** to split total into ward logs
  - inserts logs → triggers optimization for next-day (optional `final=true`)

---

## 5) ML + Optimization (MVP approach)
### Waste Prediction
- Train per ward using logs time series.
- Features: day_of_week, lag_1, lag_7, rolling_mean_7.
- Model: RandomForestRegressor.
- Hyperparameter tuning: GridSearchCV (small grid for MVP).
- If data is insufficient (< 14 days), fallback to simple average.

### Vehicle Allocation + Route
- Inputs: predicted ward waste, vehicle capacities, ward coordinates.
- Heuristic allocation:
  - Greedy clustering by proximity (haversine distance from panchayat center)
  - Pack wards into vehicles to minimize unused capacity
  - Allow multiple round trips if required
- Route ordering:
  - Nearest-neighbor ordering from panchayat center through assigned wards.

---

## 6) Frontend Flows (MVP)
- **Signup wizard** (3 pages):
  1) Panchayat details + email + password → signup + store token
  2) Add wards (add/remove rows). Each save triggers geocoding.
  3) Add vehicles (add/remove rows) → finish → dashboard

- **Dashboard**:
  - Add Log form: select ward, select vehicle, enter waste collected
  - Logs table (latest first)
  - Call schedule form (HH:MM)
  - Latest route plan card + per-vehicle route list
  - Buttons to “Run Optimization” manually

- **Edit pages** for wards and vehicles: table + edit dialogs + delete.

---

## 7) Testing Approach
- Backend:
  - Auth signup/login/me
  - CRUD wards/vehicles
  - Create logs + list logs
  - Run optimization and verify route docs created
  - Retell webhook inserts logs via proportional allocation

- Frontend:
  - Navigation routing works
  - Private routes blocked when logged out
  - Signup wizard completes and lands on dashboard
  - Add/edit/delete wards/vehicles works
  - Add logs and list updates

---

## 8) Out of Scope (for MVP)
- Actually placing calls via Retell API (we only build payload + webhook ingestion)
- Production-grade optimization (exact VRP solver) and full ML evaluation dashboards
