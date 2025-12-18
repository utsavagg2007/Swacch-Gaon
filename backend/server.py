from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime, timezone, timedelta, date
from zoneinfo import ZoneInfo
from pathlib import Path
import os
import uuid
import logging
import re
import anyio
import requests

# ML
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change")
JWT_ALG = "HS256"
TOKEN_TTL_MIN = int(os.environ.get("JWT_TTL_MIN", "10080"))  # 7 days

security = HTTPBearer(auto_error=False)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _date_ist(d: Optional[date] = None) -> date:
    ist = ZoneInfo("Asia/Kolkata")
    now_ist = datetime.now(ist)
    return d or now_ist.date()


async def _to_thread(fn, *args, **kwargs):
    return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))


async def geocode_address(address: str) -> Optional[Dict[str, float]]:
    """Geocode via OpenStreetMap Nominatim (no API key)."""

    def _call():
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": address, "format": "json", "limit": 1}
        headers = {"User-Agent": "EmergentRuralWasteOptimizer/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}

    try:
        return await _to_thread(_call)
    except Exception:
        return None


# --- Auth helpers ---
try:
    from jose import jwt
except Exception:
    import jwt  # type: ignore

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)


def verify_password(pw: str, pw_hash: str) -> bool:
    try:
        return pwd_context.verify(pw, pw_hash)
    except Exception:
        return False


def create_access_token(panchayat_id: str) -> str:
    exp = _now_utc() + timedelta(minutes=TOKEN_TTL_MIN)
    payload = {"sub": panchayat_id, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def _jwt_decode(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])


async def get_current_panchayat(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    token = credentials.credentials
    try:
        payload = _jwt_decode(token)
        pid = payload.get("sub")
        if not pid:
            raise ValueError("Missing sub")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    p = await db.panchayats.find_one({"_id": pid}, {"_id": 1, "email": 1, "name": 1, "address": 1, "lat": 1, "lon": 1})
    if not p:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return p


# --- Models ---
class AuthSignupIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    email: EmailStr
    name: str = Field(min_length=2)
    address: str = Field(min_length=5)
    password: str = Field(min_length=6)


class AuthLoginIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    email: EmailStr
    password: str


class AuthOut(BaseModel):
    token: str


class PanchayatOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(alias="_id")
    email: EmailStr
    name: str
    address: str
    lat: Optional[float] = None
    lon: Optional[float] = None


class PanchayatUpdateIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = None
    address: Optional[str] = None


class WardIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=2)
    address: str = Field(min_length=5)


class WardOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(alias="_id")
    panchayat_id: str
    name: str
    address: str
    lat: Optional[float] = None
    lon: Optional[float] = None


class VehicleIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    driver_name: str = Field(min_length=2)
    driver_phone: str = Field(min_length=7)
    vehicle_number: str = Field(min_length=3)
    capacity: float = Field(gt=0)


class VehicleOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(alias="_id")
    panchayat_id: str
    driver_name: str
    driver_phone: str
    vehicle_number: str
    capacity: float


class LogIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ward_id: str
    vehicle_id: str
    waste_collected: float = Field(gt=0)
    log_date: Optional[str] = None  # YYYY-MM-DD


class LogOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(alias="_id")
    panchayat_id: str
    ward_id: str
    vehicle_id: str
    vehicle_number: str
    waste_collected: float
    log_date: str
    source: str
    created_at: str


class CallScheduleOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    morning_call_time_ist: str = "06:00"
    evening_call_time_ist: str = "19:00"


class CallScheduleIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    morning_call_time_ist: str
    evening_call_time_ist: str


class OptimizationRunOut(BaseModel):
    plan_date: str
    routes_created: int


class RoutePlanOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(alias="_id")
    panchayat_id: str
    plan_date: str
    vehicle_id: str
    vehicle_number: str
    driver_phone: str
    round_trips: int
    wards: List[Dict[str, Any]]
    predicted_total: float
    route_order: List[str]
    created_at: str


class RetellEveningWebhookIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    vehicle_number: str
    driver_phone: Optional[str] = None
    date: str  # YYYY-MM-DD
    total_waste_collected: float = Field(gt=0)
    wards_visited: List[str]  # ward_id or ward_name
    final: bool = True


# --- ML / optimization helpers ---

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # df columns: ward_id, log_date (datetime), waste
    df = df.sort_values(["ward_id", "log_date"]).copy()
    df["dow"] = df["log_date"].dt.dayofweek
    df["lag_1"] = df.groupby("ward_id")["waste"].shift(1)
    df["lag_7"] = df.groupby("ward_id")["waste"].shift(7)
    df["roll_7"] = df.groupby("ward_id")["waste"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    return df


async def train_and_predict_next_day(panchayat_id: str, target_date: date) -> Dict[str, float]:
    logs = await db.logs.find({"panchayat_id": panchayat_id}, {"_id": 0, "ward_id": 1, "waste_collected": 1, "log_date": 1}).to_list(100000)
    if not logs:
        return {}

    df = pd.DataFrame(logs)
    df = df.rename(columns={"waste_collected": "waste"})
    df["log_date"] = pd.to_datetime(df["log_date"], errors="coerce")
    df = df.dropna(subset=["log_date", "waste", "ward_id"])

    df_feat = build_features(df)
    preds: Dict[str, float] = {}

    for ward_id, g in df_feat.groupby("ward_id"):
        g = g.dropna(subset=["lag_1", "lag_7", "roll_7", "dow"])
        if len(g) < 10:
            # fallback average
            base = float(df[df["ward_id"] == ward_id]["waste"].tail(14).mean())
            preds[str(ward_id)] = max(0.0, base if not np.isnan(base) else 0.0)
            continue

        X = g[["dow", "lag_1", "lag_7", "roll_7"]].astype(float)
        y = g["waste"].astype(float)

        tscv = TimeSeriesSplit(n_splits=3)
        model = RandomForestRegressor(random_state=42)
        grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 6, 10],
            "min_samples_split": [2, 5],
        }
        try:
            search = GridSearchCV(model, grid, cv=tscv, n_jobs=-1, scoring="neg_mean_absolute_error")
            search.fit(X, y)
            best = search.best_estimator_
        except Exception:
            best = model
            best.fit(X, y)

        # build one row for target_date using latest known lags
        last = df[df["ward_id"] == ward_id].sort_values("log_date").copy()
        last = last.tail(14)
        if len(last) == 0:
            preds[str(ward_id)] = 0.0
            continue

        dow = pd.Timestamp(target_date).dayofweek
        lag_1 = float(last["waste"].iloc[-1])
        lag_7 = float(last["waste"].iloc[-7]) if len(last) >= 7 else float(last["waste"].mean())
        roll_7 = float(last["waste"].tail(7).mean())

        yhat = float(best.predict([[dow, lag_1, lag_7, roll_7]])[0])
        preds[str(ward_id)] = max(0.0, yhat)

    return preds


def nearest_neighbor_route(start_lat: float, start_lon: float, wards: List[dict]) -> List[str]:
    remaining = wards.copy()
    route: List[str] = []
    cur_lat, cur_lon = start_lat, start_lon
    while remaining:
        remaining.sort(key=lambda w: haversine_km(cur_lat, cur_lon, w.get("lat") or 0.0, w.get("lon") or 0.0))
        nxt = remaining.pop(0)
        route.append(nxt["ward_id"])
        cur_lat, cur_lon = nxt.get("lat") or cur_lat, nxt.get("lon") or cur_lon
    return route


def allocate_vehicles(
    panchayat_center: dict,
    wards: List[dict],
    vehicles: List[dict],
    ward_predictions: Dict[str, float],
) -> List[dict]:
    """Allocate wards across ALL available vehicles.

    Priority:
    1) Use all drivers/vehicles (assign at least one ward per vehicle when possible)
    2) Then minimize additional round trips and unused capacity.

    Note: If prediction is missing (no logs yet), we use a small default demand so
    routing still works.
    """

    center_lat = float(panchayat_center.get("lat") or 0.0)
    center_lon = float(panchayat_center.get("lon") or 0.0)

    prepared_wards = []
    for w in wards:
        pred = ward_predictions.get(w["_id"], None)
        demand = float(pred) if pred is not None else 1.0
        demand = max(0.0, demand)
        prepared_wards.append(
            {
                **w,
                "predicted_waste": demand,
                "dist": haversine_km(center_lat, center_lon, w.get("lat") or 0.0, w.get("lon") or 0.0),
            }
        )

    wards_sorted = sorted(prepared_wards, key=lambda w: w["dist"])

    vehicle_states = [
        {
            **v,
            "assigned": [],
            "assigned_total": 0.0,
        }
        for v in vehicles
    ]

    if not vehicle_states:
        return []

    # 1) Seed: give each vehicle at least one ward (if enough wards)
    remaining = wards_sorted.copy()
    for v in vehicle_states:
        if not remaining:
            break
        w = remaining.pop(0)
        v["assigned"].append(
            {
                "ward_id": w["_id"],
                "ward_name": w["name"],
                "lat": w.get("lat"),
                "lon": w.get("lon"),
                "predicted_waste": float(w["predicted_waste"]),
            }
        )
        v["assigned_total"] += float(w["predicted_waste"])

    # helper cost
    def _round_trips(load: float, cap: float) -> int:
        if load <= 0:
            return 0
        return int(np.ceil(load / cap))

    # 2) Distribute remaining wards by minimizing extra trips and unused capacity
    for w in remaining:
        demand = float(w["predicted_waste"])

        best_idx = None
        best_cost = None

        for idx, v in enumerate(vehicle_states):
            cap = float(v.get("capacity") or 1.0)
            cur_load = float(v.get("assigned_total") or 0.0)
            cur_trips = _round_trips(cur_load, cap)

            new_load = cur_load + demand
            new_trips = _round_trips(new_load, cap)

            inc_trips = new_trips - cur_trips
            unused = (new_trips * cap) - new_load if new_trips > 0 else cap

            # Lower is better: prioritize no extra trips, then lower unused capacity, then lower total trips,
            # then keep loads balanced
            cost = (
                inc_trips,
                unused,
                new_trips,
                new_load / cap,
            )

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_idx = idx

        chosen = vehicle_states[best_idx] if best_idx is not None else vehicle_states[0]
        chosen["assigned"].append(
            {
                "ward_id": w["_id"],
                "ward_name": w["name"],
                "lat": w.get("lat"),
                "lon": w.get("lon"),
                "predicted_waste": demand,
            }
        )
        chosen["assigned_total"] += demand

    # 3) Build a plan for every vehicle (even if it has 0 wards)
    plans = []
    for v in vehicle_states:
        cap = float(v.get("capacity") or 1.0)
        total = float(v.get("assigned_total") or 0.0)
        round_trips = _round_trips(total, cap)
        assigned = v.get("assigned", [])
        order = nearest_neighbor_route(center_lat, center_lon, assigned) if assigned else []

        plans.append(
            {
                "vehicle_id": v["_id"],
                "vehicle_number": v.get("vehicle_number"),
                "driver_phone": v.get("driver_phone"),
                "round_trips": int(round_trips),
                "wards": assigned,
                "predicted_total": float(total),
                "route_order": order,
            }
        )

    return plans


# --- App / Router ---
app = FastAPI()
api_router = APIRouter(prefix="/api")


@api_router.get("/", tags=["health"])
async def root():
    return {"message": "Rural Waste Optimization API"}


# --- Auth endpoints ---
@api_router.post("/auth/signup", response_model=AuthOut, tags=["auth"])
async def signup(payload: AuthSignupIn):
    existing = await db.panchayats.find_one({"email": payload.email.lower()})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    coords = await geocode_address(payload.address)
    doc = {
        "_id": _uuid(),
        "email": payload.email.lower(),
        "name": payload.name.strip(),
        "address": payload.address.strip(),
        "lat": coords["lat"] if coords else None,
        "lon": coords["lon"] if coords else None,
        "password_hash": hash_password(payload.password),
        "created_at": _iso(_now_utc()),
    }
    await db.panchayats.insert_one(doc)
    return AuthOut(token=create_access_token(doc["_id"]))


@api_router.post("/auth/login", response_model=AuthOut, tags=["auth"])
async def login(payload: AuthLoginIn):
    p = await db.panchayats.find_one({"email": payload.email.lower()})
    if not p:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    if not verify_password(payload.password, p.get("password_hash", "")):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return AuthOut(token=create_access_token(p["_id"]))


@api_router.get(
    "/auth/me",
    response_model=PanchayatOut,
    response_model_by_alias=False,
    tags=["auth"],
)
async def me(p=Depends(get_current_panchayat)):
    return p


# --- Panchayat ---
@api_router.get(
    "/panchayat",
    response_model=PanchayatOut,
    response_model_by_alias=False,
    tags=["panchayat"],
)
async def get_panchayat(p=Depends(get_current_panchayat)):
    return p


@api_router.put(
    "/panchayat",
    response_model=PanchayatOut,
    response_model_by_alias=False,
    tags=["panchayat"],
)
async def update_panchayat(payload: PanchayatUpdateIn, p=Depends(get_current_panchayat)):
    update: Dict[str, Any] = {}
    if payload.name is not None:
        update["name"] = payload.name.strip()
    if payload.address is not None:
        update["address"] = payload.address.strip()
        coords = await geocode_address(update["address"])
        if coords:
            update["lat"] = coords["lat"]
            update["lon"] = coords["lon"]

    if not update:
        return p

    await db.panchayats.update_one({"_id": p["_id"]}, {"$set": update})
    updated = await db.panchayats.find_one({"_id": p["_id"]}, {"password_hash": 0})
    return updated


# --- Wards ---
@api_router.get(
    "/wards",
    response_model=List[WardOut],
    response_model_by_alias=False,
    tags=["wards"],
)
async def list_wards(p=Depends(get_current_panchayat)):
    wards = await db.wards.find({"panchayat_id": p["_id"]}, {"_id": 1, "panchayat_id": 1, "name": 1, "address": 1, "lat": 1, "lon": 1}).to_list(10000)
    return wards


@api_router.post(
    "/wards",
    response_model=WardOut,
    response_model_by_alias=False,
    tags=["wards"],
)
async def create_ward(payload: WardIn, p=Depends(get_current_panchayat)):
    coords = await geocode_address(payload.address)
    doc = {
        "_id": _uuid(),
        "panchayat_id": p["_id"],
        "name": payload.name.strip(),
        "address": payload.address.strip(),
        "lat": coords["lat"] if coords else None,
        "lon": coords["lon"] if coords else None,
        "created_at": _iso(_now_utc()),
    }
    await db.wards.insert_one(doc)
    return doc


@api_router.post(
    "/wards/bulk",
    response_model=List[WardOut],
    response_model_by_alias=False,
    tags=["wards"],
)
async def create_wards_bulk(payload: List[WardIn], p=Depends(get_current_panchayat)):
    docs = []
    for w in payload:
        coords = await geocode_address(w.address)
        docs.append(
            {
                "_id": _uuid(),
                "panchayat_id": p["_id"],
                "name": w.name.strip(),
                "address": w.address.strip(),
                "lat": coords["lat"] if coords else None,
                "lon": coords["lon"] if coords else None,
                "created_at": _iso(_now_utc()),
            }
        )
    if docs:
        await db.wards.insert_many(docs)
    return docs


@api_router.put(
    "/wards/{ward_id}",
    response_model=WardOut,
    response_model_by_alias=False,
    tags=["wards"],
)
async def update_ward(ward_id: str, payload: WardIn, p=Depends(get_current_panchayat)):
    coords = await geocode_address(payload.address)
    update = {
        "name": payload.name.strip(),
        "address": payload.address.strip(),
        "lat": coords["lat"] if coords else None,
        "lon": coords["lon"] if coords else None,
    }
    res = await db.wards.update_one({"_id": ward_id, "panchayat_id": p["_id"]}, {"$set": update})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Ward not found")
    doc = await db.wards.find_one({"_id": ward_id}, {"_id": 1, "panchayat_id": 1, "name": 1, "address": 1, "lat": 1, "lon": 1})
    return doc


@api_router.delete("/wards/{ward_id}", tags=["wards"])
async def delete_ward(ward_id: str, p=Depends(get_current_panchayat)):
    await db.wards.delete_one({"_id": ward_id, "panchayat_id": p["_id"]})
    # also cleanup logs/routes references
    await db.logs.delete_many({"panchayat_id": p["_id"], "ward_id": ward_id})
    return {"ok": True}


# --- Vehicles ---
@api_router.get(
    "/vehicles",
    response_model=List[VehicleOut],
    response_model_by_alias=False,
    tags=["vehicles"],
)
async def list_vehicles(p=Depends(get_current_panchayat)):
    vehicles = await db.vehicles.find({"panchayat_id": p["_id"]}, {"_id": 1, "panchayat_id": 1, "driver_name": 1, "driver_phone": 1, "vehicle_number": 1, "capacity": 1}).to_list(10000)
    return vehicles


@api_router.post(
    "/vehicles",
    response_model=VehicleOut,
    response_model_by_alias=False,
    tags=["vehicles"],
)
async def create_vehicle(payload: VehicleIn, p=Depends(get_current_panchayat)):
    doc = {
        "_id": _uuid(),
        "panchayat_id": p["_id"],
        "driver_name": payload.driver_name.strip(),
        "driver_phone": re.sub(r"\s+", "", payload.driver_phone),
        "vehicle_number": payload.vehicle_number.strip().upper(),
        "capacity": float(payload.capacity),
        "created_at": _iso(_now_utc()),
    }
    await db.vehicles.insert_one(doc)
    return doc


@api_router.post(
    "/vehicles/bulk",
    response_model=List[VehicleOut],
    response_model_by_alias=False,
    tags=["vehicles"],
)
async def create_vehicles_bulk(payload: List[VehicleIn], p=Depends(get_current_panchayat)):
    docs = []
    for v in payload:
        docs.append(
            {
                "_id": _uuid(),
                "panchayat_id": p["_id"],
                "driver_name": v.driver_name.strip(),
                "driver_phone": re.sub(r"\s+", "", v.driver_phone),
                "vehicle_number": v.vehicle_number.strip().upper(),
                "capacity": float(v.capacity),
                "created_at": _iso(_now_utc()),
            }
        )
    if docs:
        await db.vehicles.insert_many(docs)
    return docs


@api_router.put(
    "/vehicles/{vehicle_id}",
    response_model=VehicleOut,
    response_model_by_alias=False,
    tags=["vehicles"],
)
async def update_vehicle(vehicle_id: str, payload: VehicleIn, p=Depends(get_current_panchayat)):
    update = {
        "driver_name": payload.driver_name.strip(),
        "driver_phone": re.sub(r"\s+", "", payload.driver_phone),
        "vehicle_number": payload.vehicle_number.strip().upper(),
        "capacity": float(payload.capacity),
    }
    res = await db.vehicles.update_one({"_id": vehicle_id, "panchayat_id": p["_id"]}, {"$set": update})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    doc = await db.vehicles.find_one({"_id": vehicle_id}, {"_id": 1, "panchayat_id": 1, "driver_name": 1, "driver_phone": 1, "vehicle_number": 1, "capacity": 1})
    return doc


@api_router.delete("/vehicles/{vehicle_id}", tags=["vehicles"])
async def delete_vehicle(vehicle_id: str, p=Depends(get_current_panchayat)):
    await db.vehicles.delete_one({"_id": vehicle_id, "panchayat_id": p["_id"]})
    await db.logs.delete_many({"panchayat_id": p["_id"], "vehicle_id": vehicle_id})
    return {"ok": True}


# --- Logs ---
@api_router.get(
    "/logs",
    response_model=List[LogOut],
    response_model_by_alias=False,
    tags=["logs"],
)
async def list_logs(limit: int = 200, p=Depends(get_current_panchayat)):
    limit = min(max(limit, 1), 1000)
    logs = (
        await db.logs.find({"panchayat_id": p["_id"]}, {"_id": 1, "panchayat_id": 1, "ward_id": 1, "vehicle_id": 1, "vehicle_number": 1, "waste_collected": 1, "log_date": 1, "source": 1, "created_at": 1})
        .sort("created_at", -1)
        .to_list(limit)
    )
    return logs


@api_router.post(
    "/logs",
    response_model=LogOut,
    response_model_by_alias=False,
    tags=["logs"],
)
async def create_log(payload: LogIn, p=Depends(get_current_panchayat)):
    ward = await db.wards.find_one({"_id": payload.ward_id, "panchayat_id": p["_id"]}, {"_id": 1})
    vehicle = await db.vehicles.find_one({"_id": payload.vehicle_id, "panchayat_id": p["_id"]}, {"_id": 1, "vehicle_number": 1})
    if not ward:
        raise HTTPException(status_code=400, detail="Invalid ward")
    if not vehicle:
        raise HTTPException(status_code=400, detail="Invalid vehicle")

    log_date = payload.log_date or _date_ist().isoformat()
    doc = {
        "_id": _uuid(),
        "panchayat_id": p["_id"],
        "ward_id": payload.ward_id,
        "vehicle_id": payload.vehicle_id,
        "vehicle_number": vehicle["vehicle_number"],
        "waste_collected": float(payload.waste_collected),
        "log_date": log_date,
        "source": "manual",
        "created_at": _iso(_now_utc()),
    }
    await db.logs.insert_one(doc)
    return doc


# --- Settings (call schedules) ---
@api_router.get("/settings/call-schedule", response_model=CallScheduleOut, tags=["settings"])
async def get_call_schedule(p=Depends(get_current_panchayat)):
    doc = await db.settings.find_one({"panchayat_id": p["_id"]}, {"_id": 0, "morning_call_time_ist": 1, "evening_call_time_ist": 1})
    if not doc:
        return CallScheduleOut()
    return CallScheduleOut(**doc)


@api_router.put("/settings/call-schedule", response_model=CallScheduleOut, tags=["settings"])
async def set_call_schedule(payload: CallScheduleIn, p=Depends(get_current_panchayat)):
    await db.settings.update_one(
        {"panchayat_id": p["_id"]},
        {
            "$set": {
                "panchayat_id": p["_id"],
                "morning_call_time_ist": payload.morning_call_time_ist,
                "evening_call_time_ist": payload.evening_call_time_ist,
                "updated_at": _iso(_now_utc()),
            }
        },
        upsert=True,
    )
    return CallScheduleOut(**payload.model_dump())


# --- Optimization + Routes ---
@api_router.post(
    "/optimization/run",
    response_model=OptimizationRunOut,
    response_model_by_alias=False,
    tags=["optimization"],
)
async def run_optimization(p=Depends(get_current_panchayat)):
    # predict for next day IST
    tomorrow = _date_ist() + timedelta(days=1)

    wards = await db.wards.find({"panchayat_id": p["_id"]}).to_list(10000)
    vehicles = await db.vehicles.find({"panchayat_id": p["_id"]}).to_list(10000)

    if not wards or not vehicles:
        raise HTTPException(status_code=400, detail="Please add wards and vehicles first")

    ward_preds = await train_and_predict_next_day(p["_id"], tomorrow)

    # clear existing plans for that date
    await db.routes.delete_many({"panchayat_id": p["_id"], "plan_date": tomorrow.isoformat()})

    plans = allocate_vehicles(p, wards, vehicles, ward_preds)
    now_iso = _iso(_now_utc())

    route_docs = []
    for pl in plans:
        route_docs.append(
            {
                "_id": _uuid(),
                "panchayat_id": p["_id"],
                "plan_date": tomorrow.isoformat(),
                **pl,
                "created_at": now_iso,
            }
        )

    if route_docs:
        await db.routes.insert_many(route_docs)

    return OptimizationRunOut(plan_date=tomorrow.isoformat(), routes_created=len(route_docs))


@api_router.get(
    "/routes",
    response_model=List[RoutePlanOut],
    response_model_by_alias=False,
    tags=["routes"],
)
async def list_routes(date: Optional[str] = None, p=Depends(get_current_panchayat)):
    plan_date = date or (_date_ist() + timedelta(days=1)).isoformat()
    routes = await db.routes.find({"panchayat_id": p["_id"], "plan_date": plan_date}).to_list(10000)
    return routes


# --- Retell-ready endpoints ---
@api_router.get("/retell/morning/payload", tags=["retell"])
async def retell_morning_payload(date: Optional[str] = None, p=Depends(get_current_panchayat)):
    plan_date = date or (_date_ist() + timedelta(days=1)).isoformat()
    routes = await db.routes.find({"panchayat_id": p["_id"], "plan_date": plan_date}).to_list(10000)
    if not routes:
        raise HTTPException(status_code=404, detail="No routes found for date")

    payloads = []
    for r in routes:
        ward_by_id = {w["ward_id"]: w for w in r.get("wards", [])}
        ordered_names = [ward_by_id.get(wid, {}).get("ward_name", wid) for wid in r.get("route_order", [])]
        route_text = " → ".join([n for n in ordered_names if n])
        payloads.append(
            {
                "to_phone": r.get("driver_phone"),
                "vehicle_number": r.get("vehicle_number"),
                "round_trips": r.get("round_trips"),
                "plan_date": plan_date,
                "route": route_text,
            }
        )
    return {"plan_date": plan_date, "drivers": payloads}


@api_router.get("/retell/evening/payload", tags=["retell"])
async def retell_evening_payload(date: Optional[str] = None, p=Depends(get_current_panchayat)):
    plan_date = date or (_date_ist() + timedelta(days=1)).isoformat()
    routes = await db.routes.find({"panchayat_id": p["_id"], "plan_date": plan_date}).to_list(10000)
    if not routes:
        raise HTTPException(status_code=404, detail="No routes found for date")

    payloads = []
    for r in routes:
        wards = r.get("wards", [])
        payloads.append(
            {
                "to_phone": r.get("driver_phone"),
                "vehicle_number": r.get("vehicle_number"),
                "plan_date": plan_date,
                "wards_expected": [{"ward_id": w.get("ward_id"), "ward_name": w.get("ward_name")} for w in wards],
            }
        )
    return {"plan_date": plan_date, "drivers": payloads}


@api_router.post("/retell/webhook/evening-report", tags=["retell"])
async def retell_evening_webhook(payload: RetellEveningWebhookIn):
    # This endpoint is intentionally unauthenticated to allow Retell webhook calls.
    # In production you should validate a Retell signature header.

    # Find panchayat by matching route doc (vehicle_number + date)
    route = await db.routes.find_one({"plan_date": payload.date, "vehicle_number": payload.vehicle_number})
    if not route:
        raise HTTPException(status_code=404, detail="Route not found for vehicle/date")

    pid = route["panchayat_id"]

    # Map wards_visited (ids or names) to ward ids
    wards_all = await db.wards.find({"panchayat_id": pid}).to_list(10000)
    by_id = {w["_id"]: w for w in wards_all}
    by_name = {w["name"].strip().lower(): w for w in wards_all}

    visited_ids: List[str] = []
    for token in payload.wards_visited:
        t = str(token).strip()
        if t in by_id:
            visited_ids.append(t)
        else:
            w = by_name.get(t.lower())
            if w:
                visited_ids.append(w["_id"])

    visited_ids = list(dict.fromkeys(visited_ids))
    if not visited_ids:
        raise HTTPException(status_code=400, detail="No valid wards_visited")

    # Method A: proportional allocation based on predicted_waste from route doc
    route_wards = {w.get("ward_id"): float(w.get("predicted_waste") or 0.0) for w in route.get("wards", [])}
    weights = [max(0.0, route_wards.get(wid, 0.0)) for wid in visited_ids]
    s = float(sum(weights))
    if s <= 0:
        weights = [1.0 for _ in visited_ids]
        s = float(sum(weights))

    # Find vehicle id by vehicle number
    vehicle = await db.vehicles.find_one({"panchayat_id": pid, "vehicle_number": payload.vehicle_number})
    vehicle_id = vehicle["_id"] if vehicle else route.get("vehicle_id")

    now_iso = _iso(_now_utc())
    docs = []
    for wid, wgt in zip(visited_ids, weights):
        alloc = float(payload.total_waste_collected) * (float(wgt) / s)
        docs.append(
            {
                "_id": _uuid(),
                "panchayat_id": pid,
                "ward_id": wid,
                "vehicle_id": vehicle_id,
                "vehicle_number": payload.vehicle_number,
                "waste_collected": float(alloc),
                "log_date": payload.date,
                "source": "retell",
                "created_at": now_iso,
            }
        )

    if docs:
        await db.logs.insert_many(docs)

    # Optionally trigger next-day optimization once final data arrives
    if payload.final:
        # derive panchayat doc shape expected by allocation
        p_doc = await db.panchayats.find_one({"_id": pid}, {"_id": 1, "lat": 1, "lon": 1})
        if p_doc:
            tomorrow = (datetime.fromisoformat(payload.date).date() + timedelta(days=1))
            wards = await db.wards.find({"panchayat_id": pid}).to_list(10000)
            vehicles = await db.vehicles.find({"panchayat_id": pid}).to_list(10000)
            if wards and vehicles:
                ward_preds = await train_and_predict_next_day(pid, tomorrow)
                await db.routes.delete_many({"panchayat_id": pid, "plan_date": tomorrow.isoformat()})
                plans = allocate_vehicles(p_doc, wards, vehicles, ward_preds)
                route_docs = []
                for pl in plans:
                    route_docs.append(
                        {
                            "_id": _uuid(),
                            "panchayat_id": pid,
                            "plan_date": tomorrow.isoformat(),
                            **pl,
                            "created_at": now_iso,
                        }
                    )
                if route_docs:
                    await db.routes.insert_many(route_docs)

    return {"ok": True, "inserted_logs": len(docs)}


# include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
