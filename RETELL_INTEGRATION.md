# Swachh Gaon — Retell AI Integration (Detailed)

This guide explains **exactly** how to integrate Retell into the current Swachh Gaon app.

You chose:
- Hindi agents
- No webhook signature verification for MVP
- Driver phone numbers stored in **E.164** format
- Calls should be **auto-scheduled** using the times set on Dashboard

> Important: Please do **not** put your Retell API key in the frontend. It stays only in backend env.

---

## 0) What is already implemented in your app
### Backend (FastAPI)
We already added:
1) **Retell setup storage** (per panchayat)
- `GET /api/retell/setup`
- `PUT /api/retell/setup`

2) **Call triggers**
- `POST /api/retell/calls/morning/run` (creates calls to all drivers for today’s route)
- `POST /api/retell/calls/evening/run` (creates calls to all drivers to collect today’s logs)

3) **Webhook receiver** (Retell → our backend)
- `POST /api/retell/webhook/call-event`
  - stores raw events into MongoDB collection `retell_events`
  - if it finds extracted fields (total waste + wards visited), it calls our existing logic to insert logs

4) **Scheduler**
- Runs every minute and checks if the current IST time matches:
  - `settings.morning_call_time_ist` (default `06:00`)
  - `settings.evening_call_time_ist` (default `19:00`)
- If matched, it triggers the respective calls for that panchayat.

### Frontend (Dashboard)
We added a “Retell AI Setup” card:
- Save morning agent ID / evening agent ID / optional from-number
- “Test: Run morning calls now” and “Test: Run evening calls now” buttons

---

## 1) REQUIRED environment variable (must set)
Retell needs a publicly reachable webhook URL.

### Set in `/app/backend/.env`
- `PUBLIC_BACKEND_URL` = your app base URL (no trailing slash)

Example for this environment:
```
PUBLIC_BACKEND_URL="https://rural-waste-optimize.preview.emergentagent.com"
```
Then restart backend:
```
sudo supervisorctl restart backend
```

---

## 2) Create Retell Agents (Morning + Evening)
Because you said “No agents yet”, do this in Retell dashboard:

### A) Morning Agent (Hindi)
- Purpose: Tell driver the route + round trips for today
- Paste prompt from: `/app/backend/retell_prompts.md` (section A)

### B) Evening Agent (Hindi)
- Purpose: Ask driver for total waste + wards visited
- Paste prompt from: `/app/backend/retell_prompts.md` (section B)

When you create each agent, Retell will give you an `agent_id` like `agent_...`.
Keep both IDs.

---

## 3) Configure Retell Webhook (Retell → Swachh Gaon)
In Retell, set the webhook URL to:
```
{PUBLIC_BACKEND_URL}/api/retell/webhook/call-event
```
Example:
```
https://rural-waste-optimize.preview.emergentagent.com/api/retell/webhook/call-event
```

For MVP, we are **not** verifying signatures.

---

## 4) IMPORTANT: How we pass “route data” to the agent
When we create a call, we send `metadata` that the agent can read.

### Morning call metadata fields we send
- `vehicle_number`
- `round_trips`
- `route_text`
- `plan_date`
- `panchayat_name`

The agent prompt should speak these.

### Evening call metadata fields we send
- `vehicle_number`
- `date`
- `wards_expected` (array of ward_id + ward_name)

---

## 5) How evening data becomes logs (ward-wise)
Your requirement:
> Retell collects **route-level total waste** and we split into ward-wise logs using **Method A (Proportional Allocation)**.

We do exactly this:
- webhook receives total waste + wards visited
- backend distributes total across wards using weights from predicted waste in stored route plan
- inserts records into `logs` collection as `source="retell"`
- triggers next-day optimization (already implemented)

---

## 6) How to configure “extracted fields” in Retell (so webhook contains data)
Retell can send structured fields in the webhook payload depending on configuration.

You should configure the Evening Agent to capture:
- `total_waste_collected` (number)
- `wards_visited` (array of strings)

### Recommended approach
In Retell, configure call analysis / extracted fields (sometimes called **custom_fields**) to output:
- `total_waste_collected`
- `wards_visited`

Our webhook parser looks for values in:
- `call.call_analysis.custom_fields`
- or `call.total_waste_collected` directly

If Retell uses different field names, tell me the exact webhook JSON and I’ll map it precisely.

---

## 7) End-to-end checklist (no-error setup)
1) Make sure all driver phones are saved as E.164 (`+91...`).
2) Set `PUBLIC_BACKEND_URL` and restart backend.
3) Create 2 Retell agents (Morning + Evening) using prompts in `retell_prompts.md`.
4) In Dashboard → Retell AI Setup:
   - paste Morning Agent ID
   - paste Evening Agent ID
   - optional: From number
   - click Save
5) Test:
   - Click “Test: Run morning calls now”
   - Click “Test: Run evening calls now”
6) Confirm webhook hits:
   - backend stores raw events in `retell_events`
   - logs appear in Dashboard after evening call

---

## 8) Notes / Safety
- Keep Retell API key only in backend.
- For production you should enable webhook signature verification.

