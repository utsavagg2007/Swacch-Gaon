import React, { useEffect, useMemo, useState } from "react";
import AppShell from "@/components/AppShell";
import ProtectedRoute from "@/components/ProtectedRoute";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useAuth } from "@/context/auth";
import { CalendarClock, PlayCircle, RefreshCcw } from "lucide-react";

function RouteCard({ route }) {
  const ordered = useMemo(() => {
    const map = new Map(route.wards.map((w) => [w.ward_id, w]));
    return route.route_order.map((id) => map.get(id)).filter(Boolean);
  }, [route]);

  return (
    <Card data-testid={`route-card-${route.id}`} className="border-white/10 bg-white/5 text-zinc-50">
      <CardHeader>
        <CardTitle className="text-base" data-testid={`route-title-${route.id}`}>
          Vehicle {route.vehicle_number}
        </CardTitle>
        <CardDescription className="text-zinc-200/70" data-testid={`route-subtitle-${route.id}`}>
          Round trips: {route.round_trips} • Driver: {route.driver_phone}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        <div data-testid={`route-predicted-total-${route.id}`} className="text-sm text-zinc-200/80">
          Predicted total: {route.predicted_total.toFixed(1)}
        </div>
        <ol className="space-y-1 text-sm text-zinc-100">
          {ordered.map((w, idx) => (
            <li key={w.ward_id} data-testid={`route-ward-${route.id}-${idx}`} className="flex items-center justify-between">
              <span className="text-zinc-100">{idx + 1}. {w.ward_name}</span>
              <span className="text-zinc-300/80">{Number(w.predicted_waste || 0).toFixed(1)}</span>
            </li>
          ))}
        </ol>
      </CardContent>
    </Card>
  );
}

export default function Dashboard() {
  const { api, me } = useAuth();

  const [wards, setWards] = useState([]);
  const [vehicles, setVehicles] = useState([]);
  const [logs, setLogs] = useState([]);
  const [routes, setRoutes] = useState([]);

  const [wardId, setWardId] = useState("");
  const [vehicleId, setVehicleId] = useState("");
  const [waste, setWaste] = useState("");

  const [schedule, setSchedule] = useState({ morning_call_time_ist: "06:00", evening_call_time_ist: "19:00" });
  const [retellSetup, setRetellSetup] = useState({ morning_agent_id: "", evening_agent_id: "", from_number: "" });
  const [busyRetell, setBusyRetell] = useState(false);
  const [busyCalls, setBusyCalls] = useState(false);

  const [busyLog, setBusyLog] = useState(false);
  const [busyOpt, setBusyOpt] = useState(false);
  const [busySchedule, setBusySchedule] = useState(false);
  const [error, setError] = useState("");

  const refreshAll = async () => {
    setError("");
    try {
      const [wRes, vRes, lRes, sRes, rRes, retellRes] = await Promise.all([
        api.get("/wards"),
        api.get("/vehicles"),
        api.get("/logs?limit=200"),
        api.get("/settings/call-schedule"),
        api.get("/routes"),
        api.get("/retell/setup").catch(() => ({ data: { morning_agent_id: "", evening_agent_id: "", from_number: "" } })),
      ]);
      setWards(wRes.data);
      setVehicles(vRes.data);
      setLogs(lRes.data);
      setSchedule(sRes.data);
      setRoutes(rRes.data);
      setRetellSetup({
        morning_agent_id: retellRes.data.morning_agent_id || "",
        evening_agent_id: retellRes.data.evening_agent_id || "",
        from_number: retellRes.data.from_number || "",
      });
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to load dashboard data");
    }
  };

  useEffect(() => {
    refreshAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onAddLog = async () => {
    setBusyLog(true);
    setError("");
    try {
      const res = await api.post("/logs", {
        ward_id: wardId,
        vehicle_id: vehicleId,
        waste_collected: Number(waste),
      });
      setLogs((prev) => [res.data, ...prev]);
      setWaste("");
    } catch (err) {
      setError(err?.response?.data?.detail || "Could not add log");
    } finally {
      setBusyLog(false);
    }
  };

  const onSaveSchedule = async () => {
    setBusySchedule(true);
    setError("");
    try {
      const res = await api.put("/settings/call-schedule", schedule);
      setSchedule(res.data);
    } catch (err) {
      setError(err?.response?.data?.detail || "Could not save schedule");
    } finally {
      setBusySchedule(false);
    }
  };

  const onSaveRetell = async () => {
    setBusyRetell(true);
    setError("");
    try {
      const payload = {
        morning_agent_id: retellSetup.morning_agent_id,
        evening_agent_id: retellSetup.evening_agent_id,
        from_number: retellSetup.from_number ? retellSetup.from_number : null,
      };
      const res = await api.put("/retell/setup", payload);
      setRetellSetup({
        morning_agent_id: res.data.morning_agent_id || "",
        evening_agent_id: res.data.evening_agent_id || "",
        from_number: res.data.from_number || "",
      });
    } catch (err) {
      setError(err?.response?.data?.detail || "Could not save Retell setup");
    } finally {
      setBusyRetell(false);
    }
  };

  const onRunCallsNow = async (type) => {
    setBusyCalls(true);
    setError("");
    try {
      const path = type === "morning" ? "/retell/calls/morning/run" : "/retell/calls/evening/run";
      await api.post(path);
    } catch (err) {
      setError(err?.response?.data?.detail || "Could not start calls");
    } finally {
      setBusyCalls(false);
    }
  };

  const onRunOptimization = async () => {
    setBusyOpt(true);
    setError("");
    try {
      await api.post("/optimization/run");
      const rRes = await api.get("/routes");
      setRoutes(rRes.data);
    } catch (err) {
      setError(err?.response?.data?.detail || "Optimization failed");
    } finally {
      setBusyOpt(false);
    }
  };

  return (
    <ProtectedRoute>
      <AppShell variant="private">
        <div data-testid="dashboard-page" className="space-y-6">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h1 data-testid="dashboard-title" className="text-3xl font-semibold tracking-tight">
                Dashboard
              </h1>
              <p data-testid="dashboard-subtitle" className="text-sm text-zinc-200/75">
                {me ? `Signed in as ${me.email}` : ""}
              </p>
            </div>
            <Button
              onClick={refreshAll}
              data-testid="dashboard-refresh-button"
              className="rounded-full bg-white/10 text-zinc-50 ring-1 ring-white/10 hover:bg-white/15"
              size="sm"
            >
              <RefreshCcw className="h-4 w-4" /> Refresh
            </Button>
          </div>

          {error ? (
            <div
              data-testid="dashboard-error"
              className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-100"
            >
              {error}
            </div>
          ) : null}

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
            <div className="lg:col-span-5 space-y-4">
              <Card className="border-white/10 bg-white/5 text-zinc-50" data-testid="dashboard-add-log-card">
                <CardHeader>
                  <CardTitle className="text-base" data-testid="dashboard-add-log-title">Add Log</CardTitle>
                  <CardDescription className="text-zinc-200/70">Ward name, vehicle number, waste collected</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    <Label data-testid="dashboard-log-ward-label">Ward</Label>
                    <Select value={wardId} onValueChange={setWardId}>
                      <SelectTrigger
                        data-testid="dashboard-log-ward-select"
                        className="border-white/15 bg-black/20 text-zinc-50"
                      >
                        <SelectValue placeholder="Select ward" />
                      </SelectTrigger>
                      <SelectContent>
                        {wards.map((w) => (
                          <SelectItem
                            key={w.id}
                            value={w.id}
                            data-testid={`dashboard-log-ward-option-${w.id}`}
                          >
                            {w.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label data-testid="dashboard-log-vehicle-label">Vehicle</Label>
                    <Select value={vehicleId} onValueChange={setVehicleId}>
                      <SelectTrigger
                        data-testid="dashboard-log-vehicle-select"
                        className="border-white/15 bg-black/20 text-zinc-50"
                      >
                        <SelectValue placeholder="Select vehicle" />
                      </SelectTrigger>
                      <SelectContent>
                        {vehicles.map((v) => (
                          <SelectItem
                            key={v.id}
                            value={v.id}
                            data-testid={`dashboard-log-vehicle-option-${v.id}`}
                          >
                            {v.vehicle_number} — {v.driver_name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label data-testid="dashboard-log-waste-label">Waste collected</Label>
                    <Input
                      type="number"
                      min={0}
                      step={0.1}
                      value={waste}
                      onChange={(e) => setWaste(e.target.value)}
                      data-testid="dashboard-log-waste-input"
                      className="border-white/15 bg-black/20 text-zinc-50"
                      placeholder="e.g., 350"
                    />
                  </div>

                  <Button
                    onClick={onAddLog}
                    disabled={busyLog || !wardId || !vehicleId || !waste}
                    data-testid="dashboard-log-submit-button"
                    className="w-full rounded-full bg-emerald-400/20 text-emerald-100 ring-1 ring-emerald-200/25 hover:bg-emerald-400/25"
                  >
                    {busyLog ? "Adding…" : "Add Log"}
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-white/10 bg-white/5 text-zinc-50" data-testid="dashboard-schedule-card">
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2" data-testid="dashboard-schedule-title">
                    <CalendarClock className="h-4 w-4 text-emerald-200" /> Call Schedule (IST)
                  </CardTitle>
                  <CardDescription className="text-zinc-200/70">Set times for morning route briefing and evening log collection</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label data-testid="dashboard-morning-time-label">Morning call time</Label>
                      <Input
                        type="time"
                        value={schedule.morning_call_time_ist}
                        onChange={(e) => setSchedule((s) => ({ ...s, morning_call_time_ist: e.target.value }))}
                        data-testid="dashboard-morning-time-input"
                        className="border-white/15 bg-black/20 text-zinc-50"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label data-testid="dashboard-evening-time-label">Evening call time</Label>
                      <Input
                        type="time"
                        value={schedule.evening_call_time_ist}
                        onChange={(e) => setSchedule((s) => ({ ...s, evening_call_time_ist: e.target.value }))}
                        data-testid="dashboard-evening-time-input"
                        className="border-white/15 bg-black/20 text-zinc-50"
                      />
                    </div>
                  </div>

                  <Button
                    onClick={onSaveSchedule}
                    disabled={busySchedule}
                    data-testid="dashboard-schedule-save-button"
                    className="w-full rounded-full bg-white/10 text-zinc-50 ring-1 ring-white/10 hover:bg-white/15"
                  >
                    {busySchedule ? "Saving…" : "Save schedule"}
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-white/10 bg-white/5 text-zinc-50" data-testid="dashboard-retell-card">
                <CardHeader>
                  <CardTitle className="text-base" data-testid="dashboard-retell-title">Retell AI Setup</CardTitle>
                  <CardDescription className="text-zinc-200/70">
                    Add your Retell Agent IDs (Hindi). Calls will auto-run at your scheduled times.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    <Label data-testid="dashboard-retell-morning-agent-label">Morning agent ID</Label>
                    <Input
                      value={retellSetup.morning_agent_id}
                      onChange={(e) => setRetellSetup((s) => ({ ...s, morning_agent_id: e.target.value }))}
                      data-testid="dashboard-retell-morning-agent-input"
                      className="border-white/15 bg-black/20 text-zinc-50"
                      placeholder="agent_..."
                    />
                  </div>
                  <div className="space-y-2">
                    <Label data-testid="dashboard-retell-evening-agent-label">Evening agent ID</Label>
                    <Input
                      value={retellSetup.evening_agent_id}
                      onChange={(e) => setRetellSetup((s) => ({ ...s, evening_agent_id: e.target.value }))}
                      data-testid="dashboard-retell-evening-agent-input"
                      className="border-white/15 bg-black/20 text-zinc-50"
                      placeholder="agent_..."
                    />
                  </div>
                  <div className="space-y-2">
                    <Label data-testid="dashboard-retell-from-number-label">From number (optional, E.164)</Label>
                    <Input
                      value={retellSetup.from_number}
                      onChange={(e) => setRetellSetup((s) => ({ ...s, from_number: e.target.value }))}
                      data-testid="dashboard-retell-from-number-input"
                      className="border-white/15 bg-black/20 text-zinc-50"
                      placeholder="+91..."
                    />
                  </div>

                  <Button
                    onClick={onSaveRetell}
                    disabled={busyRetell || !retellSetup.morning_agent_id || !retellSetup.evening_agent_id}
                    data-testid="dashboard-retell-save-button"
                    className="w-full rounded-full bg-white/10 text-zinc-50 ring-1 ring-white/10 hover:bg-white/15"
                  >
                    {busyRetell ? "Saving…" : "Save Retell setup"}
                  </Button>

                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                    <Button
                      onClick={() => onRunCallsNow("morning")}
                      disabled={busyCalls}
                      data-testid="dashboard-retell-run-morning-button"
                      className="rounded-full bg-emerald-400/20 text-emerald-100 ring-1 ring-emerald-200/25 hover:bg-emerald-400/25"
                    >
                      {busyCalls ? "Starting…" : "Test: Run morning calls now"}
                    </Button>
                    <Button
                      onClick={() => onRunCallsNow("evening")}
                      disabled={busyCalls}
                      data-testid="dashboard-retell-run-evening-button"
                      className="rounded-full bg-violet-400/15 text-violet-100 ring-1 ring-violet-200/20 hover:bg-violet-400/20"
                    >
                      {busyCalls ? "Starting…" : "Test: Run evening calls now"}
                    </Button>
                  </div>

                  <div data-testid="dashboard-retell-hint" className="text-xs text-zinc-200/70">
                    Requirement: backend must have PUBLIC_BACKEND_URL set so Retell can reach the webhook.
                  </div>
                </CardContent>
              </Card>

            </div>

            <div className="lg:col-span-7 space-y-4">
              <Card className="border-white/10 bg-white/5 text-zinc-50" data-testid="dashboard-optimization-card">
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2" data-testid="dashboard-optimization-title">
                    <PlayCircle className="h-4 w-4 text-sky-200" /> Next-day Prediction & Routes
                  </CardTitle>
                  <CardDescription className="text-zinc-200/70">
                    Uses logs + vehicle capacities to create an optimized plan.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Button
                    onClick={onRunOptimization}
                    disabled={busyOpt}
                    data-testid="dashboard-run-optimization-button"
                    className="rounded-full bg-emerald-400/20 text-emerald-100 ring-1 ring-emerald-200/25 hover:bg-emerald-400/25"
                  >
                    {busyOpt ? "Running…" : "Run optimization for next day"}
                  </Button>

                  <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                    {routes.map((r) => (
                      <RouteCard key={r.id} route={r} />
                    ))}
                    {routes.length === 0 ? (
                      <div
                        data-testid="dashboard-no-routes"
                        className="rounded-2xl border border-white/10 bg-black/10 p-6 text-sm text-zinc-200/75"
                      >
                        No routes yet. Add some logs and run optimization.
                      </div>
                    ) : null}
                  </div>
                </CardContent>
              </Card>

              <Card className="border-white/10 bg-white/5 text-zinc-50" data-testid="dashboard-logs-card">
                <CardHeader>
                  <CardTitle className="text-base" data-testid="dashboard-logs-title">Recent Logs</CardTitle>
                  <CardDescription className="text-zinc-200/70">Latest 200 entries</CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead data-testid="logs-th-date">Date</TableHead>
                        <TableHead data-testid="logs-th-vehicle">Vehicle</TableHead>
                        <TableHead data-testid="logs-th-waste">Waste</TableHead>
                        <TableHead data-testid="logs-th-source">Source</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {logs.map((l) => (
                        <TableRow key={l.id} data-testid={`logs-row-${l.id}`}>
                          <TableCell data-testid={`logs-date-${l.id}`}>{l.log_date}</TableCell>
                          <TableCell data-testid={`logs-vehicle-${l.id}`}>{l.vehicle_number}</TableCell>
                          <TableCell data-testid={`logs-waste-${l.id}`}>{Number(l.waste_collected).toFixed(1)}</TableCell>
                          <TableCell data-testid={`logs-source-${l.id}`}>{l.source}</TableCell>
                        </TableRow>
                      ))}
                      {logs.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan={4} data-testid="logs-empty" className="text-zinc-200/70">
                            No logs yet.
                          </TableCell>
                        </TableRow>
                      ) : null}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </AppShell>
    </ProtectedRoute>
  );
}
