import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import AppShell from "@/components/AppShell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Plus, Trash2, ArrowLeft, CheckCircle2 } from "lucide-react";
import { useAuth } from "@/context/auth";

function VehicleRow({ idx, vehicle, onChange, onRemove, canRemove }) {
  return (
    <div
      data-testid={`signup-vehicle-row-${idx}`}
      className="grid grid-cols-1 gap-3 rounded-xl border border-white/10 bg-black/15 p-4 md:grid-cols-12"
    >
      <div className="md:col-span-3">
        <Label data-testid={`signup-vehicle-driver-label-${idx}`}>Driver name</Label>
        <Input
          value={vehicle.driver_name}
          onChange={(e) => onChange(idx, { ...vehicle, driver_name: e.target.value })}
          data-testid={`signup-vehicle-driver-input-${idx}`}
          className="mt-2 border-white/15 bg-black/20 text-zinc-50"
          required
        />
      </div>
      <div className="md:col-span-3">
        <Label data-testid={`signup-vehicle-phone-label-${idx}`}>Driver phone</Label>
        <Input
          value={vehicle.driver_phone}
          onChange={(e) => onChange(idx, { ...vehicle, driver_phone: e.target.value })}
          data-testid={`signup-vehicle-phone-input-${idx}`}
          className="mt-2 border-white/15 bg-black/20 text-zinc-50"
          placeholder="+91…"
          required
        />
      </div>
      <div className="md:col-span-3">
        <Label data-testid={`signup-vehicle-number-label-${idx}`}>Vehicle number</Label>
        <Input
          value={vehicle.vehicle_number}
          onChange={(e) => onChange(idx, { ...vehicle, vehicle_number: e.target.value })}
          data-testid={`signup-vehicle-number-input-${idx}`}
          className="mt-2 border-white/15 bg-black/20 text-zinc-50"
          placeholder="MH12AB1234"
          required
        />
      </div>
      <div className="md:col-span-2">
        <Label data-testid={`signup-vehicle-capacity-label-${idx}`}>Capacity</Label>
        <Input
          type="number"
          value={vehicle.capacity}
          onChange={(e) => onChange(idx, { ...vehicle, capacity: e.target.value })}
          data-testid={`signup-vehicle-capacity-input-${idx}`}
          className="mt-2 border-white/15 bg-black/20 text-zinc-50"
          placeholder="e.g., 800"
          required
          min={1}
          step={1}
        />
      </div>
      <div className="md:col-span-1 flex items-end justify-end">
        <Button
          type="button"
          variant="outline"
          disabled={!canRemove}
          onClick={() => onRemove(idx)}
          data-testid={`signup-vehicle-remove-button-${idx}`}
          className="border-white/15 bg-white/5 text-zinc-50 hover:bg-white/10"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

export default function SignupVehicles() {
  const nav = useNavigate();
  const { api } = useAuth();

  const [vehicles, setVehicles] = useState([
    { driver_name: "", driver_phone: "", vehicle_number: "", capacity: "" },
  ]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const addRow = () =>
    setVehicles((v) => [...v, { driver_name: "", driver_phone: "", vehicle_number: "", capacity: "" }]);

  const updateRow = (idx, next) => setVehicles((v) => v.map((r, i) => (i === idx ? next : r)));
  const removeRow = (idx) => setVehicles((v) => v.filter((_, i) => i !== idx));

  const onFinish = async () => {
    setBusy(true);
    setError("");
    try {
      const payload = vehicles.map((v) => ({ ...v, capacity: Number(v.capacity) }));
      await api.post("/vehicles/bulk", payload);
      nav("/dashboard");
    } catch (err) {
      setError(err?.response?.data?.detail || "Could not save vehicles");
    } finally {
      setBusy(false);
    }
  };

  return (
    <AppShell variant="public">
      <div data-testid="signup-vehicles-page" className="space-y-6">
        <Card className="border-white/10 bg-white/5 text-zinc-50">
          <CardHeader>
            <CardTitle data-testid="signup-vehicles-title" className="text-xl">Add Vehicles & Drivers</CardTitle>
            <CardDescription className="text-zinc-200/70">Step 3 of 3 — Capacity helps compute round trips.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div data-testid="signup-vehicles-hint" className="text-sm text-zinc-200/75">
                Add all vehicles used for collection.
              </div>
              <Button
                type="button"
                onClick={addRow}
                data-testid="signup-vehicle-add-row-button"
                className="rounded-full bg-white/10 text-zinc-50 ring-1 ring-white/10 hover:bg-white/15"
              >
                <Plus className="h-4 w-4" /> Add another vehicle
              </Button>
            </div>

            <div className="space-y-3">
              {vehicles.map((v, idx) => (
                <VehicleRow
                  key={idx}
                  idx={idx}
                  vehicle={v}
                  onChange={updateRow}
                  onRemove={removeRow}
                  canRemove={vehicles.length > 1}
                />
              ))}
            </div>

            {error ? (
              <div
                data-testid="signup-vehicles-error"
                className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-100"
              >
                {error}
              </div>
            ) : null}

            <div className="flex flex-col gap-3 sm:flex-row">
              <Button
                type="button"
                variant="outline"
                onClick={() => nav("/signup/wards")}
                data-testid="signup-vehicles-back-button"
                className="w-full border-white/15 bg-white/5 text-zinc-50 hover:bg-white/10 sm:w-auto"
              >
                <ArrowLeft className="h-4 w-4" /> Back
              </Button>
              <Button
                type="button"
                onClick={onFinish}
                disabled={busy}
                data-testid="signup-vehicles-finish-button"
                className="w-full rounded-full bg-emerald-400/20 text-emerald-100 ring-1 ring-emerald-200/25 hover:bg-emerald-400/25 sm:flex-1"
              >
                {busy ? "Finishing…" : "Finish & Go to Dashboard"} <CheckCircle2 className="h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}
