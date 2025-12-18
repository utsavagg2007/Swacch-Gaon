import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import AppShell from "@/components/AppShell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Plus, Trash2, ArrowRight, ArrowLeft } from "lucide-react";
import { useAuth } from "@/context/auth";

function WardRow({ idx, ward, onChange, onRemove, canRemove }) {
  return (
    <div
      data-testid={`signup-ward-row-${idx}`}
      className="grid grid-cols-1 gap-3 rounded-xl border border-white/10 bg-black/15 p-4 md:grid-cols-12"
    >
      <div className="md:col-span-4">
        <Label data-testid={`signup-ward-name-label-${idx}`}>Ward name</Label>
        <Input
          value={ward.name}
          onChange={(e) => onChange(idx, { ...ward, name: e.target.value })}
          data-testid={`signup-ward-name-input-${idx}`}
          className="mt-2 border-white/15 bg-black/20 text-zinc-50"
          placeholder="e.g., Ward 1"
          required
        />
      </div>
      <div className="md:col-span-7">
        <Label data-testid={`signup-ward-address-label-${idx}`}>Ward full address</Label>
        <Input
          value={ward.address}
          onChange={(e) => onChange(idx, { ...ward, address: e.target.value })}
          data-testid={`signup-ward-address-input-${idx}`}
          className="mt-2 border-white/15 bg-black/20 text-zinc-50"
          placeholder="Street, village, landmark, district"
          required
        />
      </div>
      <div className="md:col-span-1 flex items-end justify-end">
        <Button
          type="button"
          variant="outline"
          disabled={!canRemove}
          onClick={() => onRemove(idx)}
          data-testid={`signup-ward-remove-button-${idx}`}
          className="border-white/15 bg-white/5 text-zinc-50 hover:bg-white/10"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

export default function SignupWards() {
  const nav = useNavigate();
  const { api } = useAuth();

  const [wards, setWards] = useState([{ name: "", address: "" }]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const addRow = () => setWards((w) => [...w, { name: "", address: "" }]);
  const updateRow = (idx, next) => setWards((w) => w.map((r, i) => (i === idx ? next : r)));
  const removeRow = (idx) => setWards((w) => w.filter((_, i) => i !== idx));

  const onNext = async () => {
    setBusy(true);
    setError("");
    try {
      await api.post("/wards/bulk", wards);
      nav("/signup/vehicle");
    } catch (err) {
      setError(err?.response?.data?.detail || "Could not save wards");
    } finally {
      setBusy(false);
    }
  };

  return (
    <AppShell variant="public">
      <div data-testid="signup-wards-page" className="space-y-6">
        <Card className="border-white/10 bg-white/5 text-zinc-50">
          <CardHeader>
            <CardTitle data-testid="signup-wards-title" className="text-xl">Add Wards</CardTitle>
            <CardDescription className="text-zinc-200/70">Step 2 of 3 — Ward name + address (auto geocoded)</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div data-testid="signup-wards-hint" className="text-sm text-zinc-200/75">
                Add as many wards as needed.
              </div>
              <Button
                type="button"
                onClick={addRow}
                data-testid="signup-ward-add-row-button"
                className="rounded-full bg-white/10 text-zinc-50 ring-1 ring-white/10 hover:bg-white/15"
              >
                <Plus className="h-4 w-4" /> Add another ward
              </Button>
            </div>

            <div className="space-y-3">
              {wards.map((w, idx) => (
                <WardRow
                  key={idx}
                  idx={idx}
                  ward={w}
                  onChange={updateRow}
                  onRemove={removeRow}
                  canRemove={wards.length > 1}
                />
              ))}
            </div>

            {error ? (
              <div
                data-testid="signup-wards-error"
                className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-100"
              >
                {error}
              </div>
            ) : null}

            <div className="flex flex-col gap-3 sm:flex-row">
              <Button
                type="button"
                variant="outline"
                onClick={() => nav("/signup")}
                data-testid="signup-wards-back-button"
                className="w-full border-white/15 bg-white/5 text-zinc-50 hover:bg-white/10 sm:w-auto"
              >
                <ArrowLeft className="h-4 w-4" /> Back
              </Button>
              <Button
                type="button"
                onClick={onNext}
                disabled={busy}
                data-testid="signup-wards-next-button"
                className="w-full rounded-full bg-emerald-400/20 text-emerald-100 ring-1 ring-emerald-200/25 hover:bg-emerald-400/25 sm:flex-1"
              >
                {busy ? "Saving…" : "Next: Add Vehicles"} <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}
