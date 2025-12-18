import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import AppShell from "@/components/AppShell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useAuth } from "@/context/auth";

export default function Signup() {
  const nav = useNavigate();
  const { api, setToken } = useAuth();

  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [address, setAddress] = useState("");
  const [password, setPassword] = useState("");

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const onSubmit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const res = await api.post("/auth/signup", { email, name, address, password });
      setToken(res.data.token);
      nav("/signup/wards");
    } catch (err) {
      setError(err?.response?.data?.detail || "Signup failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <AppShell variant="public">
      <div data-testid="signup-page" className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="lg:col-span-6">
          <Card className="border-white/10 bg-white/5 text-zinc-50">
            <CardHeader>
              <CardTitle data-testid="signup-title" className="text-xl">Create Panchayat Account</CardTitle>
              <CardDescription className="text-zinc-200/70">
                Step 1 of 3 — Panchayat details
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={onSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label data-testid="signup-email-label" htmlFor="email">Panchayat email</Label>
                  <Input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    data-testid="signup-email-input"
                    className="border-white/15 bg-black/20 text-zinc-50 placeholder:text-zinc-400"
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label data-testid="signup-name-label" htmlFor="name">Panchayat name</Label>
                  <Input
                    id="name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    data-testid="signup-name-input"
                    className="border-white/15 bg-black/20 text-zinc-50 placeholder:text-zinc-400"
                    placeholder="e.g., Gram Panchayat X"
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label data-testid="signup-address-label" htmlFor="address">Full address</Label>
                  <Textarea
                    id="address"
                    value={address}
                    onChange={(e) => setAddress(e.target.value)}
                    data-testid="signup-address-input"
                    className="min-h-[96px] border-white/15 bg-black/20 text-zinc-50 placeholder:text-zinc-400"
                    placeholder="Village, block, district, state"
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label data-testid="signup-password-label" htmlFor="password">Create password</Label>
                  <Input
                    id="password"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    data-testid="signup-password-input"
                    className="border-white/15 bg-black/20 text-zinc-50 placeholder:text-zinc-400"
                    required
                  />
                </div>

                {error ? (
                  <div
                    data-testid="signup-error"
                    className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-100"
                  >
                    {error}
                  </div>
                ) : null}

                <Button
                  type="submit"
                  disabled={busy}
                  data-testid="signup-next-button"
                  className="w-full rounded-full bg-emerald-400/20 text-emerald-100 ring-1 ring-emerald-200/25 hover:bg-emerald-400/25"
                >
                  {busy ? "Creating…" : "Next: Add Wards"}
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-6">
          <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
            <div data-testid="signup-side-title" className="text-sm font-semibold text-zinc-100">
              Why we ask for address
            </div>
            <p data-testid="signup-side-text" className="mt-2 text-sm text-zinc-200/80">
              We convert the address to latitude/longitude so routes can be optimized with real distances.
            </p>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
