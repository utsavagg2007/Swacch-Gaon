import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AppShell from "@/components/AppShell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/context/auth";

export default function Login() {
  const nav = useNavigate();
  const { api, setToken } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const onSubmit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const res = await api.post("/auth/login", { email, password });
      setToken(res.data.token);
      nav("/dashboard");
    } catch (err) {
      setError(err?.response?.data?.detail || "Login failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <AppShell variant="public">
      <div data-testid="login-page" className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="lg:col-span-5">
          <Card className="border-white/10 bg-white/5 text-zinc-50">
            <CardHeader>
              <CardTitle data-testid="login-title" className="text-xl">Sign in</CardTitle>
              <CardDescription className="text-zinc-200/70">
                Use your Panchayat email and password.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={onSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email" data-testid="login-email-label">Panchayat email</Label>
                  <Input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    data-testid="login-email-input"
                    className="border-white/15 bg-black/20 text-zinc-50 placeholder:text-zinc-400"
                    placeholder="panchayat@example.com"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="password" data-testid="login-password-label">Password</Label>
                  <Input
                    id="password"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    data-testid="login-password-input"
                    className="border-white/15 bg-black/20 text-zinc-50 placeholder:text-zinc-400"
                    placeholder="••••••••"
                    required
                  />
                </div>

                {error ? (
                  <div
                    data-testid="login-error"
                    className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-100"
                  >
                    {error}
                  </div>
                ) : null}

                <Button
                  type="submit"
                  disabled={busy}
                  data-testid="login-submit-button"
                  className="w-full rounded-full bg-emerald-400/20 text-emerald-100 ring-1 ring-emerald-200/25 hover:bg-emerald-400/25"
                >
                  {busy ? "Signing in…" : "Sign in"}
                </Button>

                <div className="text-sm text-zinc-200/75">
                  Don’t have an account?{" "}
                  <Link
                    to="/signup"
                    data-testid="login-signup-link"
                    className="text-emerald-200 hover:text-emerald-100 underline underline-offset-4"
                  >
                    Sign up
                  </Link>
                </div>
              </form>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-7">
          <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
            <div data-testid="login-side-title" className="text-sm font-semibold text-zinc-100">
              What you can do after signing in
            </div>
            <ul className="mt-3 space-y-2 text-sm text-zinc-200/80">
              <li data-testid="login-side-item-1">• Add wards with auto geocoding (lat/long)</li>
              <li data-testid="login-side-item-2">• Add vehicles/drivers with capacity</li>
              <li data-testid="login-side-item-3">• Create daily logs and view history</li>
              <li data-testid="login-side-item-4">• Run prediction + optimized routes for next day</li>
            </ul>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
