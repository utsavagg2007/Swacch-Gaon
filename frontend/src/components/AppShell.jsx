import React from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { useAuth } from "@/context/auth";
import { Leaf, LogIn, LogOut, Map, Truck, LayoutDashboard } from "lucide-react";

export default function AppShell({ variant = "public", children }) {
  const { pathname } = useLocation();
  const nav = useNavigate();
  const { isAuthed, logout, me } = useAuth();

  const onLogout = () => {
    logout();
    nav("/home");
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(1200px_circle_at_10%_10%,rgba(16,185,129,0.18),transparent_55%),radial-gradient(900px_circle_at_80%_30%,rgba(14,165,233,0.16),transparent_45%),linear-gradient(to_bottom,#070A0D,#0B1020)] text-zinc-50" style={{ minHeight: "100vh" }}>
      <div className="sticky top-0 z-40 border-b border-white/10 bg-black/30 backdrop-blur supports-[backdrop-filter]:bg-black/20">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <Link
            to="/home"
            data-testid="navbar-logo-link"
            className="flex items-center gap-2"
          >
            <div className="grid h-10 w-10 place-items-center rounded-xl bg-white/10 ring-1 ring-white/15">
              <Leaf className="h-5 w-5 text-emerald-200" />
            </div>
            <div>
              <div className="text-sm font-semibold tracking-wide">Rural Waste Optimizer</div>
              <div className="text-xs text-zinc-300/80">Routes • Calls • Logs</div>
            </div>
          </Link>

          <div className="flex items-center gap-2">
            {variant === "private" && me ? (
              <Badge data-testid="navbar-panchayat-badge" className="bg-white/10 text-zinc-100 ring-1 ring-white/10">
                {me.name}
              </Badge>
            ) : null}

            {variant === "private" ? (
              <>
                <Link to="/dashboard" data-testid="navbar-dashboard-link">
                  <Button
                    variant={pathname.startsWith("/dashboard") ? "secondary" : "outline"}
                    className={cn(
                      "border-white/15 bg-white/5 text-zinc-50 hover:bg-white/10",
                      pathname === "/dashboard" ? "bg-white/10" : ""
                    )}
                    size="sm"
                  >
                    <LayoutDashboard className="h-4 w-4" /> Dashboard
                  </Button>
                </Link>
                <Link to="/dashboard/wards" data-testid="navbar-wards-link">
                  <Button
                    variant="outline"
                    className="border-white/15 bg-white/5 text-zinc-50 hover:bg-white/10"
                    size="sm"
                  >
                    <Map className="h-4 w-4" /> Wards
                  </Button>
                </Link>
                <Link to="/dashboard/vehicle" data-testid="navbar-vehicles-link">
                  <Button
                    variant="outline"
                    className="border-white/15 bg-white/5 text-zinc-50 hover:bg-white/10"
                    size="sm"
                  >
                    <Truck className="h-4 w-4" /> Vehicles
                  </Button>
                </Link>

                <Button
                  onClick={onLogout}
                  data-testid="navbar-logout-button"
                  className="bg-emerald-400/15 text-emerald-100 ring-1 ring-emerald-200/20 hover:bg-emerald-400/20"
                  size="sm"
                >
                  <LogOut className="h-4 w-4" /> Logout
                </Button>
              </>
            ) : (
              <>
                {isAuthed ? (
                  <Link to="/dashboard" data-testid="navbar-go-dashboard-link">
                    <Button
                      className="bg-emerald-400/15 text-emerald-100 ring-1 ring-emerald-200/20 hover:bg-emerald-400/20"
                      size="sm"
                    >
                      <LayoutDashboard className="h-4 w-4" /> Dashboard
                    </Button>
                  </Link>
                ) : (
                  <Link to="/login" data-testid="navbar-signin-link">
                    <Button
                      className="bg-emerald-400/15 text-emerald-100 ring-1 ring-emerald-200/20 hover:bg-emerald-400/20"
                      size="sm"
                    >
                      <LogIn className="h-4 w-4" /> Sign in
                    </Button>
                  </Link>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      <main className="mx-auto max-w-6xl px-4 py-10">{children}</main>

      <footer className="border-t border-white/10 bg-black/20">
        <div className="mx-auto max-w-6xl px-4 py-6 text-xs text-zinc-300/70">
          <div data-testid="footer-text">MVP build — Retell endpoints ready, calls not yet automated.</div>
        </div>
      </footer>
    </div>
  );
}
