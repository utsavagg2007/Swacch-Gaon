import React from "react";
import { Link } from "react-router-dom";
import AppShell from "@/components/AppShell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowRight, Route, PhoneCall, BrainCircuit, ShieldCheck } from "lucide-react";

const heroImg = "https://images.pexels.com/photos/19935567/pexels-photo-19935567.jpeg";

export default function Home() {
  return (
    <AppShell variant="public">
      <div data-testid="home-page" className="space-y-10">
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-12 lg:items-center">
          <div className="lg:col-span-7">
            <Badge data-testid="home-badge" className="bg-white/10 text-zinc-100 ring-1 ring-white/10">
              Panchayat Operations • Next-day Planning
            </Badge>
            <h1
              data-testid="home-hero-title"
              className="mt-4 text-4xl sm:text-5xl lg:text-6xl font-semibold leading-[1.05] tracking-tight"
            >
              Predict tomorrow’s waste.
              <span className="block text-emerald-200">Optimize routes today.</span>
            </h1>
            <p data-testid="home-hero-subtitle" className="mt-4 text-base md:text-lg text-zinc-200/85 max-w-xl">
              An end-to-end workflow for rural waste collection: ward setup, vehicle & driver management, daily
              logs, and automated route plans — with Retell-ready call payloads.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <Link to="/login" data-testid="home-cta-login-link">
                <Button
                  data-testid="home-cta-login-button"
                  className="h-10 rounded-full bg-emerald-400/20 px-5 text-emerald-100 ring-1 ring-emerald-200/25 hover:bg-emerald-400/25"
                >
                  Sign in <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
              <Link to="/signup" data-testid="home-cta-signup-link">
                <Button
                  data-testid="home-cta-signup-button"
                  variant="outline"
                  className="h-10 rounded-full border-white/15 bg-white/5 px-5 text-zinc-50 hover:bg-white/10"
                >
                  Create Panchayat Account
                </Button>
              </Link>
            </div>
          </div>

          <div className="lg:col-span-5">
            <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/5">
              <img
                data-testid="home-hero-image"
                src={heroImg}
                alt="Rural road"
                className="h-[320px] w-full object-cover opacity-85"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/30 to-transparent" />
              <div className="absolute bottom-4 left-4 right-4">
                <div className="text-sm font-medium">Designed for early-morning operations</div>
                <div className="text-xs text-zinc-200/75">Routes + driver briefing payload at 06:00 IST</div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card data-testid="home-feature-1" className="border-white/10 bg-white/5 text-zinc-50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <BrainCircuit className="h-4 w-4 text-emerald-200" /> Prediction
              </CardTitle>
              <CardDescription className="text-zinc-200/70">Random Forest with hyperparameter tuning.</CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-zinc-200/85">
              Uses your daily logs to forecast ward-wise waste for the next day.
            </CardContent>
          </Card>
          <Card data-testid="home-feature-2" className="border-white/10 bg-white/5 text-zinc-50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Route className="h-4 w-4 text-sky-200" /> Routing
              </CardTitle>
              <CardDescription className="text-zinc-200/70">Haversine distance + efficient ordering.</CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-zinc-200/85">
              Assigns wards to vehicles and produces route order with round trips.
            </CardContent>
          </Card>
          <Card data-testid="home-feature-3" className="border-white/10 bg-white/5 text-zinc-50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <PhoneCall className="h-4 w-4 text-violet-200" /> Retell-ready
              </CardTitle>
              <CardDescription className="text-zinc-200/70">Endpoints for data IN / data OUT.</CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-zinc-200/85">
              Morning payload (routes) + evening webhook (logs), with proportional allocation.
            </CardContent>
          </Card>
          <Card data-testid="home-feature-4" className="border-white/10 bg-white/5 text-zinc-50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <ShieldCheck className="h-4 w-4 text-amber-200" /> Secure Access
              </CardTitle>
              <CardDescription className="text-zinc-200/70">JWT sign-in for dashboard routes.</CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-zinc-200/85">
              Private dashboard and CRUD tools for wards, vehicles, and daily logs.
            </CardContent>
          </Card>
        </div>
      </div>
    </AppShell>
  );
}
