import React from "react";
import "@/App.css";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider } from "@/context/auth";
import Home from "@/pages/Home";
import Login from "@/pages/Login";
import Signup from "@/pages/Signup";
import SignupWards from "@/pages/SignupWards";
import SignupVehicles from "@/pages/SignupVehicles";
import Dashboard from "@/pages/Dashboard";
import EditWards from "@/pages/EditWards";
import EditVehicles from "@/pages/EditVehicles";
import { Toaster } from "@/components/ui/toaster";

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Navigate to="/home" replace />} />
          <Route path="/home" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/signup/wards" element={<SignupWards />} />
          <Route path="/signup/vehicle" element={<SignupVehicles />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/dashboard/wards" element={<EditWards />} />
          <Route path="/dashboard/vehicle" element={<EditVehicles />} />
          <Route path="*" element={<Navigate to="/home" replace />} />
        </Routes>
        <Toaster />
      </BrowserRouter>
    </AuthProvider>
  );
}
