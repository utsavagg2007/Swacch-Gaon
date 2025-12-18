import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "@/context/auth";

export default function ProtectedRoute({ children }) {
  const { isAuthed, loading } = useAuth();
  if (loading) {
    return (
      <div data-testid="auth-loading" className="text-zinc-200">
        Loading…
      </div>
    );
  }
  if (!isAuthed) return <Navigate to="/login" replace />;
  return children;
}
