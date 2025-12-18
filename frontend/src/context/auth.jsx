import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { createApiClient } from "@/api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem("rwo_token") || "");
  const [me, setMe] = useState(null);
  const [loading, setLoading] = useState(true);

  const api = useMemo(() => createApiClient(() => token), [token]);

  const logout = () => {
    localStorage.removeItem("rwo_token");
    setToken("");
    setMe(null);
  };

  const refreshMe = async () => {
    if (!token) {
      setMe(null);
      setLoading(false);
      return;
    }
    try {
      const res = await api.get("/auth/me");
      setMe(res.data);
    } catch (e) {
      logout();
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshMe();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const value = {
    token,
    me,
    loading,
    api,
    setToken: (t) => {
      localStorage.setItem("rwo_token", t);
      setToken(t);
    },
    logout,
    refreshMe,
    isAuthed: !!token,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
