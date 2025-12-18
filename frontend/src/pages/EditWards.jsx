import React, { useEffect, useState } from "react";
import AppShell from "@/components/AppShell";
import ProtectedRoute from "@/components/ProtectedRoute";
import { useAuth } from "@/context/auth";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Plus, Pencil, Trash2 } from "lucide-react";

function WardEditor({ mode, initial, onSave, busy }) {
  const [name, setName] = useState(initial?.name || "");
  const [address, setAddress] = useState(initial?.address || "");

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label data-testid="ward-editor-name-label">Ward name</Label>
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          data-testid="ward-editor-name-input"
          className="border-white/15 bg-black/20 text-zinc-50"
        />
      </div>
      <div className="space-y-2">
        <Label data-testid="ward-editor-address-label">Ward address</Label>
        <Input
          value={address}
          onChange={(e) => setAddress(e.target.value)}
          data-testid="ward-editor-address-input"
          className="border-white/15 bg-black/20 text-zinc-50"
        />
      </div>
      <Button
        onClick={() => onSave({ name, address })}
        disabled={busy || !name || !address}
        data-testid="ward-editor-save-button"
        className="w-full rounded-full bg-emerald-400/20 text-emerald-100 ring-1 ring-emerald-200/25 hover:bg-emerald-400/25"
      >
        {busy ? "Saving…" : mode === "create" ? "Create ward" : "Save changes"}
      </Button>
    </div>
  );
}

export default function EditWards() {
  const { api } = useAuth();
  const [wards, setWards] = useState([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const load = async () => {
    setError("");
    const res = await api.get("/wards");
    setWards(res.data);
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const createWard = async (payload) => {
    setBusy(true);
    setError("");
    try {
      await api.post("/wards", payload);
      await load();
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to create ward");
    } finally {
      setBusy(false);
    }
  };

  const updateWard = async (id, payload) => {
    setBusy(true);
    setError("");
    try {
      await api.put(`/wards/${id}`, payload);
      await load();
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to update ward");
    } finally {
      setBusy(false);
    }
  };

  const deleteWard = async (id) => {
    setBusy(true);
    setError("");
    try {
      await api.delete(`/wards/${id}`);
      await load();
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to delete ward");
    } finally {
      setBusy(false);
    }
  };

  return (
    <ProtectedRoute>
      <AppShell variant="private">
        <div data-testid="edit-wards-page" className="space-y-6">
          <Card className="border-white/10 bg-white/5 text-zinc-50">
            <CardHeader className="flex flex-row items-start justify-between gap-4">
              <div>
                <CardTitle data-testid="edit-wards-title" className="text-xl">Edit Wards</CardTitle>
                <CardDescription className="text-zinc-200/70">Add, update or remove ward records</CardDescription>
              </div>
              <Dialog>
                <DialogTrigger asChild>
                  <Button
                    data-testid="edit-wards-add-button"
                    className="rounded-full bg-white/10 text-zinc-50 ring-1 ring-white/10 hover:bg-white/15"
                  >
                    <Plus className="h-4 w-4" /> Add ward
                  </Button>
                </DialogTrigger>
                <DialogContent className="border-white/10 bg-[#0b1020] text-zinc-50">
                  <DialogHeader>
                    <DialogTitle data-testid="edit-wards-create-dialog-title">Create ward</DialogTitle>
                    <DialogDescription className="text-zinc-200/70">Address will be converted to lat/long automatically.</DialogDescription>
                  </DialogHeader>
                  <WardEditor mode="create" onSave={createWard} busy={busy} />
                </DialogContent>
              </Dialog>
            </CardHeader>
            <CardContent className="space-y-3">
              {error ? (
                <div
                  data-testid="edit-wards-error"
                  className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-100"
                >
                  {error}
                </div>
              ) : null}

              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead data-testid="edit-wards-th-name">Name</TableHead>
                    <TableHead data-testid="edit-wards-th-address">Address</TableHead>
                    <TableHead data-testid="edit-wards-th-actions">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {wards.map((w) => (
                    <TableRow key={w.id} data-testid={`edit-wards-row-${w.id}`}>
                      <TableCell data-testid={`edit-wards-name-${w.id}`}>{w.name}</TableCell>
                      <TableCell data-testid={`edit-wards-address-${w.id}`}>{w.address}</TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Dialog>
                            <DialogTrigger asChild>
                              <Button
                                variant="outline"
                                data-testid={`edit-wards-edit-button-${w.id}`}
                                className="border-white/15 bg-white/5 text-zinc-50 hover:bg-white/10"
                                size="sm"
                              >
                                <Pencil className="h-4 w-4" /> Edit
                              </Button>
                            </DialogTrigger>
                            <DialogContent className="border-white/10 bg-[#0b1020] text-zinc-50">
                              <DialogHeader>
                                <DialogTitle data-testid={`edit-wards-edit-dialog-title-${w.id}`}>Edit ward</DialogTitle>
                                <DialogDescription className="text-zinc-200/70">Update name/address.</DialogDescription>
                              </DialogHeader>
                              <WardEditor mode="edit" initial={w} onSave={(p) => updateWard(w.id, p)} busy={busy} />
                            </DialogContent>
                          </Dialog>

                          <Button
                            variant="outline"
                            onClick={() => deleteWard(w.id)}
                            data-testid={`edit-wards-delete-button-${w.id}`}
                            className="border-rose-500/25 bg-rose-500/10 text-rose-100 hover:bg-rose-500/15"
                            size="sm"
                          >
                            <Trash2 className="h-4 w-4" /> Delete
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                  {wards.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={3} data-testid="edit-wards-empty" className="text-zinc-200/70">
                        No wards yet.
                      </TableCell>
                    </TableRow>
                  ) : null}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>
      </AppShell>
    </ProtectedRoute>
  );
}
