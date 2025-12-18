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

function VehicleEditor({ mode, initial, onSave, busy }) {
  const [driver_name, setDriverName] = useState(initial?.driver_name || "");
  const [driver_phone, setDriverPhone] = useState(initial?.driver_phone || "");
  const [vehicle_number, setVehicleNumber] = useState(initial?.vehicle_number || "");
  const [capacity, setCapacity] = useState(initial?.capacity || "");

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <div className="space-y-2">
          <Label data-testid="vehicle-editor-driver-label">Driver name</Label>
          <Input
            value={driver_name}
            onChange={(e) => setDriverName(e.target.value)}
            data-testid="vehicle-editor-driver-input"
            className="border-white/15 bg-black/20 text-zinc-50"
          />
        </div>
        <div className="space-y-2">
          <Label data-testid="vehicle-editor-phone-label">Driver phone</Label>
          <Input
            value={driver_phone}
            onChange={(e) => setDriverPhone(e.target.value)}
            data-testid="vehicle-editor-phone-input"
            className="border-white/15 bg-black/20 text-zinc-50"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <div className="space-y-2">
          <Label data-testid="vehicle-editor-number-label">Vehicle number</Label>
          <Input
            value={vehicle_number}
            onChange={(e) => setVehicleNumber(e.target.value)}
            data-testid="vehicle-editor-number-input"
            className="border-white/15 bg-black/20 text-zinc-50"
          />
        </div>
        <div className="space-y-2">
          <Label data-testid="vehicle-editor-capacity-label">Capacity</Label>
          <Input
            type="number"
            min={1}
            step={1}
            value={capacity}
            onChange={(e) => setCapacity(e.target.value)}
            data-testid="vehicle-editor-capacity-input"
            className="border-white/15 bg-black/20 text-zinc-50"
          />
        </div>
      </div>

      <Button
        onClick={() => onSave({ driver_name, driver_phone, vehicle_number, capacity: Number(capacity) })}
        disabled={busy || !driver_name || !driver_phone || !vehicle_number || !capacity}
        data-testid="vehicle-editor-save-button"
        className="w-full rounded-full bg-emerald-400/20 text-emerald-100 ring-1 ring-emerald-200/25 hover:bg-emerald-400/25"
      >
        {busy ? "Saving…" : mode === "create" ? "Create vehicle" : "Save changes"}
      </Button>
    </div>
  );
}

export default function EditVehicles() {
  const { api } = useAuth();
  const [vehicles, setVehicles] = useState([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const load = async () => {
    setError("");
    const res = await api.get("/vehicles");
    setVehicles(res.data);
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const createVehicle = async (payload) => {
    setBusy(true);
    setError("");
    try {
      await api.post("/vehicles", payload);
      await load();
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to create vehicle");
    } finally {
      setBusy(false);
    }
  };

  const updateVehicle = async (id, payload) => {
    setBusy(true);
    setError("");
    try {
      await api.put(`/vehicles/${id}`, payload);
      await load();
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to update vehicle");
    } finally {
      setBusy(false);
    }
  };

  const deleteVehicle = async (id) => {
    setBusy(true);
    setError("");
    try {
      await api.delete(`/vehicles/${id}`);
      await load();
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to delete vehicle");
    } finally {
      setBusy(false);
    }
  };

  return (
    <ProtectedRoute>
      <AppShell variant="private">
        <div data-testid="edit-vehicles-page" className="space-y-6">
          <Card className="border-white/10 bg-white/5 text-zinc-50">
            <CardHeader className="flex flex-row items-start justify-between gap-4">
              <div>
                <CardTitle data-testid="edit-vehicles-title" className="text-xl">Edit Vehicles</CardTitle>
                <CardDescription className="text-zinc-200/70">Manage drivers, vehicle numbers and capacities</CardDescription>
              </div>
              <Dialog>
                <DialogTrigger asChild>
                  <Button
                    data-testid="edit-vehicles-add-button"
                    className="rounded-full bg-white/10 text-zinc-50 ring-1 ring-white/10 hover:bg-white/15"
                  >
                    <Plus className="h-4 w-4" /> Add vehicle
                  </Button>
                </DialogTrigger>
                <DialogContent className="border-white/10 bg-[#0b1020] text-zinc-50">
                  <DialogHeader>
                    <DialogTitle data-testid="edit-vehicles-create-dialog-title">Create vehicle</DialogTitle>
                    <DialogDescription className="text-zinc-200/70">Capacity is used for round trip calculation.</DialogDescription>
                  </DialogHeader>
                  <VehicleEditor mode="create" onSave={createVehicle} busy={busy} />
                </DialogContent>
              </Dialog>
            </CardHeader>
            <CardContent className="space-y-3">
              {error ? (
                <div
                  data-testid="edit-vehicles-error"
                  className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-100"
                >
                  {error}
                </div>
              ) : null}

              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead data-testid="edit-vehicles-th-number">Vehicle</TableHead>
                    <TableHead data-testid="edit-vehicles-th-driver">Driver</TableHead>
                    <TableHead data-testid="edit-vehicles-th-phone">Phone</TableHead>
                    <TableHead data-testid="edit-vehicles-th-capacity">Capacity</TableHead>
                    <TableHead data-testid="edit-vehicles-th-actions">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {vehicles.map((v) => (
                    <TableRow key={v.id} data-testid={`edit-vehicles-row-${v.id}`}>
                      <TableCell data-testid={`edit-vehicles-number-${v.id}`}>{v.vehicle_number}</TableCell>
                      <TableCell data-testid={`edit-vehicles-driver-${v.id}`}>{v.driver_name}</TableCell>
                      <TableCell data-testid={`edit-vehicles-phone-${v.id}`}>{v.driver_phone}</TableCell>
                      <TableCell data-testid={`edit-vehicles-capacity-${v.id}`}>{v.capacity}</TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Dialog>
                            <DialogTrigger asChild>
                              <Button
                                variant="outline"
                                data-testid={`edit-vehicles-edit-button-${v.id}`}
                                className="border-white/15 bg-white/5 text-zinc-50 hover:bg-white/10"
                                size="sm"
                              >
                                <Pencil className="h-4 w-4" /> Edit
                              </Button>
                            </DialogTrigger>
                            <DialogContent className="border-white/10 bg-[#0b1020] text-zinc-50">
                              <DialogHeader>
                                <DialogTitle data-testid={`edit-vehicles-edit-dialog-title-${v.id}`}>Edit vehicle</DialogTitle>
                                <DialogDescription className="text-zinc-200/70">Update fields below.</DialogDescription>
                              </DialogHeader>
                              <VehicleEditor mode="edit" initial={v} onSave={(p) => updateVehicle(v.id, p)} busy={busy} />
                            </DialogContent>
                          </Dialog>

                          <Button
                            variant="outline"
                            onClick={() => deleteVehicle(v.id)}
                            data-testid={`edit-vehicles-delete-button-${v.id}`}
                            className="border-rose-500/25 bg-rose-500/10 text-rose-100 hover:bg-rose-500/15"
                            size="sm"
                          >
                            <Trash2 className="h-4 w-4" /> Delete
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                  {vehicles.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={5} data-testid="edit-vehicles-empty" className="text-zinc-200/70">
                        No vehicles yet.
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
