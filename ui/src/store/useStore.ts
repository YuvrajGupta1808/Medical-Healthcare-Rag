import { create } from 'zustand';

export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: string;
  doc_ids: string[];
}

interface AppState {
  doctorName: string | null;
  setDoctorName: (name: string) => void;
  
  patients: Patient[];
  addPatient: (patient: Patient) => void;
  removePatient: (id: string) => void;
  createPatient: (patient: Omit<Patient, 'doc_ids'>) => Promise<void>;
  deletePatient: (id: string) => Promise<void>;
  addDocToPatient: (patientId: string, docId: string) => void;
  removeDocFromPatient: (patientId: string, docId: string) => void;
  deleteDocFromPatient: (docId: string) => Promise<void>;
  
  selectedPatientId: string | null;
  setSelectedPatientId: (id: string | null) => void;
  fetchPatients: () => Promise<void>;
  checkHealth: () => Promise<void>;
  isLoading: boolean;
  backendStatus: 'online' | 'offline' | 'loading';
}

const API_BASE = "http://localhost:8000";

export const useStore = create<AppState>((set) => ({
  doctorName: "Dr. Sarah Chen",
  setDoctorName: (name) => set({ doctorName: name }),
  
  patients: [],
  isLoading: false,
  backendStatus: 'loading',
  addPatient: (patient) => set((state) => ({ patients: [...state.patients, patient] })),
  removePatient: (id: string) => set((state) => ({
    patients: state.patients.filter(p => p.id !== id),
    selectedPatientId: state.selectedPatientId === id ? null : state.selectedPatientId
  })),
  
  fetchPatients: async () => {
    set({ isLoading: true });
    try {
      const res = await fetch(`${API_BASE}/patients`);
      if (res.ok) {
        const data = await res.json();
        set({ patients: data, isLoading: false, backendStatus: 'online' });
      }
    } catch (err) {
      console.error("Failed to fetch patients:", err);
      set({ isLoading: false, backendStatus: 'offline' });
    }
  },

  checkHealth: async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      set({ backendStatus: res.ok ? 'online' : 'offline' });
    } catch (err) {
      set({ backendStatus: 'offline' });
    }
  },

  createPatient: async (patientData) => {
    try {
      const patient = { ...patientData, doc_ids: [] };
      const res = await fetch(`${API_BASE}/patients`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patient),
      });
      if (res.ok) {
        const data = await res.json();
        set((state) => ({ patients: [...state.patients, data] }));
      }
    } catch (err) {
      console.error("Failed to create patient:", err);
    }
  },

  deletePatient: async (id) => {
    try {
      const res = await fetch(`${API_BASE}/patients/${id}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        set((state) => ({
          patients: state.patients.filter(p => p.id !== id),
          selectedPatientId: state.selectedPatientId === id ? null : state.selectedPatientId
        }));
      }
    } catch (err) {
      console.error("Failed to delete patient:", err);
    }
  },
  
  addDocToPatient: (patientId, docId) => set((state) => ({
    patients: state.patients.map(p => 
      p.id === patientId ? { ...p, doc_ids: Array.from(new Set([...p.doc_ids, docId])) } : p
    )
  })),
  
  removeDocFromPatient: (patientId, docId) => set((state) => ({
    patients: state.patients.map(p => 
      p.id === patientId ? { ...p, doc_ids: p.doc_ids.filter(id => id !== docId) } : p
    )
  })),

  deleteDocFromPatient: async (docId: string) => {
    try {
      const res = await fetch(`${API_BASE}/ingest/${docId}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        set((state) => ({
          patients: state.patients.map(p => ({
            ...p,
            doc_ids: p.doc_ids.filter(id => id !== docId)
          }))
        }));
      }
    } catch (err) {
      console.error("Failed to delete document:", err);
    }
  },

  selectedPatientId: null,
  setSelectedPatientId: (id) => set({ selectedPatientId: id }),
}));
