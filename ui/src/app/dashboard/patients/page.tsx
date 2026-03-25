"use client";
import React, { useState } from "react";
import { useStore } from "@/store/useStore";
import { useRouter } from "next/navigation";
import { Users, Plus, Search, MoreHorizontal, User, Trash2, FileUp } from "lucide-react";

export default function PatientsView() {
  const { patients, createPatient, deletePatient, setSelectedPatientId, isLoading } = useStore();
  const [showAdd, setShowAdd] = useState(false);
  const [activeDropdown, setActiveDropdown] = useState<string | null>(null);
  const [newPatient, setNewPatient] = useState({ name: "", age: 0, gender: "Male" });
  const router = useRouter();

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (newPatient.name) {
      await createPatient({
        id: `p${Date.now()}`,
        name: newPatient.name,
        age: newPatient.age,
        gender: newPatient.gender
      });
      setShowAdd(false);
      setNewPatient({ name: "", age: 0, gender: "Male" });
    }
  };

  return (
    <div className="space-y-6">
      {/* Header section */}
      <div className="flex justify-between items-center bg-white p-6 rounded-2xl border border-gray-100 shadow-sm">
        <div>
          <h2 className="text-2xl font-extrabold tracking-tight text-gray-900">Patient Directory</h2>
          <p className="text-gray-500 text-sm mt-1">Manage institutional clinical records mapping</p>
        </div>
        <button 
          onClick={() => setShowAdd(!showAdd)}
          className="bg-primary hover:bg-primary/90 text-white px-5 py-2.5 rounded-xl font-semibold flex items-center gap-2 transition"
        >
          <Plus size={18} /> Add Patient
        </button>
      </div>

      {/* Add Patient form overlay */}
      {showAdd && (
        <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm mb-6 max-w-2xl">
          <h3 className="text-lg font-bold mb-4 text-gray-900">Register New Patient</h3>
          <form onSubmit={handleCreate} className="space-y-4">
             <div className="grid grid-cols-2 gap-4">
               <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-1">Full Name</label>
                  <input type="text" value={newPatient.name} onChange={e => setNewPatient({...newPatient, name: e.target.value})} className="w-full px-4 py-2 border border-gray-200 rounded-xl text-gray-900 bg-white focus:outline-none focus:border-blue-500" required />
               </div>
               <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-1">Age</label>
                  <input type="number" value={newPatient.age || ''} onChange={e => setNewPatient({...newPatient, age: parseInt(e.target.value)})} className="w-full px-4 py-2 border border-gray-200 rounded-xl text-gray-900 bg-white focus:outline-none focus:border-blue-500" required />
               </div>
             </div>
             <div>
                <label className="block text-sm font-semibold text-gray-700 mb-1">Biological Gender</label>
                <select value={newPatient.gender} onChange={e => setNewPatient({...newPatient, gender: e.target.value})} className="w-full px-4 py-2 border border-gray-200 rounded-xl text-gray-900 bg-white focus:outline-none focus:border-blue-500">
                  <option>Male</option>
                  <option>Female</option>
                  <option>Other</option>
                </select>
             </div>
             <div className="flex justify-end gap-3 mt-4">
               <button type="button" onClick={() => setShowAdd(false)} className="px-4 py-2 rounded-xl text-gray-600 border hover:bg-gray-50">Cancel</button>
               <button type="submit" className="px-4 py-2 rounded-xl bg-primary text-white font-semibold">Confirm Registration</button>
             </div>
          </form>
        </div>
      )}

      {/* Patients Data Table */}
      <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
        <div className="p-4 border-b border-gray-100 flex items-center justify-between bg-gray-50/50">
           <div className="relative">
             <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
             <input type="text" placeholder="Filter patients..." className="pl-9 pr-4 py-2 border border-gray-200 rounded-lg text-sm bg-white focus:outline-none focus:ring-1 focus:ring-primary" />
           </div>
        </div>
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-gray-50/50 border-b border-gray-100 text-xs font-semibold text-gray-500 uppercase tracking-wider">
              <th className="px-6 py-4">Patient Profile</th>
              <th className="px-6 py-4">P-ID</th>
              <th className="px-6 py-4">Age / Gender</th>
              <th className="px-6 py-4">Linked Docs</th>
              <th className="px-6 py-4 text-center">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {patients.map((p) => (
              <tr key={p.id} className="hover:bg-gray-50/50 transition duration-150">
                <td className="px-6 py-4 flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold">
                    {p.name.charAt(0)}
                  </div>
                  <div>
                    <span className="font-bold text-gray-900 block">{p.name}</span>
                    <span className="text-xs text-gray-500">Registered locally</span>
                  </div>
                </td>
                <td className="px-6 py-4 text-sm font-medium text-gray-600">{p.id}</td>
                <td className="px-6 py-4 text-sm text-gray-600">{p.age} y/o • {p.gender}</td>
                <td className="px-6 py-4 text-sm">
                  <span className="px-2.5 py-1 bg-primary/10 text-primary rounded-full font-semibold">{p.doc_ids.length} docs</span>
                </td>
                <td className="px-6 py-4 text-center relative">
                  <button 
                    onClick={() => setActiveDropdown(activeDropdown === p.id ? null : p.id)}
                    className="text-gray-400 hover:text-gray-900 transition p-1 hover:bg-gray-100 rounded-lg"
                  >
                    <MoreHorizontal size={18} />
                  </button>
                  
                  {activeDropdown === p.id && (
                    <div className="absolute right-10 top-10 w-44 bg-white border border-gray-100 shadow-xl rounded-xl flex flex-col py-1 z-50 text-left">
                       <button 
                         onClick={() => { setSelectedPatientId(p.id); router.push('/dashboard/ingest'); }}
                         className="px-4 py-2.5 text-sm font-semibold text-gray-700 hover:bg-gray-50 flex items-center gap-2"
                       >
                         <FileUp size={16} className="text-blue-500" /> Upload Docs
                       </button>
                       <button 
                         onClick={() => {
                           if(confirm(`Remove patient ${p.name}?`)) {
                             deletePatient(p.id);
                             setActiveDropdown(null);
                           }
                         }}
                         className="px-4 py-2.5 text-sm font-semibold text-red-600 hover:bg-red-50 flex items-center gap-2 border-t border-gray-50"
                       >
                         <Trash2 size={16} /> Remove Patient
                       </button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
