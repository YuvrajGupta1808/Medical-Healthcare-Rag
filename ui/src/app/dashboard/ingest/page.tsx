"use client";
import React, { useState } from "react";
import { useStore } from "@/store/useStore";
import { UploadCloud, FileText, Trash2, ShieldCheck, Search } from "lucide-react";

export default function IngestionEngineView() {
  const { patients, selectedPatientId, setSelectedPatientId, addDocToPatient, deleteDocFromPatient } = useStore();
  const [uploading, setUploading] = useState(false);

  const API_URL = "http://localhost:8000";

  React.useEffect(() => {
    if (!selectedPatientId && patients.length > 0) {
      setSelectedPatientId(patients[0].id);
    }
  }, [patients, selectedPatientId, setSelectedPatientId]);

  const handleFileUpload = async (e: any) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!selectedPatientId) return alert("❌ Please select a patient context first (top-right of ingestion panel).");

    setUploading(true);
    const formData = new FormData();
    const docUuid = `doc_${Date.now()}`;
    formData.append("file", file);
    formData.append("doc_id", docUuid);
    formData.append("doc_title", file.name);
    formData.append("patient_id", selectedPatientId);

    try {
      const resp = await fetch(`${API_URL}/ingest`, {
        method: "POST",
        body: formData,
      });
      if (resp.status === 201) {
        const result = await resp.json();
        // Link doc ID to selected patient, confirming mapping to the backend API response
        addDocToPatient(selectedPatientId, result.doc_id);
      } else {
        const errData = await resp.json().catch(() => ({ detail: "Unknown backend error" }));
        alert(`❌ Ingestion failed: ${errData.detail || "Check server logs."}`);
      }
    } catch (err) {
      alert("❌ Engine Connection Error: Please ensure your FastAPI backend is running and rebuilt with the latest ingestion logic.");
    } finally {
       setUploading(false);
    }
  };

  const handleDelete = async (docId: string) => {
    if(confirm(`Are you sure you want to delete ${docId}?`)) {
      await deleteDocFromPatient(docId);
    }
  }

  // Flatten out mapping for the data table
  const allDocs = patients.flatMap(p => p.doc_ids.map(docId => ({ patientId: p.id, patientName: p.name, docId })));

  return (
    <div className="space-y-6">
      
      <div className="bg-white p-8 rounded-2xl border border-gray-100 shadow-sm">
        <div className="flex justify-between items-center mb-6">
           <div>
              <h2 className="text-2xl font-extrabold tracking-tight text-gray-900">Ingestion Control</h2>
              <p className="text-gray-500 text-sm mt-1">Upload institutional documentation securely isolated by patient vectors.</p>
           </div>
           
           <div className="flex items-center gap-3">
             <label className="text-sm font-semibold text-gray-700">Scope Patient Context:</label>
             <select 
               className="border border-gray-200 rounded-lg px-4 py-2 font-medium text-gray-900 bg-white"
               value={selectedPatientId || ''} 
               onChange={(e) => setSelectedPatientId(e.target.value)}
             >
                {patients.map(p => <option key={p.id} value={p.id}>{p.name} (P-ID: {p.id})</option>)}
             </select>
           </div>
        </div>

        <label className={`block border-2 border-dashed ${uploading ? 'border-primary bg-primary/5' : 'border-gray-300 hover:border-primary'} rounded-2xl p-16 text-center transition cursor-pointer`}>
          <input type="file" className="hidden" onChange={handleFileUpload} disabled={uploading} multiple />
          <UploadCloud size={48} className="mx-auto text-primary mb-4" />
          <h3 className="text-xl font-bold text-gray-900">{uploading ? "Transmitting & Vectorizing..." : "Drag files or click here"}</h3>
          <p className="text-gray-500 mt-2 text-sm">Supports PDF, JPG, PNG, Audio streams up to 50MB. Embedded directly into Weaviate isolated spaces.</p>
        </label>
      </div>

      {/* Docs Data Table */}
      <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden mt-8">
        <div className="p-5 border-b border-gray-100 flex items-center justify-between">
            <h3 className="text-lg font-bold text-gray-900">Indexed Documents</h3>
           <div className="relative">
             <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
             <input type="text" placeholder="Search doc schemas..." className="pl-9 pr-4 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-1 text-gray-900 bg-white" />
           </div>
        </div>
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-gray-50/50 border-b border-gray-100 text-xs font-semibold text-gray-500 uppercase tracking-wider">
              <th className="px-6 py-4">Document Title (ID)</th>
              <th className="px-6 py-4">Mapped Patient</th>
              <th className="px-6 py-4">Ingestion Status</th>
              <th className="px-6 py-4 text-center">Manage</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {allDocs.length === 0 ? (
               <tr><td colSpan={4} className="p-8 text-center text-gray-500">No documents ingested onto vectors yet.</td></tr>
            ) : (
              allDocs.map((doc, i) => (
                <tr key={i} className="hover:bg-gray-50/50 transition">
                  <td className="px-6 py-4 flex items-center gap-3">
                    <FileText size={18} className="text-primary" />
                    <span className="font-semibold text-gray-800">{doc.docId}</span>
                  </td>
                  <td className="px-6 py-4 text-sm font-bold text-black">{doc.patientName} <span className="text-gray-400 font-medium">({doc.patientId})</span></td>
                  <td className="px-6 py-4 text-sm">
                    <span className="flex items-center gap-1 text-emerald-600 bg-emerald-50 px-2.5 py-1 rounded-full w-max font-semibold text-xs"><ShieldCheck size={14}/> Indexed</span>
                  </td>
                  <td className="px-6 py-4 text-center flex justify-center">
                    <button 
                       onClick={() => handleDelete(doc.docId)}
                       className="text-red-400 hover:text-red-600 p-1.5 hover:bg-red-50 rounded transition"
                    ><Trash2 size={16} /></button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

    </div>
  );
}
