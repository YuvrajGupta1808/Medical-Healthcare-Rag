"use client";
import React from "react";
import { 
  Users, Activity, Database, Server, ShieldCheck, ArrowRight,
  MessageSquare, FileText, UploadCloud, AlertTriangle, CheckCircle2,
  ListPlus
} from "lucide-react";
import { useStore } from "@/store/useStore";
import Link from "next/link";
import { useState, useEffect } from "react";

interface SystemStats {
  retrieval_analytics: {
    total_chunks_indexed: number;
    avg_retrieval_time_ms: number;
  };
  system_config: {
    vector_database: string;
    ai_model: string;
    orchestration: string;
    system_status: string;
  };
}

export default function DashboardOverview() {
  const patients = useStore((state) => state.patients);
  const totalDocs = patients.reduce((acc, p) => acc + (p.doc_ids?.length || 0), 0);
  
  const [stats, setStats] = useState<SystemStats | null>(null);

  useEffect(() => {
    fetch("http://localhost:8000/system/stats")
      .then(res => res.json())
      .then(json => setStats(json.data))
      .catch(err => console.error("Failed to fetch stats", err));
  }, []);

  return (
    <div className="space-y-5 max-w-6xl mx-auto pb-4 w-full pt-4">
      
      {/* Header & AI Badge */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-extrabold text-gray-900 tracking-tight">Clinical Intelligence Overview</h2>
          <p className="text-xs text-gray-500 mt-1 font-medium">Real-time metrics from the MedQuery engine.</p>
        </div>
      </div>

      {/* Top Action Banner */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Link href="/dashboard/chat" className="group bg-white border border-gray-200 hover:border-blue-500 hover:shadow-md hover:-translate-y-1 transition-all rounded-xl p-4 flex items-center justify-between">
           <div className="flex items-center gap-3">
             <div className="bg-blue-50 text-blue-600 p-2 rounded-lg group-hover:bg-blue-600 group-hover:text-white transition-colors">
               <MessageSquare size={18} />
             </div>
             <span className="font-bold text-sm text-gray-900">Start Clinical Query</span>
           </div>
           <ArrowRight size={16} className="text-gray-400 group-hover:text-blue-600 transition-colors" />
        </Link>
        <Link href="/dashboard/ingest" className="group bg-white border border-gray-200 hover:border-purple-500 hover:shadow-md hover:-translate-y-1 transition-all rounded-xl p-4 flex items-center justify-between">
           <div className="flex items-center gap-3">
             <div className="bg-purple-50 text-purple-600 p-2 rounded-lg group-hover:bg-purple-600 group-hover:text-white transition-colors">
               <UploadCloud size={18} />
             </div>
             <span className="font-bold text-sm text-gray-900">Index Documents</span>
           </div>
           <ArrowRight size={16} className="text-gray-400 group-hover:text-purple-600 transition-colors" />
        </Link>
        <Link href="/dashboard/patients" className="group bg-white border border-gray-200 hover:border-emerald-500 hover:shadow-md hover:-translate-y-1 transition-all rounded-xl p-4 flex items-center justify-between">
           <div className="flex items-center gap-3">
             <div className="bg-emerald-50 text-emerald-600 p-2 rounded-lg group-hover:bg-emerald-600 group-hover:text-white transition-colors">
               <Users size={18} />
             </div>
             <span className="font-bold text-sm text-gray-900">View Patient Context</span>
           </div>
           <ArrowRight size={16} className="text-gray-400 group-hover:text-emerald-600 transition-colors" />
        </Link>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="bg-white p-4 lg:p-5 rounded-xl border border-gray-100 shadow-sm flex flex-col justify-center hover:-translate-y-1 hover:shadow-md transition-all cursor-default">
           <div className="flex items-center justify-between mb-2">
             <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Patients Loaded</span>
             <Users size={18} className="text-blue-500" />
           </div>
           <div className="text-3xl font-black text-gray-900">{patients.length}</div>
        </div>

        <div className={`p-4 lg:p-5 rounded-xl border ${totalDocs === 0 ? 'bg-amber-50/50 border-amber-200' : 'bg-white border-gray-100'} shadow-sm flex flex-col justify-center hover:-translate-y-1 hover:shadow-md transition-all cursor-default`}>
           <div className="flex items-center justify-between mb-2">
             <span className={`text-xs font-bold ${totalDocs === 0 ? 'text-amber-700' : 'text-gray-500'} uppercase tracking-wider`}>Documents Indexed</span>
             {totalDocs === 0 ? <AlertTriangle size={18} className="text-amber-500" /> : <Database size={18} className="text-emerald-500" />}
           </div>
           <div className="text-3xl font-black text-gray-900 flex flex-col sm:flex-row sm:items-center gap-2 lg:gap-3">
             {totalDocs} 
             {totalDocs === 0 && <span className="text-[10px] sm:text-[11px] mt-1 sm:mt-0 font-bold text-amber-700 bg-amber-200/50 px-2 py-0.5 rounded-md">⚠️ Upload to enable RAG</span>}
           </div>
        </div>

        <div className="bg-white p-4 lg:p-5 rounded-xl border border-gray-100 shadow-sm flex flex-col justify-center hover:-translate-y-1 hover:shadow-md transition-all cursor-default">
           <div className="flex items-center justify-between mb-2">
             <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">System Status</span>
             <Activity size={18} className="text-emerald-500" />
           </div>
           <div className={`text-xl font-black ${stats?.system_config.system_status === 'Operational' ? 'text-emerald-600' : 'text-amber-600'} flex items-center gap-2 mt-1`}>
              <CheckCircle2 size={20} /> {stats?.system_config.system_status || "Loading..."}
           </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-2">
        
        {/* Elite CTA Card */}
        <div className="bg-gradient-to-br from-blue-600 to-indigo-900 p-8 rounded-2xl shadow-xl shadow-blue-900/20 text-white relative overflow-hidden flex flex-col justify-between transform transition hover:-translate-y-1 hover:shadow-2xl">
          <div className="absolute top-[-20%] right-[-10%] w-64 h-64 bg-white/10 rounded-full blur-3xl z-0 pointer-events-none" />
          <div className="absolute bottom-[-10%] left-[-10%] w-48 h-48 bg-blue-400/20 rounded-full blur-2xl z-0 pointer-events-none" />
          
          <div className="relative z-10">
            <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur-md border border-white/20 px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest mb-4 shadow-sm">
               <ShieldCheck size={12} /> Ready for Queries
            </div>
            <h3 className="text-2xl lg:text-3xl font-extrabold tracking-tight mb-3 leading-tight text-white drop-shadow-md">
              Launch Clinical Copilot
            </h3>
            
            {patients.length === 0 ? (
               <div className="bg-white/10 backdrop-blur-md rounded-xl p-4 border border-white/20 mt-3 max-w-sm shadow-sm">
                 <h4 className="font-bold text-xs uppercase tracking-wide mb-2 flex items-center gap-2"><ListPlus size={14}/> To begin:</h4>
                 <ol className="text-blue-50 text-xs font-medium space-y-1.5 list-decimal list-inside">
                   <li>Register a patient context</li>
                   <li>Upload clinical documents</li>
                   <li>Launch Copilot for analysis</li>
                 </ol>
               </div>
            ) : (
               <p className="text-blue-100/90 text-sm font-medium leading-relaxed max-w-md">
                 You have {patients.length} patient contexts actively synchronized in memory. Launch the terminal to securely query across {totalDocs} indexed document chunks.
               </p>
            )}

          </div>

          <div className="relative z-10 mt-8">
            <Link href="/dashboard/chat" className="inline-flex items-center justify-center gap-2 bg-white text-blue-900 px-6 py-3 rounded-xl font-black hover:scale-[1.03] hover:shadow-xl transition-all focus:outline-none focus:ring-4 focus:ring-white/30 text-sm w-full sm:w-auto shadow-md">
              <MessageSquare size={18} className="text-blue-600" /> Open Copilot <ArrowRight size={18} />
            </Link>
          </div>
        </div>

        <div className="space-y-4 flex flex-col justify-center">
          {/* Retrieval Stats */}
          <div className="bg-white p-5 rounded-2xl border border-gray-100 shadow-sm hover:-translate-y-1 hover:shadow-md transition-all">
             <h3 className="text-sm font-bold text-gray-900 mb-3 flex items-center gap-2">
               <FileText size={16} className="text-blue-500" /> Retrieval Analytics
             </h3>
             <ul className="space-y-3">
               <li className="flex justify-between items-center text-xs border-b border-gray-50 pb-2">
                 <span className="text-gray-500 font-semibold">Total Chunks Indexed</span>
                 <span className="font-bold text-gray-900 text-sm">{stats?.retrieval_analytics.total_chunks_indexed || 0} <span className="text-gray-400 text-[10px] font-normal">est.</span></span>
               </li>
               <li className="flex justify-between items-center text-xs border-b border-gray-50 pb-2">
                 <span className="text-gray-500 font-semibold">Avg Retrieval Time</span>
                 <span className="font-bold text-emerald-600 text-sm">~{stats?.retrieval_analytics.avg_retrieval_time_ms || 150}ms</span>
               </li>
               <li className="flex justify-between items-center text-xs pb-1">
                 <span className="text-gray-500 font-semibold">Top Grounding Sources</span>
                 <span className="font-bold text-gray-900">Clinical History, Lab PDFs</span>
               </li>
             </ul>
          </div>

          {/* System Config */}
          <div className="bg-white p-5 rounded-2xl border border-gray-100 shadow-sm hover:-translate-y-1 hover:shadow-md transition-all">
             <div className="flex items-center justify-between mb-3">
               <h3 className="text-sm font-bold text-gray-900 flex items-center gap-2">
                 <Server size={16} className="text-purple-500" /> System Configuration
               </h3>
             </div>
             <ul className="space-y-3">
               <li className="flex justify-between items-center text-xs border-b border-gray-50 pb-2">
                 <span className="text-gray-500 font-semibold">Vector Database</span>
                 <span className="font-bold text-gray-900 bg-gray-100 px-2 py-0.5 rounded-md text-[10px] tracking-wide">{stats?.system_config.vector_database || "Weaviate"}</span>
               </li>
               <li className="flex justify-between items-center text-xs border-b border-gray-50 pb-2">
                 <span className="text-gray-500 font-semibold">AI Model</span>
                 <span className="font-bold text-gray-900 bg-gray-100 px-2 py-0.5 rounded-md text-[10px] tracking-wide">{stats?.system_config.ai_model || "Gemini Pro"}</span>
               </li>
               <li className="flex justify-between items-center text-xs pb-1">
                 <span className="text-gray-500 font-semibold">Orchestration</span>
                 <span className="font-bold text-gray-900 bg-gray-100 px-2 py-0.5 rounded-md text-[10px] tracking-wide">{stats?.system_config.orchestration || "LangGraph"}</span>
               </li>
             </ul>
          </div>
        </div>

      </div>

    </div>
  );
}
