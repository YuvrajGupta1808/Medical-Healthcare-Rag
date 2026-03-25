"use client";
import React, { useState, useRef } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useStore } from "@/store/useStore";
import { 
  LayoutDashboard, Users, FileUp, MessageSquare, 
  Settings, LogOut, Activity, Bell, Search,
  ShieldAlert
} from "lucide-react";
import { useEffect } from "react";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { doctorName, fetchPatients, checkHealth, backendStatus } = useStore();

  useEffect(() => {
    fetchPatients();
    
    // Poll health every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [fetchPatients, checkHealth]);

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [headerOpen, setHeaderOpen] = useState(false);

  const sidebarTimer = useRef<NodeJS.Timeout>();
  const headerTimer = useRef<NodeJS.Timeout>();

  const handleSidebarEnter = () => {
    clearTimeout(sidebarTimer.current);
    setSidebarOpen(true);
  };
  
  const handleSidebarLeave = () => {
    // Stays open for 6000ms before naturally gracefully sweeping away
    sidebarTimer.current = setTimeout(() => setSidebarOpen(false), 6000); 
  };

  const handleHeaderEnter = () => {
    clearTimeout(headerTimer.current);
    setHeaderOpen(true);
  };
  
  const handleHeaderLeave = () => {
    headerTimer.current = setTimeout(() => setHeaderOpen(false), 6000); 
  };

  const navLinks = [
    { name: "Copilot", href: "/dashboard/chat", icon: MessageSquare },
    { name: "Patients", href: "/dashboard/patients", icon: Users },
    { name: "Ingestion", href: "/dashboard/ingest", icon: FileUp },
    { name: "Overview", href: "/dashboard", icon: LayoutDashboard },
  ];

  return (
    <div className="flex h-screen bg-slate-100 overflow-hidden font-sans relative">
      
      {/* 
        Side Navigation - Stateful Push
      */}
      <div 
        onMouseEnter={handleSidebarEnter}
        onMouseLeave={handleSidebarLeave}
        className={`fixed left-0 top-0 h-full bg-white/95 backdrop-blur-3xl border-r border-gray-200 shadow-2xl flex flex-col transition-all duration-500 ease-[cubic-bezier(0.23,1,0.32,1)] z-[100] ${
          sidebarOpen ? "w-64 translate-x-0" : "w-64 -translate-x-[calc(100%-12px)] opacity-60 hover:opacity-100"
        }`}
      >
        <div className="p-6 pb-2 flex items-center gap-3 mt-4 shrink-0">
          <div className="w-10 h-10 bg-blue-50 text-blue-600 rounded-xl flex items-center justify-center shadow-sm shrink-0">
            <Activity size={20} />
          </div>
          <span className="text-xl font-bold text-gray-900 tracking-tight whitespace-nowrap">MedQuery</span>
        </div>

        <div className="px-6 py-4 shrink-0">
          <button className="w-full bg-white border border-gray-200 hover:bg-gray-50 text-gray-700 rounded-xl py-2.5 px-4 font-semibold text-sm shadow-sm transition flex justify-between items-center group/btn whitespace-nowrap truncate">
            Register Patient <span className="text-gray-400 group-hover/btn:text-blue-600 transition-colors font-bold">+</span>
          </button>
        </div>

        <nav className="flex-1 px-4 py-4 space-y-1 overflow-hidden hover:overflow-y-auto">
          {navLinks.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link 
                key={link.name} 
                href={link.href}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 whitespace-nowrap ${
                  isActive 
                  ? "bg-blue-50 text-blue-600 font-semibold" 
                  : "text-gray-500 hover:bg-gray-50 hover:text-gray-900 font-medium"
                }`}
              >
                <link.icon size={18} className={isActive ? "text-blue-500" : "text-gray-400"} />
                {link.name}
              </Link>
            );
          })}
        </nav>

        <div className="p-4 mt-auto space-y-1 border-t border-gray-100 bg-white/90 shrink-0">
          <button className="flex items-center gap-3 px-4 py-3 rounded-xl w-full text-left text-gray-500 hover:bg-gray-50 hover:text-gray-900 font-medium transition-colors whitespace-nowrap">
            <Settings size={18} className="text-gray-400 shrink-0" /> Settings
          </button>
          <Link href="/" className="flex items-center gap-3 px-4 py-3 rounded-xl w-full text-left text-gray-500 hover:bg-red-50 hover:text-red-600 font-medium transition-colors whitespace-nowrap">
            <LogOut size={18} className="text-gray-400 shrink-0" /> Logout
          </Link>
        </div>
      </div>

      {/* 
        Top Navigation - Stateful Push
      */}
      <div 
        onMouseEnter={handleHeaderEnter}
        onMouseLeave={handleHeaderLeave}
        className={`fixed top-0 left-0 w-full h-20 bg-white/95 backdrop-blur-3xl shadow-[0_20px_40px_-15px_rgba(0,0,0,0.1)] border-b border-gray-200 flex items-center justify-between px-8 transition-all duration-500 ease-[cubic-bezier(0.23,1,0.32,1)] z-[90] ${
          headerOpen ? "translate-y-0" : "-translate-y-[calc(100%-12px)] opacity-60 hover:opacity-100"
        } ${sidebarOpen ? 'pl-64' : 'pl-10'}`}
      >
        <div className="flex-1 flex justify-center w-full pt-1">
          <div className="flex items-center bg-gray-50/80 border border-gray-200 rounded-2xl px-4 py-2.5 w-[32rem] shadow-inner hover:bg-white focus-within:bg-white focus-within:ring-2 focus-within:ring-blue-50 focus-within:border-blue-300 transition-all cursor-text group/search">
            <Search size={18} className="text-gray-400 group-hover/search:text-blue-500 transition-colors shrink-0" />
            <input 
              type="text" 
              placeholder="Search patients, clinical notes, or recent queries..." 
              className="w-full bg-transparent border-none outline-none text-sm px-3 text-gray-900 placeholder-gray-400 font-medium"
            />
            <span className="hidden sm:inline-flex bg-white text-gray-400 font-bold px-2 py-0.5 rounded-md text-[10px] tracking-widest border border-gray-200 shadow-sm shadow-black/5 shrink-0">⌘K</span>
          </div>
        </div>

        <div className="flex items-center gap-6 absolute right-8 pt-1 bg-white/50 backdrop-blur pl-4 rounded-xl">
          <button className="relative text-gray-400 hover:text-gray-600 transition shrink-0">
            <Bell size={20} />
            <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full border-2 border-white"></span>
          </button>
          <div className="flex items-center gap-3 pl-6 border-l border-gray-200">
            <div className="w-9 h-9 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold text-sm shadow-md ring-2 ring-blue-50 shrink-0">
              {doctorName?.charAt(0) || "D"}
            </div>
            <span className="text-sm font-bold text-black whitespace-nowrap">{doctorName}</span>
          </div>
        </div>
      </div>

      {/* Main Content Area - True Push/Squish Behavior */}
      <div 
        className={`flex-1 transition-all duration-500 ease-[cubic-bezier(0.23,1,0.32,1)] h-full relative flex ${
           sidebarOpen ? "ml-64" : "ml-[12px]" // 12px leaves room for the peek handle
        } ${
           headerOpen ? "mt-20" : "mt-[12px]" 
        }`}
      >
        <main className="flex-1 overflow-y-auto w-full h-full relative z-0 p-4 lg:p-5 lg:pl-6 bg-slate-100 transition-all rounded-tl-xl border-t border-l border-gray-200 shadow-inner overflow-hidden">
           <div className="mx-auto w-full max-w-[1600px] h-full flex flex-col">
             {children}
           </div>
        </main>
      </div>

    </div>
  );
}
