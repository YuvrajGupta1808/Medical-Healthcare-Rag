"use client";
import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { useStore } from "@/store/useStore";
import { ArrowRight, Activity, ShieldCheck, Database, Layers, Cloud } from "lucide-react";
import { motion } from "framer-motion";

export default function LoginPage() {
  const router = useRouter();
  const setDoctorName = useStore((state) => state.setDoctorName);
  const [name, setName] = useState("Dr. Sarah Chen");

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (name.trim()) {
      setDoctorName(name);
      router.push("/dashboard");
    }
  };

  return (
    <div className="min-h-screen bg-white text-gray-900 font-sans overflow-hidden flex flex-col relative selection:bg-blue-100">

      {/* Subtle Data Grid Pattern */}
      <div className="absolute inset-0 z-0 opacity-[0.03] pointer-events-none" 
           style={{ backgroundImage: 'linear-gradient(to right, #3b82f6 1px, transparent 1px), linear-gradient(to bottom, #3b82f6 1px, transparent 1px)', backgroundSize: '60px 60px' }} 
      />
      {/* Glowing Meteor Lines on the Left */}
      <div className="absolute top-0 left-[8%] w-[1px] h-full bg-gradient-to-b from-white via-blue-500 to-white opacity-20 z-0 blur-[1px]" />
      <div className="absolute top-0 left-[18%] w-[2px] h-full bg-gradient-to-b from-white via-blue-400 to-white opacity-10 z-0 blur-[2px]" />

      {/* Main Hero Section */}
      <main className="flex-1 flex flex-col justify-center max-w-7xl mx-auto w-full px-6 lg:px-8 relative z-10 py-12 lg:py-0">
         <div className="w-full grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            
            {/* Left Column: Typography & Login Input */}
            <motion.div 
               initial={{ opacity: 0, y: 20 }}
               animate={{ opacity: 1, y: 0 }}
               transition={{ duration: 0.8 }}
               className="flex flex-col items-start z-10 mt-12 lg:mt-0"
            >
               <div className="flex items-center gap-3 mb-8">
                 <div className="w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center text-white shadow-lg shadow-blue-600/20">
                   <Activity size={24} />
                 </div>
                 <span className="font-extrabold text-2xl tracking-tighter text-blue-900">MedQuery</span>
               </div>

               <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight leading-[1.1] mb-6 text-gray-900">
                 Clinical <br/> Intelligence <br/>
                 <span className="text-blue-600">Engine</span>
               </h1>
               
               <p className="text-lg text-gray-500 mb-10 max-w-md leading-relaxed">
                 MedQuery is a state-of-the-art hybrid RAG architecture built to securely index complex institutional medical data and surface real-time actionable insights.
               </p>

               <form onSubmit={handleLogin} className="w-full max-w-md">
                 <label className="block text-sm font-bold text-gray-700 mb-2">Authorized Provider Access</label>
                 <div className="flex flex-col sm:flex-row items-center gap-3 p-2 bg-gray-50 border border-gray-200 rounded-2xl focus-within:ring-2 focus-within:ring-blue-500/20 focus-within:border-blue-500 transition-all shadow-sm">
                   <input
                     type="text"
                     value={name}
                     onChange={(e) => setName(e.target.value)}
                     className="w-full bg-transparent border-none outline-none px-4 py-3 text-gray-900 font-medium placeholder-gray-400"
                     placeholder="e.g. Dr. Sarah Chen"
                     required
                   />
                   <button
                     type="submit"
                     className="w-full sm:w-auto py-3 px-6 bg-blue-600 text-white rounded-xl font-bold flex items-center justify-center gap-2 hover:bg-blue-700 transition shadow-md shadow-blue-600/20 whitespace-nowrap"
                   >
                     Access Platform <ArrowRight size={18} />
                   </button>
                 </div>
               </form>
            </motion.div>

            {/* Right Column: Doctor Hero Asset */}
            <motion.div 
               initial={{ opacity: 0, x: 20 }}
               animate={{ opacity: 1, x: 0 }}
               transition={{ duration: 1, delay: 0.2, ease: "easeOut" }}
               className="relative lg:h-[650px] w-full hidden lg:flex items-center justify-end"
            >
               {/* Background abstraction blob */}
               <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full bg-blue-50 rounded-full blur-3xl z-0" />
               <div className="absolute top-1/4 right-0 w-2/3 h-2/3 bg-blue-100 rounded-full blur-3xl z-0" />
               
               {/* Wrapper to control Image and Card offset */}
               <div className="relative -translate-x-8 lg:-translate-x-16">
                 {/* Image Container */}
                 <div className="relative w-[360px] lg:w-[420px] h-[480px] z-10 rounded-[2.5rem] overflow-hidden shadow-2xl border-[8px] border-white">
                   <div 
                      className="w-full h-full bg-cover bg-center"
                      style={{ backgroundImage: "url('/doctor-hero.png')" }}
                   />
                   <div className="absolute inset-0 bg-gradient-to-t from-blue-900/60 via-transparent to-transparent mix-blend-multiply" />
                 </div>
                 
                 {/* Floating aesthetic card mapped tight to bottom left corner */}
                 <div className="absolute -bottom-8 -left-8 bg-white p-5 rounded-2xl shadow-[0_10px_40px_-10px_rgba(0,0,0,0.1)] border border-gray-100 flex items-center gap-4 z-20">
                   <div className="w-12 h-12 bg-green-50 rounded-full flex items-center justify-center text-green-600">
                      <ShieldCheck size={24} />
                   </div>
                   <div>
                      <h4 className="font-bold text-gray-900 text-sm">HIPAA Compliant</h4>
                      <p className="text-xs text-gray-500 font-medium">Secured Node Vectors</p>
                   </div>
                 </div>
               </div>
            </motion.div>
         </div>
      </main>

      {/* Bottom Logos Section matching requested tech stack */}
      <div className="w-full bg-white/80 border-t border-gray-100 py-8 backdrop-blur-sm relative z-10">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-6 md:gap-0">
           <div className="text-[11px] uppercase tracking-[0.2em] font-bold text-gray-400">
             Built on modern cloud infrastructure
           </div>
           <div className="flex flex-wrap items-center justify-center gap-8 md:gap-12 text-gray-400">
              <span className="flex items-center gap-2 font-bold text-lg tracking-tight hover:text-blue-600 transition cursor-default opacity-70 hover:opacity-100">
                <Cloud size={20} className="text-blue-500"/> Azure
              </span>
              <span className="flex items-center gap-2 font-bold text-lg tracking-tight hover:text-[#844FBA] transition cursor-default opacity-70 hover:opacity-100">
                <Layers size={20} className="text-[#844FBA]"/> Terraform
              </span>
              <span className="flex items-center gap-2 font-bold text-lg tracking-tight hover:text-[#2496ED] transition cursor-default opacity-70 hover:opacity-100">
                <Database size={20} className="text-[#2496ED]"/> Docker
              </span>
              <span className="flex items-center gap-2 font-bold text-lg tracking-tight hover:text-emerald-500 transition cursor-default opacity-70 hover:opacity-100">
                <Database size={20} className="text-emerald-500"/> Weaviate
              </span>
           </div>
        </div>
      </div>
      
    </div>
  );
}
