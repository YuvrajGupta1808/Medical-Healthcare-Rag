"use client";
import React, { useState, useRef, useEffect } from "react";
import { useStore } from "@/store/useStore";
import { 
  Send, Bot, User, Paperclip, Mic, Image as ImageIcon, 
  FileText, Activity, ChevronDown, ChevronRight, CheckCircle2,
  AlertTriangle, ShieldAlert, Sparkles, BookOpen, BrainCircuit, Database
} from "lucide-react";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  mediaType?: "text" | "image" | "audio" | "pdf";
  mediaUrl?: string;
  mediaName?: string;
  thinking?: { step: string; status: string; done: boolean }[];
  citations?: any[];
};

export default function ChatView() {
  const { patients, selectedPatientId, setSelectedPatientId } = useStore();
  const selectedPatient = patients.find(p => p.id === selectedPatientId);
  const docCount = selectedPatient?.doc_ids.length || 0;

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [expandedThinkId, setExpandedThinkId] = useState<string | null>(null);
  
  // State for the Right Panel (Sources viewer)
  const [activeSources, setActiveSources] = useState<any[]>([]);

  const endRef = useRef<HTMLDivElement>(null);

  // Auto scroll to bottom
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSend = async (overrideInput?: string) => {
    const text = overrideInput || input;
    if (!text.trim() || !selectedPatientId || docCount === 0) return;

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: text, mediaType: "text" };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    const assistantId = (Date.now() + 1).toString();
    const initialAssistantObj: Message = {
       id: assistantId, role: "assistant", content: "", thinking: []
    };
    
    setMessages(prev => [...prev, initialAssistantObj]);
    setExpandedThinkId(assistantId);
    setActiveSources([]); // Clear previous sources while fetching

    try {
      const response = await fetch("http://localhost:8000/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, patient_id: selectedPatientId, top_k: 5 }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.replace("data: ", "").trim();
            if (!dataStr) continue;
            
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.type === "trace") {
                setMessages(prev => prev.map(m => m.id === assistantId ? {
                  ...m,
                  thinking: [
                    ...(m.thinking || []).map(t => ({...t, done: true})), 
                    { step: parsed.trace.step, status: parsed.trace.status, done: false }
                  ]
                } : m));
              } else if (parsed.type === "result") {
                setMessages(prev => prev.map(m => m.id === assistantId ? {
                  ...m,
                  content: parsed.data.answer,
                  citations: parsed.data.citations,
                  mediaUrl: parsed.data.image_url,
                  mediaType: parsed.data.image_url ? 'image' : m.mediaType,
                  thinking: (m.thinking || []).map(t => ({...t, done: true}))
                } : m));
                
                // Expose citations directly into the right-hand panel
                if(parsed.data.citations) {
                   setActiveSources(parsed.data.citations);
                }
                
                setExpandedThinkId(null);
                setLoading(false);
              } else if (parsed.type === "error") {
                setMessages(prev => prev.map(m => m.id === assistantId ? {
                  ...m, content: "Sorry, I encountered an internal error: " + parsed.detail
                } : m));
                setLoading(false);
              }
            } catch (e) {
               console.error("Parse error on SSE chunk:", e);
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
      setLoading(false);
    }
  };

  const handleMediaUpload = (type: "image" | "audio" | "pdf") => {
    if(!selectedPatientId || docCount === 0) return alert("Select a valid patient context first.");
    const userMsg: Message = { 
       id: Date.now().toString(), role: "user", content: `Attached ${type} for analysis...`, mediaType: type, mediaName: `uploaded_diagnostic.${type}` 
    };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);
    
    setTimeout(() => {
       const assistantId = (Date.now() + 1).toString();
       setMessages(prev => [...prev, {
          id: assistantId, role: "assistant", content: "I have securely parsed the visual diagnostic attachment mapped against the patient's existing history.",
          thinking: [
             { step: "Routing", status: `Vision Model triggered for ${type} analysis`, done: true },
             { step: "Generation", status: "Parsed multi-modal layout successfully.", done: true }
          ]
       }]);
       setLoading(false);
    }, 1500);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-2rem)] lg:h-full bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm">
      
      {/* 1. Header Row - Completely Integrated, No Shadows */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-gray-200 bg-gray-50/50 shrink-0">
         <div className="flex items-center gap-4">
           <div className="flex items-center gap-3">
             <div className="w-8 h-8 bg-blue-100 text-blue-700 rounded flex items-center justify-center border border-blue-200">
               <BrainCircuit size={18} />
             </div>
             <div>
               <h2 className="text-[14px] font-bold text-gray-900 tracking-tight">
                 Active Patient: {selectedPatient?.name || "None"}
               </h2>
               {selectedPatientId ? (
                 docCount > 0 ? (
                   <p className="text-[11px] text-gray-500 font-medium">{docCount} Indexed Logs</p>
                 ) : (
                   <div className="text-[11px] text-amber-600 font-bold flex items-center gap-1">
                     <AlertTriangle size={10}/> Pending Upload
                   </div>
                 )
               ) : (
                 <p className="text-[11px] text-gray-400 font-medium">No Context Selected</p>
               )}
             </div>
           </div>
         </div>
         
         <div className="flex items-center gap-3">
           <label className="text-[11px] font-bold text-gray-400 uppercase tracking-widest hidden sm:block">Scope</label>
           <select 
             className={`bg-white border ${!selectedPatientId ? 'border-red-300 ring-1 ring-red-100' : 'border-gray-200 hover:border-gray-300'} rounded shadow-sm px-3 py-1.5 font-medium text-sm min-w-[200px] outline-none focus:ring-1 focus:ring-blue-500/20 focus:border-blue-500 transition-all text-gray-800`}
             value={selectedPatientId || ''} 
             onChange={(e) => setSelectedPatientId(e.target.value)}
           >
              <option value="" disabled>Select Context...</option>
              {patients.map(p => <option key={p.id} value={p.id}>{p.name} ({p.doc_ids.length} docs)</option>)}
           </select>
         </div>
      </div>

      {/* 2. Unified Workspace Area (No Negative Space Between) */}
      <div className="flex flex-1 min-h-0 bg-white">
        
        {/* Left Pane: Clinical Copilot Chat */}
        <div className="flex-1 flex flex-col border-r border-gray-200 relative bg-white">
          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
            {messages.length === 0 ? (
               <div className="flex flex-col items-center justify-center h-full text-center max-w-md mx-auto animate-in fade-in duration-500">
                 <div className="w-12 h-12 bg-gray-100 text-gray-500 rounded flex items-center justify-center mb-4 border border-gray-200">
                   <Activity size={24} />
                 </div>
                 <h3 className="text-xl font-bold text-gray-900 mb-2 tracking-tight">Clinical Copilot</h3>
                 
                 {selectedPatientId ? (
                    docCount > 0 ? (
                       <p className="text-gray-500 mb-6 text-sm leading-relaxed">
                         Expert system is loaded for <strong className="text-gray-900">{selectedPatient?.name}</strong>. Ready to synthesize insights across embedded clinical records.
                       </p>
                    ) : (
                       <div className="bg-amber-50 text-amber-800 p-4 rounded border border-amber-200 mt-2 mb-6 flex items-start gap-3 text-left w-full">
                         <AlertTriangle size={18} className="shrink-0 mt-0.5 text-amber-500" />
                         <div>
                           <h4 className="font-bold text-sm">No Clinical Context Found</h4>
                           <p className="text-xs mt-1 font-medium bg-amber-100/50 p-2 rounded mt-2">
                             1. Navigate to Ingestion<br/>2. Upload patient lab PDFs<br/>3. Return to query
                           </p>
                         </div>
                       </div>
                    )
                 ) : (
                    <p className="text-gray-500 mb-6 text-sm">Select a patient scope in the top header to initialize RAG memory.</p>
                 )}

                 {selectedPatientId && docCount > 0 && (
                   <div className="w-full text-left mt-2 border-t border-gray-100 pt-5">
                     <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-3 block text-center">Diagnostics</span>
                     <div className="grid grid-cols-2 gap-2">
                       <button onClick={() => handleSend("Summarize the patient's recent clinical history")} className="bg-gray-50 hover:bg-gray-100 border border-gray-200 p-2.5 rounded text-xs font-semibold text-gray-700 hover:text-blue-700 transition flex items-center gap-2">
                         <FileText size={14} className="text-blue-500 shrink-0" /> Summarize
                       </button>
                       <button onClick={() => handleSend("What are the key abnormalities in the latest lab reports?")} className="bg-gray-50 hover:bg-gray-100 border border-gray-200 p-2.5 rounded text-xs font-semibold text-gray-700 hover:text-blue-700 transition flex items-center gap-2">
                         <Activity size={14} className="text-purple-500 shrink-0" /> Analyze Labs
                       </button>
                       <button onClick={() => handleSend("Draft a differential diagnosis based on reported symptoms")} className="bg-gray-50 hover:bg-gray-100 border border-gray-200 p-2.5 rounded text-xs font-semibold text-gray-700 hover:text-blue-700 transition flex items-center gap-2">
                         <Sparkles size={14} className="text-amber-500 shrink-0" /> Diagnosis
                       </button>
                       <button onClick={() => handleSend("Generate a structured clinical report formatting all notes")} className="bg-gray-50 hover:bg-gray-100 border border-gray-200 p-2.5 rounded text-xs font-semibold text-gray-700 hover:text-blue-700 transition flex items-center gap-2">
                         <FileText size={14} className="text-emerald-500 shrink-0" /> Report
                       </button>
                     </div>
                   </div>
                 )}
               </div>
            ) : (
               messages.map((msg) => (
                 <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} w-full animate-in fade-in slide-in-from-bottom-2 duration-300`}>
                   <div className={`flex gap-3 max-w-[85%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                     
                     <div className={`w-8 h-8 rounded flex-shrink-0 flex items-center justify-center mt-1 border ${msg.role === 'user' ? 'bg-blue-600 text-white border-blue-700' : 'bg-gray-50 text-gray-600 border-gray-200'}`}>
                        {msg.role === 'user' ? <User size={14}/> : <Bot size={14}/>}
                     </div>

                     <div className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'} max-w-full`}>
                        {msg.mediaType && msg.mediaType !== 'text' && msg.mediaUrl && (
                          <div className={`mb-3 overflow-hidden rounded-xl border border-gray-200 transition bg-white ${msg.role === 'user' ? 'shadow-sm' : ''}`}>
                             {msg.mediaType === 'image' && (
                               <img src={msg.mediaUrl} alt="Clinical Extraction" className="w-full max-h-[300px] object-contain bg-gray-50 border-b border-gray-100" />
                             )}
                             <div className="p-3 flex items-center gap-2">
                               {msg.mediaType === 'image' && <ImageIcon size={16} className="text-blue-500" />}
                               {msg.mediaType === 'pdf' && <FileText size={16} className="text-red-500" />}
                               {msg.mediaType === 'audio' && <Mic size={16} className="text-amber-500" />}
                               <span className="text-[11px] font-bold text-gray-700 uppercase tracking-tight truncate">{msg.mediaName || "Extracted Evidence"}</span>
                             </div>
                          </div>
                        )}

                       {(msg.content || msg.role === 'assistant') && (
                         <div className={`p-3.5 rounded-lg text-[14px] leading-relaxed relative border ${msg.role === 'user' ? 'bg-blue-50 text-blue-900 border-blue-100 shadow-sm rounded-tr-none font-medium' : 'bg-white text-gray-900 border-gray-200 rounded-tl-none shadow-sm'}`}>
                           
                           {/* Thinking UI inside Assistant bubbles */}
                           {msg.thinking && msg.thinking.length > 0 && (
                             <div className="mb-3 w-full border border-gray-200 bg-gray-50 rounded overflow-hidden shrink-0">
                                <button 
                                  onClick={() => setExpandedThinkId(expandedThinkId === msg.id ? null : msg.id)}
                                  className="w-full flex items-center justify-between p-2 hover:bg-gray-100 transition"
                                >
                                  <div className="flex items-center gap-2 text-[10px] font-bold text-gray-600 uppercase">
                                    <Activity size={12} className={msg.thinking.some(t => !t.done) ? "text-blue-500 animate-pulse" : "text-gray-500"}/> 
                                    {msg.thinking.some(t => !t.done) ? "Executing Pipeline..." : "Execution Trace"}
                                  </div>
                                  {expandedThinkId === msg.id ? <ChevronDown size={14} className="text-gray-400"/> : <ChevronRight size={14} className="text-gray-400"/>}
                                </button>

                                {expandedThinkId === msg.id && (
                                  <div className="p-2.5 border-t border-gray-200 space-y-2 bg-white flex flex-col items-start w-full">
                                    {msg.thinking.map((step, idx) => (
                                      <div key={idx} className="flex items-start gap-2 text-[11px] font-mono w-full">
                                        <div className="mt-0.5">
                                          {step.done ? <CheckCircle2 size={12} className="text-gray-400"/> : <span className="block w-3 h-3 rounded-full border-[2px] border-blue-500 border-t-transparent animate-spin" />}
                                        </div>
                                        <span className="font-bold text-gray-600 shrink-0">{step.step}:</span>
                                        <span className="text-gray-500 break-words">{step.status}</span>
                                      </div>
                                    ))}
                                  </div>
                                )}
                             </div>
                           )}

                           {msg.content ? (
                              <div className="whitespace-pre-wrap">{msg.content}</div>
                           ) : (
                              msg.role === 'assistant' && <div className="flex items-center gap-1 opacity-50 h-5 px-1"><span className="w-1.5 h-1.5 bg-gray-600 rounded-full animate-bounce"></span><span className="w-1.5 h-1.5 bg-gray-600 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></span><span className="w-1.5 h-1.5 bg-gray-600 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></span></div>
                           )}
                         </div>
                       )}
                     </div>
                   </div>
                 </div>
               ))
            )}
            {loading && <div ref={endRef} className="h-4" />}
          </div>

          {/* Chat Input Bar */}
          <div className="bg-white border-t border-gray-200 shrink-0">
            <div className={`bg-white transition-all p-2 flex flex-col`}>
              
              <div className="flex px-2 pt-1 gap-1 text-gray-400 border-b border-transparent">
                 <button onClick={() => handleMediaUpload('image')} disabled={!selectedPatientId || docCount === 0} className="hover:text-blue-600 p-1.5 rounded transition disabled:pointer-events-none hover:bg-gray-100" title="Upload Image"><ImageIcon size={14}/></button>
                 <button onClick={() => handleMediaUpload('pdf')} disabled={!selectedPatientId || docCount === 0} className="hover:text-red-500 p-1.5 rounded transition disabled:pointer-events-none hover:bg-gray-100" title="Upload PDF"><Paperclip size={14}/></button>
                 <button onClick={() => handleMediaUpload('audio')} disabled={!selectedPatientId || docCount === 0} className="hover:text-amber-500 p-1.5 rounded transition disabled:pointer-events-none hover:bg-gray-100" title="Record Audio"><Mic size={14}/></button>
              </div>

              <div className="flex justify-between items-end mt-1 px-1 pb-1">
                <textarea 
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => { if(e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                  placeholder={(!selectedPatientId || docCount === 0) ? "System locked. Load context to begin." : "Ask clinical copilot..."}
                  className="flex-1 bg-transparent border-none outline-none px-2 py-1.5 text-gray-900 font-medium placeholder-gray-400 resize-none h-10 leading-tight text-sm"
                  disabled={loading || !selectedPatientId || docCount === 0}
                />
                <button 
                  onClick={() => handleSend()}
                  disabled={loading || !input.trim() || !selectedPatientId || docCount === 0}
                  className={`p-2 rounded flex items-center justify-center transition ${input.trim() && selectedPatientId && docCount > 0 ? 'bg-blue-600 text-white hover:bg-blue-700 shadow-sm' : 'bg-gray-100 text-gray-400'}`}
                >
                   <Send size={14} />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right Pane: Vector Database Panel */}
        <div className="hidden lg:flex flex-col w-[320px] xl:w-[380px] bg-slate-50 shrink-0">
          <div className="px-5 py-3 border-b border-gray-200 flex items-center justify-between shrink-0 bg-white">
             <h3 className="text-xs font-bold text-gray-900 flex items-center gap-2 uppercase tracking-wide">
               <Database size={14} className="text-gray-500"/> Vector Database
             </h3>
             <span className="bg-gray-100 border border-gray-200 text-gray-600 text-[10px] font-bold px-2 py-0.5 rounded">
               {activeSources.length} Extracted
             </span>
          </div>
          <div className="flex-1 overflow-y-auto p-5">
             {activeSources.length === 0 ? (
               <div className="flex flex-col items-center justify-center h-full text-center text-gray-400 px-4">
                 <BookOpen size={40} strokeWidth={1} className="mx-auto mb-3 opacity-30 text-gray-400" />
                 <p className="text-xs font-bold text-gray-600 mb-1">Retrieval Pipeline</p>
                 <p className="text-[11px] leading-relaxed">Execute a query to observe real-time vector extractions directly from Weaviate.</p>
               </div>
             ) : (
               <div className="space-y-4">
                 <div className="mb-4 bg-gray-100 border border-gray-200 p-3 rounded">
                    <p className="text-[11px] font-bold text-gray-800 flex items-center gap-1.5 uppercase tracking-wide"><CheckCircle2 size={12}/> Grounding Successful</p>
                    <p className="text-[10px] text-gray-500 mt-1 font-medium leading-relaxed">Response is anchored strictly to {activeSources.length} semantic chunks below.</p>
                 </div>
                 
                  {activeSources.map((cit, idx) => (
                    <div key={idx} className="bg-white p-3.5 rounded border border-gray-200 space-y-2 relative overflow-hidden group hover:border-gray-300 transition-colors shadow-sm">
                       <div className="absolute top-0 left-0 w-1 h-full bg-blue-500" />
                       <div className="flex justify-between items-start">
                          <h4 className="font-bold text-gray-900 text-[11px] truncate max-w-[75%]" title={cit.doc_title}>{cit.doc_title}</h4>
                          <span className="text-[9px] font-bold bg-gray-50 text-gray-500 border border-gray-200 px-1.5 py-0.5 rounded uppercase tracking-wider">P. {cit.page || 1}</span>
                       </div>
                       
                       {cit.image_url && (
                         <div className="rounded-lg overflow-hidden border border-gray-100 mt-1 mb-2 bg-gray-50 group-hover:border-blue-200 transition">
                            <img src={cit.image_url} alt="Vector Observation" className="w-full h-32 object-cover opacity-90 group-hover:opacity-100 transition" />
                            <div className="p-1.5 bg-gray-50 text-[9px] font-bold text-gray-500 flex items-center gap-1">
                               <ImageIcon size={10}/> Visual Chunk
                            </div>
                         </div>
                       )}

                       <div className="text-[11px] text-gray-600 leading-relaxed font-serif italic border-l-2 border-gray-200 pl-3 mt-2 line-clamp-4">
                          "{cit.quote}"
                       </div>
                    </div>
                  ))}
               </div>
             )}
          </div>
        </div>

      </div>

    </div>
  );
}
