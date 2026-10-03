import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const LegalAgentUI = () => {
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [messages, setMessages] = useState({}); 
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const scrollRef = useRef(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, activeSession]);

  // Load history when switching sessions
  useEffect(() => {
    if (activeSession) {
      loadHistory(activeSession);
    }
  }, [activeSession]);

  const loadHistory = async (sessionId) => {
    try {
      const res = await axios.get(`${API_BASE}/history/${sessionId}`);
      setMessages(prev => ({ ...prev, [sessionId]: res.data.full_history }));
    } catch (e) {
      console.error("Error loading history", e);
    }
  };

  const startNewSession = async () => {
    const newId = `sess_${Date.now()}`;
    try {
      await axios.post(`${API_BASE}/create-session`, { session_id: newId });
      setSessions(prev => [...prev, newId]);
      setActiveSession(newId);
      setMessages(prev => ({ ...prev, [newId]: [] }));
    } catch (e) {
      alert("Failed to create session");
    }
  };

  const deleteSession = async (sessionId, e) => {
    e.stopPropagation(); // Prevent switching to the session being deleted
    if (!window.confirm("Delete this conversation?")) return;

    try {
      await axios.delete(`${API_BASE}/session/${sessionId}`);
      const updatedSessions = sessions.filter(id => id !== sessionId);
      setSessions(updatedSessions);
      if (activeSession === sessionId) {
        setActiveSession(updatedSessions[0] || null);
      }
      // Cleanup local state
      const newMessages = { ...messages };
      delete newMessages[sessionId];
      setMessages(newMessages);
    } catch (e) {
      alert("Error deleting session");
    }
  };

  const syncUserProfile = async () => {
    if (!activeSession) return;
    try {
      await axios.post(`${API_BASE}/sync-profile`, { session_id: activeSession });
      alert("Profile synced successfully based on this conversation!");
    } catch (e) {
      alert("Failed to sync profile.");
    }
  };

  const handleSend = async () => {
    if (!input.trim() || !activeSession) return;

    const userMsg = { role: 'user', content: input };
    const currentMsgs = messages[activeSession] || [];
    
    setMessages(prev => ({ 
      ...prev, 
      [activeSession]: [...currentMsgs, userMsg] 
    }));
    setInput('');
    setIsThinking(true);

    try {
      const response = await axios.post(`${API_BASE}/chat`, {
        session_id: activeSession,
        message: input
      });
      
      const botMsg = { role: 'bot', content: response.data.answer };
      setMessages(prev => ({ 
        ...prev, 
        [activeSession]: [...(prev[activeSession] || []), botMsg] 
      }));
    } catch (e) {
      console.error("Chat error", e);
      alert("The agent encountered an error. Please try again.");
    } finally {
      setIsThinking(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-100 font-sans text-gray-900">
      {/* SIDEBAR */}
      <div className="w-72 bg-slate-900 text-white flex flex-col shadow-xl">
        <div className="p-6">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-indigo-300 bg-clip-text text-transparent">
            LegalRAG Agent
          </h1>
          <p className="text-xs text-slate-400 mt-1">Entrepreneurship Legal Expert</p>
        </div>

        <div className="flex-1 overflow-y-auto px-4 space-y-2">
          {sessions.map(id => (
            <div 
              key={id}
              onClick={() => setActiveSession(id)}
              className={`group flex items-center justify-between p-3 rounded-lg cursor-pointer transition-all ${
                activeSession === id ? 'bg-blue-600 shadow-md' : 'hover:bg-slate-800'
              }`}
            >
              <span className="truncate text-sm font-medium">
                {id === 'default' ? 'General Chat' : id.replace('sess_', 'Chat ')}
              </span>
              <button 
                onClick={(e) => deleteSession(id, e)}
                className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition-opacity"
                title="Delete Session"
              >
                🗑️
              </button>
            </div>
          ))}
        </div>

        <div className="p-4 space-y-3 border-t border-slate-800">
          <button 
            onClick={startNewSession}
            className="w-full py-2 bg-blue-500 hover:bg-blue-400 rounded-md text-sm font-semibold transition-colors"
          >
            + New Conversation
          </button>
          <button 
            onClick={syncUserProfile}
            disabled={!activeSession}
            className={`w-full py-2 rounded-md text-sm font-semibold transition-colors ${
              activeSession ? 'bg-indigo-600 hover:bg-indigo-500 text-white' : 'bg-slate-700 text-slate-500 cursor-not-allowed'
            }`}
          >
            🔄 Sync User Profile
          </button>
        </div>
      </div>

      {/* MAIN CHAT AREA */}
      <div className="flex-1 flex flex-col bg-white">
        {!activeSession ? (
          <div className="flex-1 flex items-center justify-center text-gray-400 flex-col p-10 text-center">
            <div className="text-6xl mb-4">⚖️</div>
            <h2 className="text-xl font-semibold text-gray-600">No Conversation Selected</h2>
            <p>Start a new chat to begin your legal consultation.</p>
          </div>
        ) : (
          <>
            {/* Header */}
            <div className="h-16 border-b flex items-center px-6 justify-between bg-white z-10">
              <div className="font-medium text-gray-700">
                Session: <span className="text-blue-600">{activeSession}</span>
              </div>
              <div className="text-xs text-gray-400">
                Domain-Specific Legal Agent
              </div>
            </div>

            {/* Messages */}
            <div 
              ref={scrollRef}
              className="flex-1 overflow-y-auto p-6 space-y-6 bg-gray-50"
            >
              {(messages[activeSession] || []).map((m, i) => (
                <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] p-4 rounded-2xl shadow-sm ${
                    m.role === 'user' 
                      ? 'bg-blue-600 text-white rounded-tr-none' 
                      : 'bg-white text-gray-800 border border-gray-200 rounded-tl-none'
                  }`}>
                    <div className="text-xs opacity-50 mb-1 font-bold uppercase">
                      {m.role === 'user' ? 'You' : 'Legal Agent'}
                    </div>
                    <div className="whitespace-pre-wrap leading-relaxed text-sm">
                      {m.content}
                    </div>
                  </div>
                </div>
              ))}
              
              {isThinking && (
                <div className="flex justify-start">
                  <div className="bg-white border border-gray-200 p-4 rounded-2xl rounded-tl-none shadow-sm flex items-center gap-2 text-gray-500 text-sm italic">
                    <div className="flex gap-1">
                      <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                      <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                      <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                    </div>
                    Thinking...
                  </div>
                </div>
              )}
            </div>

            {/* Input Area */}
            <div className="p-4 bg-white border-t">
              <div className="max-w-4xl mx-auto flex gap-3">
                <input 
                  className="flex-1 border border-gray-300 p-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                  placeholder="Ask about your business legalities..."
                  disabled={isThinking}
                />
                <button 
                  onClick={handleSend} 
                  disabled={isThinking || !input.trim()}
                  className={`px-6 py-2 rounded-xl font-semibold transition-all ${
                    isThinking || !input.trim() 
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
                      : 'bg-blue-600 text-white hover:bg-blue-700 shadow-md'
                  }`}
                >
                  {isThinking ? '...' : 'Send'}
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default LegalAgentUI;
