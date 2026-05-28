import React, { useState } from "react";
import { registerUser, loginUser } from "./api";
import { User, Lock, Sparkles, Loader2, ArrowRight } from "lucide-react";

export function AuthPage({ onLogin }: { onLogin: (token: string, username: string) => void }) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim()) {
      setError("Username is required");
      return;
    }
    
    setLoading(true);
    setError("");
    
    try {
      if (!isLogin) {
        await registerUser(username);
        setIsLogin(true);
        setError("Registration successful! Please login.");
        setLoading(false);
        return;
      }
      
      const res = await loginUser(username);
      if (res.data.access_token) {
        window.localStorage.setItem("nova_jwt_token", res.data.access_token);
        window.localStorage.setItem("nova_username", username);
        onLogin(res.data.access_token, username);
      }
    } catch (err: any) {
      setError(err.message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-box">
        <div className="auth-header">
          <Sparkles size={32} className="auth-logo" />
          <h1>{isLogin ? "Welcome Back" : "Join Nova"}</h1>
          <p>{isLogin ? "Enter a demo username to personalize this browser session" : "Create a demo profile for this browser"}</p>
        </div>
        
        {error && <div className={`auth-error ${error.includes('successful') ? 'success' : ''}`}>{error}</div>}
        
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="input-group">
            <User size={18} />
            <input 
              type="text" 
              placeholder="Username" 
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={loading}
            />
          </div>
          <div className="input-group">
            <Lock size={18} />
            <input 
              type="password" 
              placeholder="Demo password" 
              value="password123"
              disabled
            />
          </div>
          
          <button type="submit" className="auth-submit" disabled={loading}>
            {loading ? <Loader2 size={18} className="spin" /> : <>{isLogin ? "Sign In" : "Create Account"} <ArrowRight size={18}/></>}
          </button>
        </form>
        
        <div className="auth-footer">
          <button type="button" onClick={() => { setIsLogin(!isLogin); setError(""); }}>
            {isLogin ? "Don't have an account? Sign up" : "Already have an account? Sign in"}
          </button>
        </div>
      </div>
    </div>
  );
}
