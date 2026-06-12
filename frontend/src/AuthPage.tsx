import React, { useState } from "react";
import { registerUser, loginUser } from "./api";
import { User, Lock, Sparkles, Loader2, ArrowRight, X } from "lucide-react";

export function AuthPage({
  onLogin,
  onClose,
}: {
  onLogin: (token: string, username: string) => void;
  onClose?: () => void;
}) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      setError("Username and password are required");
      return;
    }

    setLoading(true);
    setError("");

    try {
      if (!isLogin) {
        await registerUser(username, password);
        setIsLogin(true);
        setPassword("");
        setError("Registration successful! Please sign in.");
        setLoading(false);
        return;
      }

      const res = await loginUser(username, password);
      if (res.data.access_token) {
        window.localStorage.setItem("nova_jwt_token", res.data.access_token);
        window.localStorage.setItem("nova_username", username);
        onLogin(res.data.access_token, username);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-box">
        {onClose && (
          <button type="button" className="auth-close-btn" onClick={onClose} aria-label="Close">
            <X size={20} />
          </button>
        )}
        <div className="auth-header">
          <Sparkles size={32} className="auth-logo" />
          <h1>{isLogin ? "Welcome Back" : "Join Nova"}</h1>
          <p>{isLogin ? "Sign in to continue your personalized session" : "Create an account for this browser session"}</p>
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
              placeholder={isLogin ? "Password" : "Create password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={loading}
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
