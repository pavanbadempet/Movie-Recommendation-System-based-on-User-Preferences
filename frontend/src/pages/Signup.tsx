/**
 * Self-serve signup flow.
 *
 * Step 1 — Email + password form → POST /v1/auth/register
 * Step 2 — Show generated API key once with copy button
 * Step 3 — Redirect to /getting-started
 *
 * Route: /signup
 */

import React from "react";
import { CheckCircle2, ClipboardCopy, Eye, EyeOff, Loader2 } from "lucide-react";
import { apiPost, API_BASES } from "../api";

interface SignupProps {
  onNavigate: (page: string) => void;
  onLoginSuccess?: (token: string, username: string) => void;
}

type SignupStep = "form" | "key" | "done";

interface RegisterResponse {
  username?: string;
  api_key?: string;
  access_token?: string;
  detail?: string;
}

function PasswordStrengthBar({ password }: { password: string }) {
  const score = React.useMemo(() => {
    let s = 0;
    if (password.length >= 8) s++;
    if (password.length >= 12) s++;
    if (/[A-Z]/.test(password)) s++;
    if (/[0-9]/.test(password)) s++;
    if (/[^A-Za-z0-9]/.test(password)) s++;
    return s;
  }, [password]);

  const label = ["", "Weak", "Fair", "Good", "Strong", "Very strong"][score] ?? "";
  const colorClass = ["", "strength-weak", "strength-fair", "strength-good", "strength-strong", "strength-very-strong"][score] ?? "";

  if (!password) return null;

  return (
    <div className="password-strength" aria-live="polite" aria-label={`Password strength: ${label}`}>
      <div className="strength-bars" aria-hidden="true">
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className={`strength-bar ${i <= score ? colorClass : ""}`} />
        ))}
      </div>
      <span className={`strength-label ${colorClass}`}>{label}</span>
    </div>
  );
}

export function SignupPage({ onNavigate, onLoginSuccess }: SignupProps) {
  const [step, setStep] = React.useState<SignupStep>("form");
  const [email, setEmail] = React.useState("");
  const [password, setPassword] = React.useState("");
  const [showPassword, setShowPassword] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [apiKey, setApiKey] = React.useState<string | null>(null);
  const [copied, setCopied] = React.useState(false);

  const emailRef = React.useRef<HTMLInputElement>(null);
  const apiKeyRef = React.useRef<HTMLElement>(null);

  // Focus email on mount
  React.useEffect(() => {
    emailRef.current?.focus();
  }, []);

  // Focus API key section when it appears
  React.useEffect(() => {
    if (step === "key") {
      apiKeyRef.current?.focus();
    }
  }, [step]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!email.trim() || !password) return;

    setLoading(true);
    setError(null);

    try {
      const { data } = await apiPost<RegisterResponse>("/v1/auth/register", {
        username: email.trim(),
        password,
      });

      if (data.detail) {
        setError(data.detail);
        return;
      }

      // Store JWT if returned
      if (data.access_token) {
        window.localStorage.setItem("nova_jwt_token", data.access_token);
        onLoginSuccess?.(data.access_token, email.trim());
      }

      // Show API key if the backend returns one
      if (data.api_key) {
        setApiKey(data.api_key);
        setStep("key");
      } else {
        // No API key returned — go straight to getting started
        setStep("done");
        setTimeout(() => onNavigate("getting-started"), 800);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Registration failed";
      setError(msg.includes("already") ? "An account with this email already exists. Try logging in." : msg);
    } finally {
      setLoading(false);
    }
  }

  async function copyApiKey() {
    if (!apiKey) return;
    try {
      await navigator.clipboard.writeText(apiKey);
      setCopied(true);
      setTimeout(() => setCopied(false), 3000);
    } catch {
      // Fallback for browsers without clipboard API
      const el = document.createElement("textarea");
      el.value = apiKey;
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
      setCopied(true);
      setTimeout(() => setCopied(false), 3000);
    }
  }

  const baseUrl = API_BASES[0] ?? "https://your-apex-api.onrender.com";

  // ── Step: API key display ────────────────────────────────────────────────
  if (step === "key" && apiKey) {
    return (
      <main className="signup-page" aria-labelledby="apikey-heading">
        <div className="signup-card">
          <div className="signup-success-icon" aria-hidden="true">
            <CheckCircle2 size={48} />
          </div>

          <h1 id="apikey-heading">Your API key is ready</h1>

          <div
            className="api-key-warning"
            role="alert"
            aria-live="assertive"
          >
            <strong>Store this key now.</strong> It will not be shown again.
            You can generate a new one from your dashboard if you lose it.
          </div>

          <section
            ref={apiKeyRef}
            tabIndex={-1}
            className="api-key-display"
            aria-label="Your API key"
          >
            <code className="api-key-value" aria-label={`API key: ${apiKey}`}>
              {apiKey}
            </code>
            <button
              type="button"
              className="copy-button"
              onClick={copyApiKey}
              aria-label={copied ? "API key copied to clipboard" : "Copy API key to clipboard"}
            >
              {copied ? (
                <>
                  <CheckCircle2 size={16} aria-hidden="true" />
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <ClipboardCopy size={16} aria-hidden="true" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </section>

          <div className="api-key-usage" aria-label="Quick start example">
            <p>Try it now:</p>
            <pre className="code-block small">
              <code>{`curl "${baseUrl}/v1/recommendations/id/550" \\
  -H "X-Nova-API-Key: ${apiKey.slice(0, 10)}..."`}</code>
            </pre>
          </div>

          <div className="signup-actions">
            <button
              type="button"
              className="primary-button"
              onClick={() => onNavigate("getting-started")}
              aria-label="Continue to Getting Started guide"
            >
              Continue to Getting Started →
            </button>
            <button
              type="button"
              className="text-link"
              onClick={() => onNavigate("dashboard")}
              aria-label="Go to your dashboard"
            >
              Go to dashboard
            </button>
          </div>
        </div>
      </main>
    );
  }

  // ── Step: Done (no API key returned) ────────────────────────────────────
  if (step === "done") {
    return (
      <main className="signup-page" aria-labelledby="done-heading">
        <div className="signup-card">
          <div className="signup-success-icon" aria-hidden="true">
            <CheckCircle2 size={48} />
          </div>
          <h1 id="done-heading">Account created</h1>
          <p>Redirecting you to Getting Started…</p>
          <Loader2 size={24} className="spin" aria-label="Loading" />
        </div>
      </main>
    );
  }

  // ── Step: Registration form ──────────────────────────────────────────────
  return (
    <main className="signup-page" aria-labelledby="signup-heading">
      <div className="signup-card">
        <h1 id="signup-heading">Create your APEX account</h1>
        <p className="signup-subheading">
          Free tier included. No credit card required.
        </p>

        {error && (
          <div className="form-error" role="alert" aria-live="polite">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} noValidate aria-label="Account registration form">
          {/* Email */}
          <div className="form-field">
            <label htmlFor="signup-email">Email address</label>
            <input
              ref={emailRef}
              id="signup-email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@yourcompany.com"
              autoComplete="email"
              required
              aria-required="true"
              aria-describedby="email-hint"
            />
            <span id="email-hint" className="field-hint">
              This will be your login and tenant identifier.
            </span>
          </div>

          {/* Password */}
          <div className="form-field">
            <label htmlFor="signup-password">Password</label>
            <div className="password-input-wrapper">
              <input
                id="signup-password"
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="At least 8 characters"
                autoComplete="new-password"
                required
                aria-required="true"
                minLength={8}
                aria-describedby="password-strength-hint"
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword((v) => !v)}
                aria-label={showPassword ? "Hide password" : "Show password"}
                tabIndex={0}
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            <span id="password-strength-hint" className="visually-hidden">
              Password strength indicator below
            </span>
            <PasswordStrengthBar password={password} />
          </div>

          <button
            type="submit"
            className="primary-button full-width"
            disabled={loading || !email.trim() || password.length < 8}
            aria-busy={loading}
            aria-label="Create account and get API key"
          >
            {loading ? (
              <>
                <Loader2 size={16} className="spin" aria-hidden="true" />
                Creating account…
              </>
            ) : (
              "Create account & get API key"
            )}
          </button>
        </form>

        <div className="signup-footer">
          <p>
            Already have an account?{" "}
            <button
              type="button"
              className="text-link"
              onClick={() => onNavigate("login")}
              aria-label="Go to login page"
            >
              Sign in
            </button>
          </p>
          <p className="fine-print">
            By signing up you agree to our{" "}
            <a href="/terms" target="_blank" rel="noreferrer">
              Terms of Service
            </a>{" "}
            and{" "}
            <a href="/privacy" target="_blank" rel="noreferrer">
              Privacy Policy
            </a>
            .
          </p>
        </div>
      </div>
    </main>
  );
}
