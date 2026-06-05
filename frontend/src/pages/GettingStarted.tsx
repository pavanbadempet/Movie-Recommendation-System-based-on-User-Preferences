/**
 * Getting Started — four-step interactive onboarding wizard.
 *
 * Step 1 — Your API key (copy)
 * Step 2 — Upload your catalog (drag-and-drop CSV)
 * Step 3 — Make your first API call (live try-it)
 * Step 4 — Explore the dashboard
 *
 * Route: /getting-started
 */

import React from "react";
import {
  CheckCircle2,
  ClipboardCopy,
  Loader2,
  Play,
  Upload,
  BarChart3,
} from "lucide-react";
import { apiGet, API_BASES } from "../api";

interface GettingStartedProps {
  onNavigate: (page: string) => void;
}

type StepId = 1 | 2 | 3 | 4;
type CodeLang = "curl" | "python" | "javascript";

const BASE_URL = API_BASES[0] ?? "https://your-apex-api.onrender.com";

// ---------------------------------------------------------------------------
// Code snippets
// ---------------------------------------------------------------------------
function getSnippet(lang: CodeLang, apiKey: string): string {
  const key = apiKey || "YOUR_KEY";
  const maskedKey = key.length > 12 ? `${key.slice(0, 10)}...` : key;

  if (lang === "curl") {
    return `curl "${BASE_URL}/v1/recommendations/id/550?n=5&explain=true" \\
  -H "X-Nova-API-Key: ${maskedKey}"`;
  }
  if (lang === "python") {
    return `import httpx

BASE = "${BASE_URL}"
HEADERS = {"X-Nova-API-Key": "${maskedKey}"}

resp = httpx.get(
    f"{BASE}/v1/recommendations/id/550",
    headers=HEADERS,
    params={"n": 5, "explain": "true"},
)
recs = resp.json()["recommendations"]
for r in recs:
    print(r["title"], "-", r.get("explanation_text", ""))`;
  }
  return `const BASE = "${BASE_URL}";
const HEADERS = { "X-Nova-API-Key": "${maskedKey}" };

const resp = await fetch(
  \`\${BASE}/v1/recommendations/id/550?n=5&explain=true\`,
  { headers: HEADERS }
);
const { recommendations } = await resp.json();
recommendations.forEach(r =>
  console.log(r.title, "-", r.explanation_text ?? "")
);`;
}

// ---------------------------------------------------------------------------
// Step indicator
// ---------------------------------------------------------------------------
function StepIndicator({
  current,


  onStep,
}: {
  current: StepId;


  onStep: (step: StepId) => void;
}) {
  const labels = ["API key", "Upload catalog", "First call", "Dashboard"];
  return (
    <nav className="step-indicator" aria-label="Onboarding steps">
      <ol>
        {labels.map((label, i) => {
          const stepNum = (i + 1) as StepId;
          const state =
            stepNum < current ? "completed" : stepNum === current ? "active" : "upcoming";
          return (
            <li key={label} className={`step-dot ${state}`}>
              <button
                type="button"
                onClick={() => onStep(stepNum)}
                aria-label={`Step ${stepNum}: ${label}${state === "completed" ? " (completed)" : state === "active" ? " (current)" : ""}`}
                aria-current={state === "active" ? "step" : undefined}
                disabled={stepNum > current}
              >
                {state === "completed" ? (
                  <CheckCircle2 size={16} aria-hidden="true" />
                ) : (
                  <span aria-hidden="true">{stepNum}</span>
                )}
              </button>
              <span className="step-dot-label">{label}</span>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export function GettingStartedPage({ onNavigate }: GettingStartedProps) {
  const [step, setStep] = React.useState<StepId>(1);
  const [apiKey] = React.useState<string>(
    () => window.localStorage.getItem("nova_api_key") ?? "",
  );
  const [copied, setCopied] = React.useState(false);
  const [lang, setLang] = React.useState<CodeLang>("curl");

  // Step 2 — catalog upload
  const [uploadFile, setUploadFile] = React.useState<File | null>(null);
  const [uploadLoading, setUploadLoading] = React.useState(false);
  const [uploadResult, setUploadResult] = React.useState<{
    rows: number;
    catalog_id: string;
  } | null>(null);
  const [uploadError, setUploadError] = React.useState<string | null>(null);
  const dropRef = React.useRef<HTMLLabelElement>(null);

  // Step 3 — live call
  const [tryLoading, setTryLoading] = React.useState(false);
  const [tryResult, setTryResult] = React.useState<Record<string, unknown> | null>(null);
  const [tryError, setTryError] = React.useState<string | null>(null);

  async function copyKey() {
    const key = apiKey || "YOUR_KEY_HERE";
    try {
      await navigator.clipboard.writeText(key);
    } catch {
      const el = document.createElement("textarea");
      el.value = key;
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  }

  async function handleUpload() {
    if (!uploadFile) return;
    setUploadLoading(true);
    setUploadError(null);
    try {
      const form = new FormData();
      form.append("file", uploadFile);
      const resp = await fetch(`${BASE_URL}/v1/catalog/upload`, {
        method: "POST",
        headers: apiKey ? { "X-Nova-API-Key": apiKey } : {},
        body: form,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error((err as { detail?: string }).detail ?? `HTTP ${resp.status}`);
      }
      const data = (await resp.json()) as {
        rows_ingested?: number;
        catalog_id?: string;
      };
      setUploadResult({
        rows: data.rows_ingested ?? 0,
        catalog_id: data.catalog_id ?? "default",
      });
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploadLoading(false);
    }
  }

  async function handleTryCall() {
    setTryLoading(true);
    setTryError(null);
    setTryResult(null);
    try {
      const { data } = await apiGet<Record<string, unknown>>(
        "/v1/recommendations/id/550",
        { n: 3, explain: true },
        15000,
      );
      setTryResult(data);
    } catch (err) {
      setTryError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setTryLoading(false);
    }
  }

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <main className="getting-started-page" aria-labelledby="gs-heading">
      <header className="gs-header">
        <h1 id="gs-heading">Getting Started with APEX</h1>
        <p>Follow these four steps to make your first recommendation call.</p>
      </header>

      <StepIndicator current={step} onStep={setStep} />

      {/* ── Step 1: API Key ─────────────────────────────────────────────── */}
      {step === 1 && (
        <section className="gs-step" aria-labelledby="step1-heading">
          <h2 id="step1-heading">
            <span className="step-number-chip" aria-hidden="true">1</span>
            Your API key
          </h2>
          <p>
            Include this key in every request as the{" "}
            <code>X-Nova-API-Key</code> header.
          </p>

          {apiKey ? (
            <div className="api-key-display">
              <code className="api-key-value" aria-label="Your API key">
                {apiKey}
              </code>
              <button
                type="button"
                className="copy-button"
                onClick={copyKey}
                aria-label={copied ? "Copied" : "Copy API key to clipboard"}
              >
                {copied ? (
                  <><CheckCircle2 size={16} aria-hidden="true" /> Copied</>
                ) : (
                  <><ClipboardCopy size={16} aria-hidden="true" /> Copy</>
                )}
              </button>
            </div>
          ) : (
            <div className="gs-no-key">
              <p>
                No API key found. Sign up to get one, or check your dashboard.
              </p>
              <button
                type="button"
                className="primary-button"
                onClick={() => onNavigate("signup")}
                aria-label="Sign up to get an API key"
              >
                Get an API key
              </button>
            </div>
          )}

          <div className="gs-step-actions">
            <button
              type="button"
              className="primary-button"
              onClick={() => setStep(2)}
              aria-label="Continue to step 2: Upload catalog"
            >
              Next: Upload your catalog →
            </button>
            <button
              type="button"
              className="text-link"
              onClick={() => setStep(3)}
              aria-label="Skip to step 3: Make your first call"
            >
              Skip — I&apos;ll use the demo catalog
            </button>
          </div>
        </section>
      )}

      {/* ── Step 2: Upload catalog ──────────────────────────────────────── */}
      {step === 2 && (
        <section className="gs-step" aria-labelledby="step2-heading">
          <h2 id="step2-heading">
            <span className="step-number-chip" aria-hidden="true">2</span>
            Upload your catalog
          </h2>
          <p>
            A CSV file with <code>item_id</code>, <code>title</code>, and{" "}
            <code>description</code> columns. Optional:{" "}
            <code>genres</code>, <code>poster_url</code>.
          </p>

          {!uploadResult ? (
            <>
              <label
                ref={dropRef}
                htmlFor="catalog-file"
                className={`drop-zone ${uploadFile ? "has-file" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  dropRef.current?.classList.add("drag-over");
                }}
                onDragLeave={() => dropRef.current?.classList.remove("drag-over")}
                onDrop={(e) => {
                  e.preventDefault();
                  dropRef.current?.classList.remove("drag-over");
                  const file = e.dataTransfer.files[0];
                  if (file?.name.endsWith(".csv")) setUploadFile(file);
                }}
                aria-label="Drop a CSV file here or click to browse"
              >
                <Upload size={32} aria-hidden="true" />
                <span>
                  {uploadFile
                    ? `${uploadFile.name} (${(uploadFile.size / 1024).toFixed(1)} KB)`
                    : "Drop catalog.csv here or click to browse"}
                </span>
                <input
                  id="catalog-file"
                  type="file"
                  accept=".csv"
                  className="visually-hidden"
                  onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
                  aria-label="Choose CSV catalog file"
                />
              </label>

              {uploadError && (
                <div className="form-error" role="alert" aria-live="polite">
                  {uploadError}
                </div>
              )}

              <div className="gs-step-actions">
                <button
                  type="button"
                  className="primary-button"
                  onClick={handleUpload}
                  disabled={!uploadFile || uploadLoading}
                  aria-busy={uploadLoading}
                  aria-label="Upload catalog file"
                >
                  {uploadLoading ? (
                    <><Loader2 size={16} className="spin" aria-hidden="true" /> Uploading…</>
                  ) : (
                    "Upload catalog"
                  )}
                </button>
                <button
                  type="button"
                  className="text-link"
                  onClick={() => setStep(3)}
                  aria-label="Skip catalog upload and use demo catalog"
                >
                  Skip — use demo catalog
                </button>
              </div>
            </>
          ) : (
            <div className="upload-success" role="status" aria-live="polite">
              <CheckCircle2 size={32} aria-hidden="true" />
              <div>
                <strong>Catalog uploaded successfully</strong>
                <p>
                  {uploadResult.rows.toLocaleString()} items ingested into
                  catalog <code>{uploadResult.catalog_id}</code>.
                </p>
              </div>
              <button
                type="button"
                className="primary-button"
                onClick={() => setStep(3)}
                aria-label="Continue to step 3: Make your first call"
              >
                Next: Make your first call →
              </button>
            </div>
          )}
        </section>
      )}

      {/* ── Step 3: First call ──────────────────────────────────────────── */}
      {step === 3 && (
        <section className="gs-step" aria-labelledby="step3-heading">
          <h2 id="step3-heading">
            <span className="step-number-chip" aria-hidden="true">3</span>
            Make your first call
          </h2>
          <p>
            Run this command to get recommendations for movie ID 550 (Fight
            Club) from the demo catalog.
          </p>

          {/* Language toggle */}
          <div className="lang-tabs" role="tablist" aria-label="Code language">
            {(["curl", "python", "javascript"] as CodeLang[]).map((l) => (
              <button
                key={l}
                role="tab"
                type="button"
                className={`lang-tab ${lang === l ? "active" : ""}`}
                onClick={() => setLang(l)}
                aria-selected={lang === l}
                aria-label={`Show ${l} code example`}
              >
                {l}
              </button>
            ))}
          </div>

          <pre className="code-block" aria-label={`${lang} code example`}>
            <code>{getSnippet(lang, apiKey)}</code>
          </pre>

          <button
            type="button"
            className="try-button"
            onClick={handleTryCall}
            disabled={tryLoading}
            aria-busy={tryLoading}
            aria-label="Run recommendation request against the live API"
          >
            {tryLoading ? (
              <><Loader2 size={16} className="spin" aria-hidden="true" /> Running…</>
            ) : (
              <><Play size={16} aria-hidden="true" /> Try it now</>
            )}
          </button>

          {tryError && (
            <div className="form-error" role="alert" aria-live="polite">
              {tryError}
            </div>
          )}

          {tryResult && (
            <div className="try-result" role="region" aria-label="API response">
              <div className="try-result-header">
                <CheckCircle2 size={16} aria-hidden="true" />
                <span>Live response from APEX API</span>
              </div>
          <pre className="code-block response-block" aria-label="JSON response">
                <code>{
                  (() => {
                    const s = JSON.stringify(tryResult, null, 2);
                    return s.length > 1200 ? s.slice(0, 1200) + "\n  ... (truncated)" : s;
                  })()
                }</code>
              </pre>
            </div>
          )}

          <div className="gs-step-actions">
            <button
              type="button"
              className="primary-button"
              onClick={() => setStep(4)}
              aria-label="Continue to step 4: Explore the dashboard"
            >
              Next: Explore your dashboard →
            </button>
          </div>
        </section>
      )}

      {/* ── Step 4: Dashboard ───────────────────────────────────────────── */}
      {step === 4 && (
        <section className="gs-step" aria-labelledby="step4-heading">
          <h2 id="step4-heading">
            <span className="step-number-chip" aria-hidden="true">4</span>
            Explore your dashboard
          </h2>
          <p>
            Your dashboard shows the active serving tier, SLO compliance,
            hardware profile, and per-endpoint request counts in real time.
          </p>

          <div className="gs-dashboard-preview" aria-label="Dashboard preview">
            <div className="preview-card">
              <strong>Serving tier</strong>
              <span className="tier-badge tier2">Tier 2 — ONNX CPU</span>
            </div>
            <div className="preview-card">
              <strong>p95 latency</strong>
              <span>342 ms</span>
            </div>
            <div className="preview-card">
              <strong>Error rate</strong>
              <span>0.0%</span>
            </div>
            <div className="preview-card">
              <strong>Requests today</strong>
              <span>—</span>
            </div>
          </div>

          <div className="gs-complete">
            <CheckCircle2 size={48} aria-hidden="true" />
            <h3>You&apos;re all set</h3>
            <p>
              Your APEX integration is live. Log events to improve
              recommendations over time.
            </p>
          </div>

          <div className="gs-step-actions">
            <button
              type="button"
              className="primary-button"
              onClick={() => onNavigate("dashboard")}
              aria-label="Open your APEX dashboard"
            >
              <BarChart3 size={16} aria-hidden="true" />
              Open dashboard
            </button>
            <button
              type="button"
              className="text-link"
              onClick={() => onNavigate("home")}
              aria-label="Explore the full API"
            >
              Explore the full API
            </button>
          </div>
        </section>
      )}
    </main>
  );
}
