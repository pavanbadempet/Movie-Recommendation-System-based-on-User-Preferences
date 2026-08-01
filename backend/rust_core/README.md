Local build and benchmark instructions for `rust_core` native extension

Prerequisites
- Python 3.12 (recommended). On Windows, use the official 3.12 installer or pyenv-win.
- Rust toolchain (stable) installed via `rustup`
- `maturin` to build/install the extension

Quick local build (preferred)
1. Create and activate a Python 3.12 virtualenv:
   - Windows: `python -m venv .venv && .venv\Scripts\activate`
   - macOS / Linux: `python -m venv .venv && source .venv/bin/activate`
2. Upgrade pip and install maturin:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install maturin
   ```
3. Build & install the extension in editable/develop mode:
   ```bash
   python -m maturin develop --release
   ```
   This builds a wheel for the active Python interpreter and installs it into the venv.

Alternative: build a wheel artifact
- Build: `python -m maturin build --release -i python3.12`
- Install the wheel from `target/wheels/` using `pip install <wheel-file>`

If you cannot use Python 3.12
- PyO3 checks may block builds for newer Python versions. Either use Python 3.12 or set this environment variable to allow ABI-forward compatibility (may have subtle issues):
  - `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`

Run the micro-benchmark (after installing the extension)
1. Activate the same venv used to build/install the wheel.
2. From the crate folder run:
   ```bash
   python benchmarks/run_mmr_bench.py
   ```
   On success this writes `bench_result.json` with `{"iterations","total_time_s","avg_ms"}`.

Troubleshooting
- If `rust_core` is not importable after installation, check `pip show rust-core` (wheel name) and ensure you installed into the same interpreter that runs the backend.
- For CI builds, prefer using Python 3.12 in the workflow or the `PYO3_USE_ABI3_FORWARD_COMPATIBILITY` env var.

Contact
- If build errors persist, capture the full `maturin` output and open an issue in the repo with the `cargo` and `maturin` logs.
