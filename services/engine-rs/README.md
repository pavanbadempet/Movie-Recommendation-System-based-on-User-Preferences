# engine-rs PoC

PoC compute engine for genetic breeding math and traits using Rust.

Features:
- Axum HTTP server exposing `POST /breed` that accepts JSON {"population": [[...]], "generations": n}
- MiMalloc as the global allocator for lower-latency allocations
- Core library exposes `breed()` and optional PyO3 bindings when built with `--features python`

Build locally:
```bash
cd services/engine-rs
cargo build --release
./target/release/engine_rs
```

Build docker:
```bash
docker build -t engine-rs:local .
docker run -p 8080:8080 engine-rs:local
```

Python bindings (optional):
```bash
# Requires maturin or pyo3 build setup
maturin build -f --release
```
