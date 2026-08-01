FROM rust:1.72-slim AS builder
WORKDIR /usr/src/engine_rs
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /usr/src/engine_rs/target/release/engine_rs /usr/local/bin/engine_rs
EXPOSE 8080
CMD ["/usr/local/bin/engine_rs"]
