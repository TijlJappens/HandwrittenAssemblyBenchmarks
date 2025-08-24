# Rust development Dockerfile
FROM rust:latest AS base

RUN rustup default nightly

FROM base AS development