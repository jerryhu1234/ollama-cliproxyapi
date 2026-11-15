#!/usr/bin/env bash
# Build the plugin as a C shared library for CLIProxyAPI.
#
#   ./build.sh              -> plugins/linux/amd64/ollama-cliproxyapi.so
#   CC=clang ./build.sh     -> use another compiler
#
# Requires Go 1.26.10+ (go.mod) and a working CGO toolchain (gcc/clang).
# The plugin is loaded by the CPA host process via dlopen, so it must be
# built with -buildmode=c-shared; a plain `go build` binary is not loadable.
set -euo pipefail

cd "$(dirname "$0")"

OUT_DIR="plugins/linux/amd64"
OUT="${OUT_DIR}/ollama-cliproxyapi.so"

# The .so name determines the plugin ID. Never rename it.
mkdir -p "${OUT_DIR}"

CGO_ENABLED=1 go build \
	-trimpath \
	-buildmode=c-shared \
	-o "${OUT}" \
	.

# -buildmode=c-shared always emits a companion header. It is build output,
# not source, and .gitignore excludes it.
echo "built ${OUT}"
ls -l "${OUT}"
