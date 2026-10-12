#!/usr/bin/env bash
# Iteration build for the ollama-cliproxyapi plugin (archive strategy).
#
# Every iteration run produces:
#   1. plugins/linux/amd64/ollama-cliproxyapi.so   canonical deploy artifact
#      (stable plugin ID: the host always loads this name)
#   2. plugins/releases/ollama-cliproxyapi-<version>.so
#      frozen copy for rollback (copy it over #1 to deploy that version;
#      never leaves two .so files in the host's scan dir — two loaded plugins
#      would register duplicate providers and both predictor engines would
#      sample ollama.com, doubling pressure on the 10 req/min/user budget)
#
# The version comes from VERSION at the repo root ("v0.0.2" or "0.0.2").
# It is injected into internal/buildinfo so the registration metadata the
# host sees carries the real iteration version.
#
# Usage:
#   scripts/build-so.sh              # gates (build+vet+test) then produce both files
#   SKIP_TESTS=1 scripts/build-so.sh # skip the test gate (e.g. quick rebuild)
#   FORCE=1 scripts/build-so.sh      # overwrite an existing release archive
#
# Exits non-zero if the release file for this version already exists — bump
# VERSION for a new iteration instead of overwriting history.
set -euo pipefail

cd "$(dirname "$0")/.."

# --- version source of truth ------------------------------------------------
VERSION_FILE=VERSION
if [ ! -f "$VERSION_FILE" ]; then
    echo "error: $VERSION_FILE not found (create it, e.g. 'v0.0.2')" >&2
    exit 1
fi
RAW_VERSION="$(tr -d ' \t\r\n' <"$VERSION_FILE")"
VERSION="${RAW_VERSION#v}" # buildinfo wants the tag without its leading v
if [ -z "$VERSION" ]; then
    echo "error: $VERSION_FILE is empty" >&2
    exit 1
fi
RELEASE_NAME="ollama-cliproxyapi-${RAW_VERSION}.so"

# --- toolchain (host GOROOT env can be poisoned; pin like /tmp/opencode/gow)
if [ -d /tmp/opencode/go1.26.7 ]; then
    export GOROOT=/tmp/opencode/go1.26.7
    export GOPATH=/tmp/opencode/gopath
    export GOMODCACHE=/tmp/opencode/gopath/pkg/mod
    export GOCACHE=/tmp/opencode/gocache
    export GOTOOLCHAIN=local
    export PATH="/tmp/opencode/go1.26.7/bin:$PATH"
    GOFMT=/tmp/opencode/go1.26.7/bin/gofmt
elif command -v go >/dev/null 2>&1; then
    GOFMT="$(command -v gofmt || true)"
else
    echo "error: no Go toolchain (expected /tmp/opencode/go1.26.7 or go in PATH)" >&2
    exit 1
fi

# --- commit stamp -----------------------------------------------------------
COMMIT="$(git -C . describe --always --dirty 2>/dev/null || echo unknown)"

# --- gates ------------------------------------------------------------------
if [ "${SKIP_TESTS:-0}" != "1" ]; then
    go build ./...
    go vet ./...
    go test -count=1 ./...
fi
if [ -n "${GOFMT:-}" ] && [ -x "$GOFMT" ]; then
    UNFORMATTED="$("$GOFMT" -l internal/)"
    if [ -n "$UNFORMATTED" ]; then
        echo "error: gofmt drift (use the pinned toolchain's gofmt, not the host one —" \
            "the host gofmt predates generics):" >&2
        echo "$UNFORMATTED" >&2
        exit 1
    fi
fi

# --- build ------------------------------------------------------------------
mkdir -p plugins/linux/amd64 plugins/releases
DEPLOY=plugins/linux/amd64/ollama-cliproxyapi.so
RELEASE="plugins/releases/$RELEASE_NAME"
if [ -f "$RELEASE" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "error: $RELEASE already exists; bump $VERSION_FILE for the new iteration" \
        "(or FORCE=1 to overwrite)" >&2
    exit 1
fi

go build -trimpath -buildmode=c-shared \
    -ldflags "-X ollama-cliproxyapi/internal/buildinfo.Version=${VERSION} -X ollama-cliproxyapi/internal/buildinfo.Commit=${COMMIT}" \
    -o "$DEPLOY" .
cp "$DEPLOY" "$RELEASE"
if [ "${SKIP_TESTS:-0}" != "1" ]; then
    go version -m "$RELEASE" | grep -E "build\s+-X|vcs" || true
fi

echo
echo "deploy : $DEPLOY (stable plugin ID)"
echo "release: $RELEASE (iteration archive, version ${VERSION} commit ${COMMIT})"