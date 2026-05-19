#!/bin/sh
set -e

# Resolve script directory to an absolute POSIX path (works in Git Bash / WSL)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CERT_DIR="$SCRIPT_DIR/certs"
mkdir -p "$CERT_DIR"

SUBJ="/CN=flower.local"

echo "Generating self-signed certificates in: $CERT_DIR"

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout "$CERT_DIR/privkey.pem" \
  -out "$CERT_DIR/fullchain.pem" \
  -subj "$SUBJ"

echo "Generated self-signed certs in $CERT_DIR"
