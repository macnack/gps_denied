#!/bin/bash

# === Usage check ===
if [ -z "$1" ]; then
  echo "❌ Usage: $0 /full/path/to/run.sh"
  exit 1
fi

TARGET_SCRIPT="$1"

if [ ! -f "$TARGET_SCRIPT" ]; then
  echo "❌ File does not exist: $TARGET_SCRIPT"
  exit 1
fi

# === Ensure ~/bin exists ===
mkdir -p ~/bin

# === Remove existing symlink if needed ===
rm -f ~/bin/gpstrain

# === Create new symlink ===
ln -s "$TARGET_SCRIPT" ~/bin/gpstrain

# === Update ~/.bashrc ===
BASHRC="$HOME/.bashrc"

# Add ~/bin to PATH if not already
if ! grep -q 'export PATH="$HOME/bin:$PATH"' "$BASHRC"; then
  echo 'export PATH="$HOME/bin:$PATH"' >> "$BASHRC"
fi

# Add gpstrain() function if not present
if ! grep -q 'gpstrain() {' "$BASHRC"; then
cat <<'EOF' >> "$BASHRC"

# gpstrain wrapper function
gpstrain() {
  if [ "$1" = "log" ]; then
    docker logs -f gps_denied
  else
    ~/bin/gpstrain "$@"
  fi
}
EOF
fi

# Apply changes now
source "$BASHRC"

echo "✅ gpstrain setup complete!"
echo "You can now run: gpstrain or gpstrain log"

