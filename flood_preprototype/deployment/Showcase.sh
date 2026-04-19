#!/bin/bash
# Always start in the folder where this script is located (deployment/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the virtual environment (two levels up from deployment/)
source "$SCRIPT_DIR/../../floodenv/bin/activate"

# Verify credentials loaded before launching
python - <<'EOF'
from dotenv import load_dotenv
import os, pathlib

env_path = pathlib.Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

checks = {
    "SUPABASE_URL":        os.getenv("SUPABASE_URL"),
    "SUPABASE_SERVICE_KEY": os.getenv("SUPABASE_SERVICE_KEY"),
    "FB_PAGE_ID":          os.getenv("FB_PAGE_ID"),
    "FB_PAGE_TOKEN":       os.getenv("FB_PAGE_TOKEN"),
    "TELEGRAM_BOT_TOKEN":  os.getenv("TELEGRAM_BOT_TOKEN"),
    "TELEGRAM_CHAT_ID":    os.getenv("TELEGRAM_CHAT_ID"),
}
for k, v in checks.items():
    status = "SET" if v else "MISSING ← FIX THIS"
    print(f"  {k:<25} {status}")
EOF

# Move into the scripts folder where Rapid_Relay_Showcase.py lives
cd "$SCRIPT_DIR/../showcase"

# Run the realtime pipeline
python Rapid_Relay_Showcase.py \
    --realtime \
    --predict-on-insert \
    --force-fb

# Keep terminal open on crash so you can read the error
echo ""
echo "Pipeline exited. Press Enter to close."
read -r