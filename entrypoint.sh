#!/bin/bash
# Start Streamlit in the background, then warm the cache with a real page request
# so the first user doesn't see a loading skeleton.

streamlit run app.py &
STPID=$!

# Wait for Streamlit to be ready
echo "[warmup] Waiting for Streamlit to start..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        echo "[warmup] Streamlit is up after ${i}s, warming data cache..."
        # Hit the main page to trigger load_davidson_data() + sidebar stats
        curl -sf http://localhost:8501/ > /dev/null 2>&1
        echo "[warmup] Cache warmed — ready for users."
        break
    fi
    sleep 1
done

# Keep the container alive by waiting on the Streamlit process
wait $STPID
