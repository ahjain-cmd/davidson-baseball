#!/bin/bash
# Generate feather cache if missing (makes first Streamlit load near-instant)
FEATHER=/app/.cache/davidson_data.feather
DUCKDB=/app/davidson.duckdb

mkdir -p /app/.cache

if [ ! -f "$FEATHER" ] && [ -f "$DUCKDB" ]; then
    echo "[warmup] Exporting davidson_data.feather from DuckDB..."
    python3 -c "
import duckdb
con = duckdb.connect('$DUCKDB', read_only=True)
df = con.execute('SELECT * FROM davidson_data').fetchdf()
df.to_feather('$FEATHER')
print(f'  Exported {len(df):,} rows to feather')
con.close()
"
    echo "[warmup] Feather cache ready."
else
    echo "[warmup] Feather cache already exists, skipping export."
fi

# Start Streamlit in the background, then warm the cache with a real request
streamlit run app.py &
STREAMLIT_PID=$!

# Wait for Streamlit to be ready, then hit it to populate st.cache_data
(
    echo "[warmup] Waiting for Streamlit to start..."
    for i in $(seq 1 60); do
        if curl -sf http://localhost:8501/_stcore/health > /dev/null 2>&1; then
            echo "[warmup] Streamlit healthy after ${i}s, warming cache..."
            # Hit the main page to populate load_davidson_data cache
            curl -sf http://localhost:8501/ > /dev/null 2>&1
            echo "[warmup] Cache warmed — app is ready for users."
            break
        fi
        sleep 1
    done
) &

# Hand control back to Streamlit process
wait $STREAMLIT_PID
