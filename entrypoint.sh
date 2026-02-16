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

exec streamlit run app.py
