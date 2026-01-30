"""
Preprocess all Trackman CSVs into a single parquet file for fast loading.
Run this once (or whenever new data arrives):
    python3 preprocess.py
"""
import pandas as pd
import glob
import os
import sys

DATA_ROOT = "/Users/ahanjain/v3"
OUTPUT = "/Users/ahanjain/davidson_baseball/all_trackman.parquet"
DAVIDSON_TEAM_ID = "DAV_WIL"

NAME_MAP = {
    "Laughlin, Theo": "Loughlin, Theo",
    "Daly, Jameson": "Daly, Jamie",
    "Hall, Edward": "Hall, Ed",
    "Hamilton, Matthew": "Hamilton, Matt",
}

def main():
    all_csvs = sorted(glob.glob(os.path.join(DATA_ROOT, "**/CSV/*.csv"), recursive=True))
    all_csvs = [f for f in all_csvs if "positioning" not in f]
    print(f"Found {len(all_csvs)} CSV files")

    # Only keep files where Davidson is involved (either team column)
    frames = []
    skipped = 0
    for i, fp in enumerate(all_csvs):
        if (i + 1) % 500 == 0:
            print(f"  Processing {i+1}/{len(all_csvs)}...")
        try:
            df = pd.read_csv(fp, low_memory=False)
            # Keep row if Davidson is pitcher team, batter team, home, or away
            mask = False
            for col in ["PitcherTeam", "BatterTeam", "HomeTeam", "AwayTeam"]:
                if col in df.columns:
                    mask = mask | (df[col] == DAVIDSON_TEAM_ID)
            dav_rows = df[mask]
            if len(dav_rows) > 0:
                # Keep the FULL game data (including opponent rows) for scouting
                game_ids = dav_rows["GameID"].unique() if "GameID" in df.columns else []
                if len(game_ids) > 0:
                    frames.append(df[df["GameID"].isin(game_ids)])
                else:
                    frames.append(dav_rows)
        except Exception as e:
            skipped += 1
            continue

    if not frames:
        print("No Davidson data found!")
        sys.exit(1)

    print(f"Concatenating {len(frames)} files (skipped {skipped})...")
    data = pd.concat(frames, ignore_index=True)

    # Normalize
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Season"] = data["Date"].dt.year
    for col in ["Pitcher", "Batter", "Catcher"]:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()
            data[col] = data[col].replace(NAME_MAP)

    # Numeric columns
    num_cols = [
        "RelSpeed", "SpinRate", "InducedVertBreak", "HorzBreak",
        "PlateLocHeight", "PlateLocSide", "ExitSpeed", "Angle",
        "Direction", "Distance", "Extension", "RelHeight", "RelSide",
        "VertApprAngle", "HorzApprAngle", "SpinAxis", "VertBreak",
        "ZoneSpeed", "EffectiveVelo", "HangTime", "PopTime",
    ]
    for c in num_cols:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    # Keep only useful columns to reduce file size
    keep_cols = [
        "PitchNo", "Date", "Season", "PAofInning", "PitchofPA",
        "Pitcher", "PitcherId", "PitcherThrows", "PitcherTeam",
        "Batter", "BatterId", "BatterSide", "BatterTeam",
        "PitcherSet", "Inning", "Top/Bottom", "Outs", "Balls", "Strikes",
        "TaggedPitchType", "AutoPitchType", "PitchCall", "KorBB",
        "TaggedHitType", "PlayResult", "OutsOnPlay", "RunsScored",
        "RelSpeed", "VertRelAngle", "HorzRelAngle", "SpinRate", "SpinAxis", "Tilt",
        "RelHeight", "RelSide", "Extension",
        "VertBreak", "InducedVertBreak", "HorzBreak",
        "PlateLocHeight", "PlateLocSide", "ZoneSpeed",
        "VertApprAngle", "HorzApprAngle",
        "ExitSpeed", "Angle", "Direction", "Distance", "HangTime",
        "TaggedHitType", "EffectiveVelo",
        "HomeTeam", "AwayTeam", "Stadium", "GameID",
        "PopTime", "ExchangeTime",
        "Catcher", "CatcherId", "CatcherTeam",
    ]
    keep_cols = list(dict.fromkeys(c for c in keep_cols if c in data.columns))
    data = data[keep_cols].drop_duplicates()

    print(f"Final dataset: {len(data):,} rows, {len(data.columns)} columns")
    print(f"Seasons: {sorted(data['Season'].dropna().unique())}")
    print(f"Writing to {OUTPUT}...")
    data.to_parquet(OUTPUT, index=False)
    print("Done!")
    print(f"File size: {os.path.getsize(OUTPUT) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
