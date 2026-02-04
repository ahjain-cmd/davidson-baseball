Deployment (DigitalOcean Droplet)

Overview
- Uses Docker + docker compose
- Streamlit runs on port 8501
- Data mounted from a persistent directory on the droplet

Prereqs on the droplet
1. Ubuntu 22.04+ droplet with at least 4 GB RAM
2. Docker installed
3. Git installed

Setup steps (first time)
1. Clone the repo:
   git clone <your-repo-url> /opt/davidson_baseball
   cd /opt/davidson_baseball

2. Copy data to a persistent folder:
   sudo mkdir -p /opt/davidson_data
   sudo cp /path/to/all_trackman.parquet /opt/davidson_data/
   sudo cp /path/to/davidson.duckdb /opt/davidson_data/
   # optional
   sudo cp /path/to/export.json /opt/davidson_data/

3. Create .env:
   cp deploy/.env.example .env
   # then edit .env with your token and data dir

4. Install systemd service:
   sudo cp deploy/davidson-baseball.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable davidson-baseball
   sudo systemctl start davidson-baseball

Deploy updates (after pushing code)
1. SSH into droplet
2. Run:
   /opt/davidson_baseball/deploy/deploy.sh

Notes
- App will be at http://<droplet-ip>:8501
- Use a reverse proxy (Caddy or Nginx) for HTTPS + custom domain
