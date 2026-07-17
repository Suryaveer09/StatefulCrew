# ─────────────────────────────────────────────────────────────────
# Phase 7: Dockerfile — containerizes the Phase 6 Streamlit app for
# deployment to Azure App Service.
# ─────────────────────────────────────────────────────────────────

# Slim variant keeps the image smaller than the full python image —
# we don't need the extra build tools/libraries it includes
FROM python:3.11-slim

# All subsequent COPY/RUN commands operate relative to /app inside the
# container, and it's also where the app actually runs from
WORKDIR /app

# curl lets us fetch the Chinook database at build time — avoids needing
# to manually copy a gitignored binary file into the build context
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
    # rm -rf on the apt cache keeps the image smaller — no reason to
    # ship package-manager metadata inside the final container

# Copying requirements.txt BEFORE the app code (rather than copying
# everything at once) lets Docker cache the pip install layer — as
# long as requirements.txt doesn't change, rebuilding after an app.py
# edit skips reinstalling every dependency from scratch
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY crew.py app.py .

# Same Chinook database used since Phase 2 — downloaded fresh at build
# time so the image is fully self-contained and reproducible. This also
# means DB_PATH in crew.py must be "chinook.db" (flat), matching where
# this file actually lands: WORKDIR /app, no subfolder.
RUN curl -L -o chinook.db https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite

# Documents which port the app listens on. This alone doesn't publish
# the port to the outside world — that's what `docker run -p` (locally)
# or Azure's WEBSITES_PORT setting (in the cloud) actually does.
EXPOSE 8501

# --server.port=8501 matches EXPOSE above.
# --server.address=0.0.0.0 is essential — without it, Streamlit only
# binds to localhost INSIDE the container, making it unreachable from
# outside (including from Azure's own health checks and real traffic).
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]