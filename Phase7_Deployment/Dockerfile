FROM python:3.11-slim

WORKDIR /app

# curl lets us fetch the Chinook database at build time — avoids needing
# to manually copy a gitignored binary file into the build context
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY crew.py app.py .

# Same Chinook database used since Phase 2 — downloaded fresh at build
# time so the image is fully self-contained and reproducible
RUN curl -L -o chinook.db https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]