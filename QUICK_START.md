# Quick Start

## DEMO mode

```bash
python3 -m http.server 8080 --directory frontend
```

Open `http://localhost:8080/simulator.html?mode=demo`

## PRO mode

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/`

Default development login:
- user: `admin`
- password: `robotics`

## Docker

```bash
cp .env.local .env
docker-compose up --build
```

Open `http://localhost:8000/`

## Companion Repository

Paper sources, benchmark scripts, and experiment datasets:
- `https://github.com/ual-arm/robodimm_paper`
