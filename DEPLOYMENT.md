# Deployment Notes

## Docker

Recommended production-style run:

```bash
cp .env.server .env
docker-compose up -d --build
```

Application URL:
- `http://localhost:8000/`

## Persistent Data

These folders should be preserved across deployments:
- `saved_configs/`
- `saved_libraries/`
- `saved_programs/`
- `station/`
- `meshes/`

## Environment

Important variables:
- `BACKEND_HOST`
- `BACKEND_PORT`
- `CORS_ORIGINS`

## Security

- Replace development credentials in `backend/auth.py`
- Restrict `CORS_ORIGINS`
- Put HTTPS/TLS in front of the app for public deployment

## Operations

```bash
docker-compose logs -f
docker-compose ps
docker-compose restart
docker-compose down
```
