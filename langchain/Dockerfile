# Dockerfile
EXPOSE 8000 // [!code --]
CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8000 // [!code --]
ARG PORT // [!code ++]
EXPOSE ${PORT:-8000} // [!code ++]
CMD exec uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000} // [!code ++]
