from fastapi import FastAPI
from router import api_router

docs_kwargs = {}
app = FastAPI(**docs_kwargs)

app.include_router(api_router)