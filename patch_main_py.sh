#!/bin/bash
sed -i 's/app.include_router(admin_router)/app.include_router(admin_router, dependencies=[Depends(resolve_admin_token)])/' backend/main.py

sed -i '/from fastapi import FastAPI, HTTPException, Request/c\
from fastapi import FastAPI, HTTPException, Request, Depends' backend/main.py
