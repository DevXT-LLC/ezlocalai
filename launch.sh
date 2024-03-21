#!/bin/bash
streamlit run ui.py --server.headless true --server.port 8501 &
echo "Waiting for the back end to start"
uvicorn app:app --host 0.0.0.0 --port 8091 --workers 1 --proxy-headers