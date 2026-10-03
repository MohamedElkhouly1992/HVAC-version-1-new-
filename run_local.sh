#!/usr/bin/env bash
set -e
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
