#!/bin/bash
# Run once:  bash setup.sh
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup done. Next: source venv/bin/activate && python prepare_data.py"
