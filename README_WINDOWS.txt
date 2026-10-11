LIMBO PREDICTOR — WINDOWS SETUP

1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/ . During installation, tick “Add python.exe to PATH”.
2. Extract this ZIP to a simple folder, e.g. Desktop\limbo_predictor_windows.
3. Open that folder in File Explorer. Click the address bar, type cmd, and press Enter.
4. Create a virtual environment:
   py -m venv .venv
5. Activate it (Command Prompt):
   .venv\Scripts\activate
   If using PowerShell instead, run: .venv\Scripts\Activate.ps1
6. Upgrade pip and install packages:
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
7. Run the experiment (may take a few minutes and use substantial RAM):
   python run_experiments.py
8. When it finishes and reports “done”, create the charts:
   python make_figures.py
9. Create the written summary:
   python make_summary.py
10. Check these outputs in the same project folder:
   results\results.json
   results\test_preds.npz
   figures\fig1...fig6 PNG files
   results_summary.md

IMPORTANT INTERPRETATION
- The project uses synthetic/simulated transactions, not real SBI, GPay, NPCI, or bank telemetry.
- The model is a multi-task feed-forward neural network implemented in NumPy, not a GRU.
- Do not run the scripts from a different folder; open Command Prompt in the extracted project folder first.
- If `py` is not recognized, try `python --version`; if Python is missing, reinstall it and enable Add Python to PATH.
- If PowerShell blocks activation, use Command Prompt steps instead.
