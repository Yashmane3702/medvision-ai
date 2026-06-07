# AI Heart Disease Prediction

Simple Streamlit app to predict heart disease risk using a pre-trained model.

Requirements
- Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate on Windows
pip install -r requirements.txt
```

Run

```bash
streamlit run app.py
```

Model files
- The app expects three files in the same folder or uploaded via the app UI:
  - `knn_heart_model.pkl`
  - `heart_scaler.pkl`
  - `heart_columns.pkl`

Place them next to `app.py` or upload them when prompted in the UI.
