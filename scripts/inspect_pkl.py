import joblib
from pathlib import Path
p = Path('src/models/best_random_forest.pkl')
if not p.exists():
    print('ERROR: file not found:', p)
    raise SystemExit(1)
try:
    m = joblib.load(p)
    print('OK: Loaded', p)
    print('Type:', type(m))
    print('Attributes: n_estimators=', getattr(m,'n_estimators',None), 'feature_importances_len=', len(getattr(m,'feature_importances_',[])) if hasattr(m,'feature_importances_') else None)
    # Print a short repr
    r = repr(m)
    print('Repr head:', r[:200])
except Exception as e:
    print('ERROR loading pickle:', e)
    raise
