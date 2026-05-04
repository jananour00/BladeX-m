# QoM Transformer Model Integration TODO

## Approved Plan Steps:
- [ ] Step 1: Create backend/preprocessors/qom_preprocessor.py (new file)
- [x] Step 2: Update backend/model_utils.py
- [x] Step 3: Update backend/app.py (ModelStore.load_all(), /predict/qom, /upload/qom endpoints)
- [ ] Step 4: mkdir models/qom/ && cp qom_transformer_model.pth models/qom/
- [ ] Step 5: Test: python backend/app.py, curl /health (qom_model: true), test endpoints
- [ ] Step 6: Update frontend (app.js + index.html) for QoM UI
- [ ] Complete: attempt_completion

**Current progress: Starting Step 1**
