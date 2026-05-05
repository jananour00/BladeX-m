# BladeX-m Cleanup & Full Backend Integration
## Status: Starting Implementation

### Complete Plan (Phased)

**Phase 1: Delete Unnecessary Files [Current Phase]**
1. [ ] Delete test/integration scripts: test_models.py, load_test.py, models_complete_api.py, causal_test.py, integrate_fast_fatigue_fixed.py, test_fast_fatigue_terminal_fixed.py, model_integration_complete.py, fatigue_model_terminal_test.py, FastTemporalTransformer.py, full_model_test.py, model_test_fixed.py, perfect_model_test.py
2. [ ] Delete data: dataset.csv, dqn_sample_runner.csv  
3. [ ] Delete test data: backend/test_bayes_data.csv, backend/test_bayes_single.json

**Phase 2: Create Missing Model Files**
4. [ ] Create `models/fatigue_injury_model/` dir + fatigue_pipeline.joblib, injury_pipeline.joblib, thresholds.json (minimal sklearn)
5. [ ] Create `models/Quality + Coach/` dir + quality_model.joblib, coach_model.joblib

**Phase 3: Fix Backend Core**
6. [ ] Edit backend/app.py: Uncomment Bayesian load, fix ModelStore.load_all(), paths
7. [ ] Edit backend/model_utils.py: Make loads robust (handle missing preprocessors)
8. [ ] Edit backend/__init__.py: Import all preprocessors correctly
9. [ ] Verify all preprocessors exist & complete

**Phase 4: Test Backend**
10. [ ] `pip install -r requirements.txt`
11. [ ] `cd backend && python app.py` → All models load ✅
12. [ ] Test /health: fatigue_pipeline:true, quality_model:true, etc.
13. [ ] Test endpoints: POST /predict/fatigue, /upload/full

**Phase 5: Frontend & Completion**
14. [ ] Test frontend unified analysis
15. [ ] attempt_completion

### Commands Ready
```
pip install -r requirements.txt
cd backend && python app.py
curl http://localhost:5000/health
```

**Current Step: Phase 1 - Deletions**

