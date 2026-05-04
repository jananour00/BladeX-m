#!/usr/bin/env python3
\"\"\"Terminal tester for integrated Fast Fatigue Model.
Usage: python test_fast_fatigue_terminal.py [save|test|demo]\"\"\"

import sys
import numpy as np
from integrate_fast_fatigue import (
    save_complete_model, load_complete_model, predict_fatigue, FatiguePredictor,
    generate_sample_sequence, DEVICE, COMPLETE_MODEL_PATH
)
from backend.preprocessors.fatigue_preprocessor import FEATURES_FATIGUE

def print_features_info():
    print(\"\\n📊 FEATURES (21 total):\")
    for i, feat in enumerate(FEATURES_FATIGUE):
        print(f\"  {i:2d}: {feat:<20} (range: realistic values)\")
    print()

def demo_predictor():
    print(\"🚀 LOADING FatiguePredictor...\")
    predictor = FatiguePredictor()
    
    print(\"\\n📈 GENERATING SAMPLE SEQUENCE (20 timesteps x 21 features)...\")
    seq = generate_sample_sequence()
    print(f\"   Shape: {seq.shape}\")
    print(f\"   Sample (first 3 timesteps, first 5 feats):\\n{seq[:3, :5]}\")
    
    print(\"\\n🔮 PREDICTING FATIGUE...\")
    fatigue, info = predictor.predict(seq)
    
    print(f\"\\n✅ RESULTS:\")
    print(f\"   Fatigue Score: {fatigue:.4f}\")
    print(f\"   Info: {info}\")
    print(f\"\\n💡 Interpretation: (adapt thresholds as needed)\")
    if fatigue < 0.3:
        print(\"   LOW fatigue - Good performance\")
    elif fatigue < 0.7:
        print(\"   MODERATE fatigue - Monitor\")
    else:
        print(\"   HIGH fatigue - Rest recommended\")

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'demo'
    
    print(\"= \" * 40)
    print(\"   FAST FATIGUE MODEL TERMINAL TEST\")
    print(\"= \" * 40)
    print_features_info()
    
    if mode == 'save':
        print(\"\\n💾 SAVING COMPLETE MODEL...\")
        save_complete_model()
        print(\"\\n✅ Run 'python test_fast_fatigue_terminal.py test' to test.\")
    
    elif mode == 'test':
        print(\"\\n🧪 TESTING LOAD + PREDICT...\")
        model, scaler, feats, seq_len = load_complete_model(COMPLETE_MODEL_PATH)
        seq = generate_sample_sequence()
        fatigue, info = predict_fatigue(model, scaler, seq, DEVICE)
        print(f\"✅ Fatigue: {fatigue:.4f}\")
    
    elif mode == 'demo':
        demo_predictor()
    
    else:
        print(f\"\\n❓ Unknown mode: {mode}\")
        print(\"   Usage: python test_fast_fatigue_terminal.py [save|test|demo]\")
    
    print(\"\\n🎉 Integration complete! Model ready for terminal use.\")

