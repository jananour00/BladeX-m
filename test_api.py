"""
Test script to verify the BladeX-m API endpoints
"""
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_health():
    """Test health endpoint"""
    resp = requests.get(f"{BASE_URL}/health")
    print("=== /health ===")
    print(json.dumps(resp.json(), indent=2))
    return resp.json()

def test_models_info():
    """Test models info endpoint"""
    resp = requests.get(f"{BASE_URL}/models/info")
    print("\n=== /models/info ===")
    print(json.dumps(resp.json(), indent=2))
    return resp.json()

def test_dqn_recommend():
    """Test DQN recommend_from_metrics endpoint"""
    data = {
        "fatigue": 0.6,
        "asymmetry_knee": 18.0,
        "speed": 5.2,
        "injury_risk": 0.55,
        "consistency": 0.78
    }
    resp = requests.post(f"{BASE_URL}/recommend_from_metrics", json=data)
    print("\n=== /recommend_from_metrics ===")
    print(json.dumps(resp.json(), indent=2))
    return resp.json()

def test_frontend():
    """Test frontend loads"""
    resp = requests.get(f"{BASE_URL}/")
    print("\n=== Frontend (/) ===")
    print(f"Status: {resp.status_code}")
    print(f"Content length: {len(resp.text)} chars")
    if "GaitAI" in resp.text:
        print("✓ Frontend contains 'GaitAI'")
    if "Biomechanics" in resp.text:
        print("✓ Frontend contains 'Biomechanics'")
    return resp.status_code == 200

def test_sample_csv():
    """Test DQN sample CSV is served"""
    resp = requests.get(f"{BASE_URL}/dqn_sample_runner.csv")
    print("\n=== /dqn_sample_runner.csv ===")
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        print("First 200 chars:")
        print(resp.text[:200])
    return resp.status_code == 200

if __name__ == "__main__":
    print("Testing BladeX-m API...\n")
    
    test_health()
    test_models_info()
    test_dqn_recommend()
    test_frontend()
    test_sample_csv()
    
    print("\n✓ All tests completed!")
