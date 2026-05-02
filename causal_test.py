import pickle
import numpy as np

print("=== CAUSAL GRAPH TEST - PROFESSIONAL VERSION ===\n")

# ================================
# 1. LOAD GRAPH SAFELY
# ================================
try:
    with open('causal_graph_complete.pkl', 'rb') as f:
        graph = pickle.load(f)
    print("✅ Graph Loaded Successfully!")
except FileNotFoundError:
    print("❌ ERROR: causal_graph_complete.pkl not found!")
    exit()

# Extract components
nodes = graph.get('nodes', [])
edges = graph.get('edges', [])
ace = graph.get('causal_effects', {})

print(f"Nodes: {len(nodes)} | Edges: {len(edges)}\n")

# ================================
# 2. INPUT (INTERVENTION)
# ================================
base_fatigue = 0.5
intervention = 0.2   # reduce fatigue

print("📥 INPUT:")
print(f"Base fatigue: {base_fatigue}")
print(f"Intervention: -{intervention}\n")

# ================================
# 3. APPLY INTERVENTION (do-operator)
# ================================
new_fatigue = max(0, base_fatigue - intervention)
delta_fatigue = new_fatigue - base_fatigue   # negative

print("🔬 INTERVENTION APPLIED:")
print(f"New fatigue: {new_fatigue:.3f}")
print(f"ΔFatigue: {delta_fatigue:.3f}\n")

# ================================
# 4. PROPAGATE EFFECTS (GRAPH BASED)
# ================================
def propagate_effects(delta, ace_dict):
    results = {}

    # QoM
    qom_coeff = ace_dict.get('fatigue_to_qom_ace', -0.25)
    results['QoM'] = delta * qom_coeff

    # Injury
    injury_coeff = ace_dict.get('fatigue_to_injury_ace', 0.4)
    results['Injury'] = delta * injury_coeff

    return results

effects = propagate_effects(delta_fatigue, ace)

# ================================
# 5. ADD UNCERTAINTY (ADVANCED)
# ================================
def add_uncertainty(value, std=0.02):
    return np.random.normal(value, std)

effects_uncertain = {k: add_uncertainty(v) for k, v in effects.items()}

# ================================
# 6. OUTPUT RESULTS
# ================================
print("📊 CAUSAL EFFECTS:")

for k, v in effects_uncertain.items():
    print(f"{k} change: {v:.3f}")

# ================================
# 7. CORRECT INTERPRETATION
# ================================
print("\n🎯 INTERPRETATION:")

if effects['QoM'] > 0:
    print("✔ QoM improved")
else:
    print("✖ QoM worsened")

if effects['Injury'] > 0:
    print("⚠ Injury risk increased")
else:
    print("✔ Injury risk decreased")

# ================================
# 8. FINAL DECISION LOGIC
# ================================
print("\n🧠 DECISION:")

if effects['QoM'] > 0 and effects['Injury'] < 0:
    print("🔥 Optimal: Performance improved & risk reduced (WIN-WIN)")
elif effects['QoM'] > 0 and effects['Injury'] > 0:
    print("⚖ Trade-off: Better performance but higher risk")
else:
    print("❗ Not beneficial intervention")

# ================================
# 9. GRAPH INFO
# ================================
print("\n📈 GRAPH STRUCTURE:")
print("Nodes:", nodes if nodes else "N/A")
print("Edges:", edges if edges else "N/A")

print("\n🚀 Ready for counterfactual simulation & multi-step causal inference!")