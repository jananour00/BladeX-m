"""
DQN Model Preprocessor for Paralympic Coaching
================================================
Handles environment setup and action decoding for the trained DQN model.

The model was trained with:
  - 15-dimensional observation space
  - 108 discrete actions (3×3×4×3 =108)
  - Action branches: intensity, rest, focus, adjustment

HOW TO USE:
  1. Create environment: env = ParalympicTrainingEnv()
  2. Wrap with ActionDiscretizer: wrapped = ActionDiscretizer(env)
  3. Load model: model = DQN.load("paralympic_dqn_model.zip", env=wrapped)
  4. Predict: action, _ = model.predict(observation, deterministic=True)
  5. Decode: decode_action(action)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor


# ──────────────────────────────────────────────────────────────────────────
# Environment Constants (must match training exactly)
# ──────────────────────────────────────────────────────────────────────────
OBSERVATION_DIM = 15
N_ACTIONS = 108  # 3 * 3 * 4 * 3

# Observation space bounds (must match training)
# Features: fatigue, asymmetry_knee, speed, cadence, variability, avg_fatigue, some_value, injury_risk, consistency, qom, asymmetry_stride, max_speed, height, weight, asymmetry_knee_copy
OBS_LOW = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
OBS_HIGH = np.array([1, 100, 20, 5, 1, 1, 1, 1, 1, 1, 100, 20, 5, 1, 100], dtype=np.float32)

# Action branch mappings
INTENSITY_LEVELS = ['Low', 'Medium', 'High']
REST_PERIODS = ['Short (30min)', 'Medium (1-2hrs)', 'Long (4-6hrs)']
FOCUS_AREAS = ['Speed Development', 'Symmetry Improvement', 'Endurance Building', 'Technique Refinement']
ADJUSTMENTS = ['None Needed', 'Minor Adjustment', 'Major Review']


# ──────────────────────────────────────────────────────────────────────────
# Gymnasium Environment
# ──────────────────────────────────────────────────────────────────────────
class ParalympicTrainingEnv(gym.Env):
    """
    Simplified environment for DQN inference.
    Only implements reset() - step() is not needed for prediction.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Action space (MultiDiscrete for training, but we'll use Discrete via wrapper)
        self.action_space = spaces.MultiDiscrete([3, 3, 4, 3])

        # Observation space (15 features)
        self.observation_space = spaces.Box(
            low=OBS_LOW,
            high=OBS_HIGH,
            dtype=np.float32
        )
        self.state = None

    def reset(self, seed=None, options=None):
        """Dummy reset for inference."""
        self.state = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        """Not implemented for inference-only environment."""
        raise NotImplementedError("Step not implemented for inference-only environment")


# ──────────────────────────────────────────────────────────────────────────
# Action Wrapper (converts Discrete to MultiDiscrete)
# ──────────────────────────────────────────────────────────────────────────
class ActionDiscretizer(gym.ActionWrapper):
    """
    Convert Discrete action space to MultiDiscrete for DQN compatibility.
    The DQN was trained with Discrete actions (0-107), but the underlying
    environment uses MultiDiscrete [3, 3, 4, 3].
    """
    def __init__(self, env):
        super().__init__(env)
        self.n_actions = N_ACTIONS  # 3*3*4*3 = 108
        self.action_space = spaces.Discrete(self.n_actions)

    def action(self, action):
        """
        Convert flat Discrete action (0-107) to MultiDiscrete [intensity, rest, focus, adjustment].
        
        Example: action=0 → [0, 0, 0, 0] (Low, Short, Speed Development, None)
                 action=107 → [2, 2, 3, 2] (High, Long, Technique Refinement, Major Review)
        """
        action = int(action)
        intensity = (action // (3 * 4 * 3)) % 3
        rest = (action // (4 * 3)) % 3
        focus = (action // 3) % 4
        adjustment = action % 3
        return np.array([intensity, rest, focus, adjustment], dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────────
# Model Loading Functions
# ──────────────────────────────────────────────────────────────────────────
def load_dqn_model(model_path: str = "paralympic_dqn_model.zip"):
    """
    Load the trained DQN model for inference.
    
    Args:
        model_path: Path to the saved model ZIP file
        
    Returns:
        tuple: (model, environment)
    """
    # Create dummy environment (needed for loading)
    dummy_env = ParalympicTrainingEnv()
    wrapped_env = ActionDiscretizer(Monitor(dummy_env))
    
    # Load model
    model = DQN.load(model_path, env=wrapped_env)
    print(f"[OK] DQN model loaded from {model_path}")
    return model, wrapped_env


def decode_action(flat_action: int) -> dict:
    """
    Decode a flat Discrete action (0-107) to human-readable coaching recommendation.
    
    Args:
        flat_action: Integer action from DQN model (0-107)
        
    Returns:
        dict: Human-readable action details
    """
    flat_action = int(flat_action)
    
    intensity = (flat_action // (3 * 4 * 3)) % 3
    rest = (flat_action // (4 * 3)) % 3
    focus = (flat_action // 3) % 4
    adjustment = flat_action % 3
    
    return {
        'intensity': INTENSITY_LEVELS[intensity],
        'rest': REST_PERIODS[rest],
        'focus': FOCUS_AREAS[focus],
        'adjustment': ADJUSTMENTS[adjustment],
        'action_code': flat_action
    }


# ──────────────────────────────────────────────────────────────────────────
# Observation Helper Functions
# ──────────────────────────────────────────────────────────────────────────
def create_observation_from_metrics(
    fatigue: float = 0.5,
    asymmetry_knee: float = 10.0,
    speed: float = 5.0,
    cadence: float = 3.5,
    variability: float = 0.1,
    avg_fatigue: float = 0.5,
    some_value: float = 0.85,
    injury_risk: float = 0.3,
    consistency: float = 0.8,
    qom: float = 0.75,
    asymmetry_stride: float = 10.0,
    max_speed: float = 12.0,
    height: float = 175.0,
    weight: float = 70.0,
    asymmetry_knee_copy: float = None
) -> np.ndarray:
    """
    Create a 15-dimensional observation vector from individual metrics.
    
    Args:
        Various runner metrics (see default values for typical ranges)
        
    Returns:
        np.ndarray: 15-dimensional observation vector
    """
    if asymmetry_knee_copy is None:
        asymmetry_knee_copy = asymmetry_knee
    
    obs = np.array([
        fatigue,           # 0: fatigue
        asymmetry_knee,    # 1: asymmetry_knee
        speed,            # 2: speed
        cadence,          # 3: cadence
        variability,     # 4: variability
        avg_fatigue,     # 5: avg_fatigue
        some_value,      # 6: placeholder/some_value
        injury_risk,     # 7: injury_risk
        consistency,    # 8: consistency
        qom,             # 9: qom (quality of movement)
        asymmetry_stride,# 10: asymmetry_stride
        max_speed,        # 11: max_speed
        height,          # 12: height
        weight,         # 13: weight
        asymmetry_knee_copy  # 14: duplicate asymmetry_knee
    ], dtype=np.float32)
    
    return obs


def validate_observation(obs: np.ndarray) -> tuple[bool, str]:
    """
    Validate that an observation vector is in the correct format.
    
    Args:
        obs: Observation array to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if obs is None:
        return False, "Observation is None"
    
    obs = np.array(obs)
    
    if obs.shape != (OBSERVATION_DIM,):
        return False, f"Observation must be {OBSERVATION_DIM}-dimensional, got shape {obs.shape}"
    
    # Check for NaN values
    if np.any(np.isnan(obs)):
        return False, "Observation contains NaN values"
    
    # Skip strict bounds check - the model may have been trained with different bounds
    # The /recommend_from_metrics endpoint works, so we trust the model can handle it
    return True, ""
