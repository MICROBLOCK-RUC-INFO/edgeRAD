import torch
import os
# Learning rates
LR_ACTOR = float(os.getenv("LR_ACTOR", 1e-4))
LR_CRITIC = float(os.getenv("LR_CRITIC", 1e-3))
LR_PREDICT = float(os.getenv("LR_PREDICT", 1e-3))

# Hyperparameters
GAMMA = float(os.getenv("GAMMA", 0.8))
TAU = float(os.getenv("TAU", 0.005))
MEMORY_SIZE = int(os.getenv("MEMORY_SIZE", 100000))
batch_size = int(os.getenv("batch_size", 64))

# Bounds
MINIMUN = float(os.getenv("MINIMUN", -30))
MAXIMUN = float(os.getenv("MAXIMUN", 30))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Single service configuration
SERVICE_NAME = os.getenv("SERVICE_NAME", "A")
EXP_MODE = os.getenv("EXP_MODE", "simulation")
# EXP_MODE = os.getenv("EXP_MODE", "real")

# State and action dimensions for single service
STATE_LEN = int(os.getenv("STATE_LEN", 10))
ACTION_LEN = int(os.getenv("ACTION_LEN", 10))  
ACTION_MAX_NUM = int(os.getenv("ACTION_MAX_NUM", 10))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", 512))

# INTERVAL_LEN = int(os.getenv("INTERVAL_LEN", 1))  

# sample
RESERVOIR_LEN = int(os.getenv("RESERVOIR_LEN",70))
FEATURE_LEN = int(os.getenv("FEATURE_LEN", 70))

# Training hyperparameters
NUM_EPISODE = int(os.getenv("NUM_EPISODE", 666666))
NUM_STEP = int(os.getenv("NUM_STEP", 200))
EPSILON_START = float(os.getenv("EPSILON_START", 1.0))
EPSILON_END = float(os.getenv("EPSILON_END", 0.066))
EPSILON_DECAY = int(os.getenv("EPSILON_DECAY", 10000))
SAVE_INTERVAL_EPISODE = int(os.getenv("SAVE_INTERVAL_EPISODE", 10))  # Save model every N episodes

# Recovery bound
RECOVERY_BOUND = float(os.getenv("RECOVERY_BOUND", 0.6))
ANOMALY_THR = float(os.getenv("ANOMALY_THR", 0.8))

# Database configuration
DB_USERNAME = os.getenv("DB_USERNAME", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "12345678")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_NAME = os.getenv("DB_NAME", "service_monitoring")
DB_PORT = os.getenv("DB_PORT", "3306")
