import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 10000
area = np.random.randint(40, 300, n_samples)
rooms = np.random.randint(1, 7, n_samples)
bathrooms = np.random.randint(1, 5, n_samples)

# Estimate cost based on simplified logic
base_cost = 50000
cost_per_m2 = 2000
room_cost = 40000
bathroom_cost = 30000
noise = np.random.normal(0, 20000, n_samples)  # Adding some randomness

estimated_cost = (
    base_cost +
    area * cost_per_m2 +
    rooms * room_cost +
    bathrooms * bathroom_cost +
    noise
).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Area_m2': area,
    'Rooms': rooms,
    'Bathrooms': bathrooms,
    'Estimated_Cost_INR': estimated_cost
})

# Save to CSV
df.to_csv("house_cost_prediction_dataset_10000.csv", index=False)
