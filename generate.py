import pandas as pd
import numpy as np

# Generate random data for classification demo
np.random.seed(42)
feature1 = np.random.uniform(0, 10, 100)
feature2 = np.random.uniform(0, 10, 100)
label = np.where(feature1 + feature2 > 10, 1, 0)

# Create DataFrame
classification_data = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'label': label
})

# Save to CSV
classification_file_path = 'classification_data.csv'
classification_data.to_csv(classification_file_path, index=False)

print(f"CSV file generated and saved as {classification_file_path}")
