import os

output_dir = os.path.join(os.path.dirname(__file__), "output")
print("Checking contents of:", output_dir)

# List files in output/
if os.path.exists(output_dir):
    print("Files in output/:", os.listdir(output_dir))
else:
    print("output/ folder does NOT exist")

# Construct full model path
model_path = os.path.join(output_dir, "fraud_best_model.pkl")
print("Resolved model path:", model_path)
print("Exists?", os.path.exists(model_path))
