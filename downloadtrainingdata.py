import kagglehub

# Download latest version
path = kagglehub.dataset_download("saiharim/fifa-player-faces")

print("Path to dataset files:", path)