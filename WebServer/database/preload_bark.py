import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from bark import preload_models

if __name__ == "__main__":
    print("Downloading/loading Bark models...")
    preload_models()
    print("Bark models are ready.")