import os
import re

def patch_keras_imports(root_dir="esp_dnn"):
    keras_pattern = re.compile(r"\bfrom keras(?:\.engine\.topology)?\b")
    replacements = {
        "from keras.engine.topology import Layer, InputSpec":
            "from tensorflow.keras.layers import Layer\nfrom tensorflow.keras.engine.input_spec import InputSpec",
        "from keras.engine.topology import Layer":
            "from tensorflow.keras.layers import Layer",
        "from keras.engine.topology import InputSpec":
            "from tensorflow.keras.engine.input_spec import InputSpec",
        "from keras.models": "from tensorflow.keras.models",
        "from keras.layers": "from tensorflow.keras.layers",
        "from keras.optimizers": "from tensorflow.keras.optimizers",
        "from keras import backend as K": "from tensorflow.keras import backend as K",
    }

    files_patched = []

    for subdir, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".py"):
                path = os.path.join(subdir, filename)
                with open(path, "r") as f:
                    lines = f.readlines()

                modified = False
                new_lines = []
                for line in lines:
                    new_line = line
                    for old, new in replacements.items():
                        if old in line:
                            new_line = new_line.replace(old, new)
                            modified = True
                    new_lines.append(new_line)

                if modified:
                    with open(path, "w") as f:
                        f.writelines(new_lines)
                    files_patched.append(path)

    if files_patched:
        print("[✓] Patched Keras imports in the following files:")
        for file in files_patched:
            print(f"  - {file}")
    else:
        print("[i] No changes were necessary — looks like it's already patched.")

if __name__ == "__main__":
    patch_keras_imports()
