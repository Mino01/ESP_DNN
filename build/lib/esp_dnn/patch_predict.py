import os

def patch_predict_py(file_path=None):
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), "predict.py")

    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    skip_indent = None
    for line in lines:
        # Remove tf.get_default_graph()
        if "tf.get_default_graph()" in line:
            continue
        # Remove "with self.graph.as_default():"
        elif "with self.graph.as_default():" in line:
            skip_indent = len(line) - len(line.lstrip()) + 4  # indent of block
            continue
        elif skip_indent is not None:
            # Dedent block
            indent = len(line) - len(line.lstrip())
            if indent >= skip_indent:
                new_lines.append(line[4:] if line.startswith("    ") else line)
                continue
            else:
                skip_indent = None

        new_lines.append(line)

    with open(file_path, "w") as f:
        f.writelines(new_lines)

    print(f"[✓] Patched {file_path} for TensorFlow 2.x compatibility.")

if __name__ == "__main__":
    patch_predict_py()
