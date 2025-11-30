from utils.toml import read_toml

if __name__ == "__main__":
    project = read_toml("./pyproject.toml")["project"]
    print(f"ONNX PaddleOCR CLI 工具 v{project['version']}")
