from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    module_dir = Path(__file__).resolve().parent
    print(project_root)
    print(module_dir)
