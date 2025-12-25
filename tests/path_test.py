import datetime
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    module_dir = Path(__file__).resolve().parent
    print(project_root)
    print(module_dir)
    print(Path(__file__).resolve().parents[0])
    print(Path(__file__).resolve().parents[2])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    print(timestamp)
