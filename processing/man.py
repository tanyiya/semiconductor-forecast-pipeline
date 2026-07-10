import os

# Change this if you want another folder
ROOT_DIR = r"D:\01_Bomi\03_Projects\semiconductor-forecast-pipeline"


def print_directory_tree(path, indent=""):
    items = sorted(os.listdir(path))

    for index, item in enumerate(items):
        full_path = os.path.join(path, item)

        is_last = index == len(items) - 1
        branch = "└── " if is_last else "├── "

        print(indent + branch + item)

        if os.path.isdir(full_path):
            new_indent = indent + ("    " if is_last else "│   ")
            print_directory_tree(full_path, new_indent)


if __name__ == "__main__":
    print(ROOT_DIR)
    print_directory_tree(ROOT_DIR)