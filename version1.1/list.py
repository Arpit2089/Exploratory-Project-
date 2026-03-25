import os

def print_tree(path):
    for item in os.listdir(path):
        full = os.path.join(path, item)

        if item == "venv":
            continue

        print(full)

        if os.path.isdir(full):
            print_tree(full)

print_tree(".")