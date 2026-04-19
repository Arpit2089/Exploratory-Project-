import os

def bundle_python_files(root_dir, output_file):
    """
    Recursively finds all .py files in root_dir, ignoring 'venv' folders,
    and writes their content into output_file with markers showing the 
    immediate parent folder and file name.
    """
    ignore_folder = 'venv'
    
    # Get the absolute path of the output file to avoid reading itself
    abs_output_path = os.path.abspath(output_file)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(root_dir):
            # Skip the venv directory entirely
            if ignore_folder in dirs:
                dirs.remove(ignore_folder)
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    # Safety check to avoid reading the output file
                    if os.path.abspath(file_path) == abs_output_path:
                        continue

                    # Get the name of the folder the file is sitting in
                    parent_folder = os.path.basename(root)
                    # If it's in the base directory, label it as [Root]
                    if not parent_folder or parent_folder == '.':
                        parent_folder = "Root Directory"

                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            
                            # START MARKER with Folder and Filename
                            outfile.write(f"{'#'*80}\n")
                            outfile.write(f"FOLDER: {parent_folder}\n")
                            outfile.write(f"FILE:   {file}\n")
                            outfile.write(f"PATH:   {file_path}\n")
                            outfile.write(f"{'#'*80}\n\n")
                            
                            # CONTENT
                            outfile.write(content)
                            
                            # END MARKER
                            outfile.write(f"\n\n{'='*80}\n")
                            outfile.write(f"END OF: {parent_folder}/{file}\n")
                            outfile.write(f"{'='*80}\n\n\n")
                            
                        print(f"Processed: {parent_folder}/{file}")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    # Settings
    target_dir = "."  # Start from current folder
    output_name = "combined_project_code.txt"
    
    print("Gathering Python files...")
    bundle_python_files(target_dir, output_name)
    print(f"\nDone! You can find the code in {output_name}")