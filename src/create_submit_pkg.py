import datetime
import glob
import subprocess
import tarfile
import os


def create_submit_pkg():

    # Source files
    src_files = glob.glob("src/*.py")

    # Notebooks
    notebooks = glob.glob("*.ipynb")

    # Checkpoints file
    checkpoints_file = "checkpoints/transfer_exported.pt"
    checkpoints_files = [checkpoints_file] if os.path.exists(checkpoints_file) else []

    # Generate HTML files from the notebooks
    for nb in notebooks:
        cmd_line = f"jupyter nbconvert --to html {nb}"

        print(f"executing: {cmd_line}")
        subprocess.check_call(cmd_line, shell=True)

    html_files = glob.glob("*.htm*")

    now = datetime.datetime.today().isoformat(timespec="minutes").replace(":", "h")+"m"
    outfile = f"submission_{now}.tar.gz"
    print(f"Adding files to {outfile}")

    with tarfile.open(outfile, "w:gz") as tar:
        # Add source files
        for name in src_files:
            print(name)
            tar.add(name)

        # Add notebooks
        for name in notebooks:
            print(name)
            tar.add(name)

        # Add HTML files
        for name in html_files:
            print(name)
            tar.add(name)

        # Add specific checkpoints file
        for file in checkpoints_files:
            print(file)
            tar.add(file, arcname=os.path.basename(file))

    print("")
    msg = f"Done. Please submit the file {outfile}"
    print("-" * len(msg))
    print(msg)
    print("-" * len(msg))


if __name__ == "__main__":
    create_submit_pkg()
