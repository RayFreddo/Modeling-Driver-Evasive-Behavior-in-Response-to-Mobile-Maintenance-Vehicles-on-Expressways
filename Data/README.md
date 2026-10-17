# Trajectory workbooks

Five source XLSX files are split into 24 MiB binary parts for GitHub web upload. The parts are not individual spreadsheets. They preserve every byte of the original workbooks, including all three worksheets. Upload every `.partNNN` file, `manifest.json`, and `reassemble.py`.

Run `python reassemble.py` after downloading this directory. The script checks each part and the reconstructed XLSX against SHA-256 before keeping it in `restored_workbooks/`. All five original files under the workspace `Data` directory remain untouched.
