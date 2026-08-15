# Split Excel Data Files

This directory contains GitHub-uploadable parts for the Excel datasets that
exceed the ordinary GitHub web-upload limit. Every part is 20 MB or smaller.
The original files in `../Data/` have not been changed.

## Files

- `data_0407.xlsx01` through `data_0407.xlsx05`
- `data_0409.xlsx01` through `data_0409.xlsx05`
- `data_0410.xlsx01` through `data_0410.xlsx04`
- `SHA256SUMS.txt`: SHA-256 checksums for every part

## Restore a dataset

Download all parts for one dataset into the same directory, then concatenate
them in numerical order. On macOS or Linux:

```bash
cat data_0407.xlsx01 data_0407.xlsx02 data_0407.xlsx03 data_0407.xlsx04 data_0407.xlsx05 > data_0407.xlsx
cat data_0409.xlsx01 data_0409.xlsx02 data_0409.xlsx03 data_0409.xlsx04 data_0409.xlsx05 > data_0409.xlsx
cat data_0410.xlsx01 data_0410.xlsx02 data_0410.xlsx03 data_0410.xlsx04 > data_0410.xlsx
shasum -a 256 -c SHA256SUMS.txt
```

The resulting `.xlsx` files are byte-for-byte identical to the originals.
