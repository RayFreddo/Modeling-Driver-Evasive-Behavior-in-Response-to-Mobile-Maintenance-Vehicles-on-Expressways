# Split Data Files

Download every numbered part for a dataset, keep them in one directory, then
run:

```bash
cat data_0407.xlsx01 data_0407.xlsx02 data_0407.xlsx03 data_0407.xlsx04 data_0407.xlsx05 > data_0407.xlsx
cat data_0409.xlsx{01,02,03,04,05,06,07,08,09,10} > data_0409.xlsx
cat data_0410.xlsx{01,02,03,04,05,06,07,08} > data_0410.xlsx
shasum -a 256 -c SHA256SUMS.txt
```

`SHA256SUMS.txt` verifies downloaded parts. Restored Excel files are identical
to the originals.
