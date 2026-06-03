# Litiaina Reed-Solomon Erasure Coding GF(2^8)

An educational and experimental Reed-Solomon erasure coding implementation over GF(2^8), intended for learning, testing, and storage-system research.

This project is not presented as production-ready storage software.

## Usage

### Encoding a file

```bash
RUST_LOG=info cargo run --release -- encode --input my_large_file.bin --output shards_out --data-shards 10 --parity-shards 4
```

### Decoding a file

```bash
RUST_LOG=info cargo run --release -- decode --input shards_out --output recovered_file.bin
```
