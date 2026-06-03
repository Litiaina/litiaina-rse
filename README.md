# Litiaina Reed Solomon Erasure GF(2^8)

A educational and experimental Reed-Solomon erasure coding implementation, not production storage software.

## Usage

### Encoding a file

```bash
RUST_LOG=info cargo run --release -- encode --input my_large_file.bin --output shards_out --data-shards 10 --parity-shards 4
```

### Decoding a file

```bash
RUST_LOG=info cargo run --release -- decode --input shards_out --output recovered_file.bin
```
