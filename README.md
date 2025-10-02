# Litiaina Reed Solomon Erasure GF(2^8)

A Reed-Solomon erasure coding implementation over **GF(2^8)** designed for **Litiaina D2R** to provide data durability, redundancy, and atomic recovery.

## Usage

### Encoding a file

```bash
RUST_LOG=info cargo run --release -- encode --input my_large_file.bin --output shards_out --data-shards 10 --parity-shards 4
```

### Decoding a file

```bash
RUST_LOG=info cargo run --release -- decode --input shards_out --output recovered_file.bin
```
