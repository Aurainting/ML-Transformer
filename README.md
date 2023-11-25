# Introduction

mlpack-based Transformer 

- [x] Forward
- [ ] Backward

# Usage

Run the `scripts/init.sh` file, and a Docker image named `ml_transformer:v1.0` will be generated, which is the runtime environment.

After preparing the runtime environment, go to the project directory (the same directory as this `README.md`), execute the following command
```bash
cmake -B build -S .
cmake --build build/
```
and then the corresponding executable file will be generated in the `bin` directory.

# Future Work

Add `Backward` function in Transformer.

