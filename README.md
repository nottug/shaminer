# SHA256 Miner

This is a simple CUDA miner for SHA256 adapted from [CudaSHA256](https://github.com/Horkyze/CudaSHA256).

## Usage

```
usage: ./shaminer {INPUT} {TARGET}
```

`{INPUT}` is the seed to hash, `{TARGET}` is the target to match. An example
could be `./shaminer test 21e800`.