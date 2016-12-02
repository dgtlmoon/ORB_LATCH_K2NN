# ORB_LATCH_K2NN

## About 

Implementation of komrad36's LATCH and K2NN libraries from komrad36, based on ORB descriptors from OpenCV's features2d

- https://github.com/komrad36/LATCH
- https://github.com/komrad36/K2NN

## Usage

Builds a `training.dat` of all descriptors found in images located at `--index-path` (and a random image is chosen and added to `query.dat`)

OR

Search all of `training.dat` for the nearest neighbours (K2NN) from descriptors in `query.dat`


```
Basic Command Line Parameter App
Options:
  --help                Print help messages
  --index-path arg      /path/to; Create an index/training.dat file from images
                        and a simple file with a single image query.dat set
  --search arg          fast-approx or brute-force; Search training.dat for 
                        what is present in query.dat
```

  
### Todo

- Use CUDA libraries once basics are working

dgtlmoon@gmail.com
