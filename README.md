# ORB_LATCH_K2NN

## About 

Implementation of komrad36's LATCH and K2NN libraries from komrad36, based on ORB keypoints from OpenCV's features2d

- https://github.com/komrad36/LATCH
- https://github.com/komrad36/K2NN

## Usage

`--index-path <path>` Builds a `training.dat` of all descriptors found in images located at `<path>` (and a random image is chosen and added to `query.dat`)

OR

`--search <type>` Search all of `training.dat` for the nearest neighbours (K2NN) from descriptors in `query.dat`, where type in `brute-force` or `fast-approx`


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
