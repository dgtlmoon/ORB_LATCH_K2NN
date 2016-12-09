# ORB_LATCH_K2NN

## About 

Implementation of komrad36's LATCH and K2NN libraries from komrad36, based on ORB keypoints from OpenCV's features2d

- https://github.com/komrad36/LATCH
- https://github.com/komrad36/K2NN

## Usage

`--index-path <path>` Builds a `training.dat` of all descriptors found in images located at `<path>` (and a random image is chosen and added to `query.dat`)

Or to create a predictable set of 255 descriptors of 64 bytes each (512-bit) into training.dat

`--test-index` Builds a `training.dat` of all descriptors found in images located at `<path>` (and a random image is chosen and added to `query.dat`)

Last byte of each 8 byte descriptor is set `0..255`

```
 [first 7 byte]..[last byte]
 11111111111111..0000001
 11111111111111..0000010
 11111111111111..0000011
 ...
 11111111111111..1111101
 11111111111111..1111110
 11111111111111..1111111
```
 
 


`--search` Search all of `training.dat` for the nearest neighbours (K2NN) from descriptors in `query.dat`


## Goal
  
To be able to find images sorted by their _"similarity"_ in terms of ORB feature similarity to detect tshirts with 
similar designs on them but taken by different photographers at different distances.

### Problems

None of the descriptors are matche at all

```
./ORB_LATCH_K2NN --search
Loaded 64 bytes from query.dat
Loaded 16320 bytes from training.dat
First descriptor in query.dat looks like.. 
11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111100000000
First descriptor in training.dat looks like.. 
11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111100000000
---- fastApproxMatch results ----
Warming up...
.....
No matches found
---- bruteMatch results ----
Warming up...
.....
No matches found
---- exactMatch results ----
Warming up...
.....
No matches found


```
### Todo

- Use CUDA libraries once basics are working

dgtlmoon@gmail.com
