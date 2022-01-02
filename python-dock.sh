docker run \
    --privileged \
    --rm \
    -w /work\
    --net irnet\
    -v $PWD:/work \
    -it\
    irproj\
    python3.6 $1
