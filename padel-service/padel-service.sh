docker run -d\
    --privileged \
    -w /work\
    -v "$PWD/padel-service/":"/work" \
    -it\
    --rm \
    --net irnet\
    -h padel-server\
    --publish 2323:2323\
    padel-box\
    socat TCP-LISTEN:2323,reuseaddr,fork EXEC:"python3 padel-daemon.py"
