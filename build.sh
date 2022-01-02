docker build -t padel-box -f ./build/DockerfilePadel . 
docker build -t irproj -f ./build/Dockerfilepy . 
docker network create --subnet=172.18.0.0/16 irnet
docker pull mongo
