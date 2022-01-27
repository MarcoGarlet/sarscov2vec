docker run --rm -d --net irnet -h mongo-server -p 2717:27017 -v "$(pwd)/mongo":"/data/db" -w /data/db --name mongo mongo:latest

