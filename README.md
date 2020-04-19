For monolingual,
```
$ mkdir build && cd build
$ # Copy the downloaded models into this directory
$ echo -e 'FROM pytext/predictor_service_torchscript:who\nCOPY *.torchscript /app/\nCMD ["./server","mono.model.pt.torchscript"]' >> Dockerfile
$ docker build -t server .
$ docker run -it -p 8080:8080 server
$ curl -d '{"text": "hi"}' -H 'Content-Type: application/json' localhost:8080
```
For multilingual,
```
echo -e 'FROM pytext/predictor_service_torchscript:who\nCOPY *.torchscript /app/\nCMD ["./server","multi.model.pt.torchscript", "multi.vocab.model.pt.torchscript"]' >> Dockerfile
```
