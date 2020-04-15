```
$ mkdir build && cd build
$ echo -e 'FROM pytext/xlm_predictor_service_torchscript:latest\nCOPY xlm_model.pt.torchscript /app\nCOPY sentencepiece.bpe.model /app\nCMD ["./server","xlm_model.pt.torchscript", "sentencepiece.bpe.model"]' >> Dockerfile
$ docker build -t server .
$ docker run -it -p 8080:8080 server
$ curl -G "http://localhost:8080" --data-urlencode "doc=what is"
```
