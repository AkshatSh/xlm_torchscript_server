# TorchScript Server for XLM-R
In this repository we have a server along with a console for both and XLM-R document classification model along with a DocNN document classification model, that are trained in PyText and exported into torchscript.


## Server
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


## Console
The console contains a quick HTML page to view the predictions of the document model in a webpage. 

{TODO screenshots after Mrinal's CSS changes}

### Console Setup

```
$ python3 -m venv env
$ source env/bin/activate
$ (env) pip install -r requirements.txt
```

running the server:
```
$ python server.py -- {ARGS}
```
