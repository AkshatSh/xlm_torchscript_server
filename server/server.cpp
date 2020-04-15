// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// AKSHAT WAS HERE

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <math.h> 

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TSimpleServer.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>
#include "gen-cpp/Predictor.h"

#include "pistache/endpoint.h"
#include <sentencepiece_processor.h>

#include <curl/curl.h>

#include <torch/script.h>

#include <gflags/gflags.h>
DEFINE_int32(
    port_thrift,
    9090,
    "Port which the Thrift server should listen on");
DEFINE_bool(rest, true, "Set up a REST proxy to the Thrift server");
DEFINE_int32(port_rest, 8080, "Port which the REST proxy should listen on");

using namespace std;

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;
using namespace apache::thrift::server;
using namespace predictor_service;

using namespace Pistache;

// Main handler for the predictor service
class PredictorHandler : virtual public PredictorIf {
 private:
  torch::jit::script::Module mModule;
  sentencepiece::SentencePieceProcessor processor;

  c10::optional<vector<string>> c10mDummyVec;
  c10::optional<vector<vector<string>>> c10mDummyVecVec;

  void sentencepiece_tokenize(vector<string>& tokens, string& doc) {
    processor.Encode(doc, &tokens);
  }

 public:
  PredictorHandler(string& modelFile, string& sentencepieceVocabFile) {
    mModule = torch::jit::load(modelFile);
    // sentencepiece::SentencePieceProcessor processor;
    const auto status = processor.Load(sentencepieceVocabFile);
    if (!status.ok()) {
      std::cerr << status.ToString() << std::endl;
      // error
    } else {
      std::cout << "Successfully loaded in SentencePiece model" << std::endl;
    }
  }

  void predict(map<string, double>& _return, const string& doc) {
    // Pre-process: tokenize input doc
    vector<string> tokens;
    string docCopy = doc;
    sentencepiece_tokenize(tokens, docCopy);

    // Prepare input for the model as a batch
    vector<vector<string>> batch{tokens};
    vector<torch::jit::IValue> inputs{
        // tokens,
        c10mDummyVec, // texts in model.forward
        c10mDummyVecVec, // multi_texts in model.forward
        batch, // tokens in model.forward
        c10mDummyVec // languages
    };

    // Run the model
    auto output =
        mModule.forward(inputs).toGenericListRef().at(0).toGenericDict();

    // Extract and populate results into the response
    for (const auto& elem : output) {
      _return.insert({elem.key().toStringRef(), exp(elem.value().toDouble())});
    }
  }
};

// REST proxy for the predictor Thrift service (not covered in tutorial)
class RestProxyHandler : public Http::Handler {
 private:
  shared_ptr<TTransport> mTransport;
  shared_ptr<PredictorClient> mPredictorClient;

  string urlDecode(const string &encoded)
  {
    CURL *curl = curl_easy_init();
    int decodedLength;
    char *decodedCstr = curl_easy_unescape(
        curl, encoded.c_str(), encoded.length(), &decodedLength);
    string decoded(decodedCstr, decodedCstr + decodedLength);
    curl_free(decodedCstr);
    curl_easy_cleanup(curl);
    return decoded;
  }

 public:
  HTTP_PROTOTYPE(RestProxyHandler)

  RestProxyHandler(
      shared_ptr<TTransport>& transport,
      shared_ptr<PredictorClient>& predictorClient
    ) {
    mTransport = transport;
    mPredictorClient = predictorClient;
  }

  void onRequest(const Http::Request& request, Http::ResponseWriter response) {
    const string docParam = "doc";
    if (!mTransport->isOpen()) {
      mTransport->open();
    }

    stringstream out;
    if (request.query().has(docParam)) {
      string doc = urlDecode(request.query().get(docParam).get());
      map<string, double> scores;
      mPredictorClient->predict(scores, doc);
      for (const auto& score : scores) {
        out << score.first << ":" << score.second << endl;
        }

      response.send(Http::Code::Ok, out.str());
    }
    else {
      out << "Missing query parameter: " << docParam << endl;
      response.send(Http::Code::Bad_Request, out.str());
    }
  }
};

int main(int argc, char **argv) {
  // Parse command line args
  if (argc < 3) {
    cerr << "Usage:" << endl;
    cerr << "./server <XLM model file> <XLM sentencepiece vocab file>" << endl;
    return 1;
  }

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  string modelFile = argv[1];
  string sentencepiece_vocab = argv[2];

  // Initialize predictor thrift service
  shared_ptr<PredictorHandler> handler(new PredictorHandler(modelFile, sentencepiece_vocab));
  shared_ptr<TProcessor> processor(new PredictorProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(FLAGS_port_thrift));
  shared_ptr<TTransportFactory> transportFactory(
      new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());
  TSimpleServer thriftServer(
      processor, serverTransport, transportFactory, protocolFactory);
  thread thriftThread([&](){ thriftServer.serve(); });
  cout << "Server running. Thrift port: " << FLAGS_port_thrift;

  if (FLAGS_rest) {
    // Initialize Thrift client used to foward requests from REST
    shared_ptr<TTransport> socket(new TSocket("127.0.0.1", FLAGS_port_thrift));
    shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
    shared_ptr<PredictorClient> predictorClient(new PredictorClient(protocol));

    // Initialize REST proxy
    Address addr(Ipv4::any(), Port(FLAGS_port_rest));
    auto opts = Http::Endpoint::options().threads(1);
    Http::Endpoint restServer(addr);
    restServer.init(opts);
    restServer.setHandler(
        make_shared<RestProxyHandler>(transport, predictorClient));
    thread restThread([&](){ restServer.serve(); });

    cout << ", REST port: " << FLAGS_port_rest << endl;
    restThread.join();
  }

  cout << endl;
  thriftThread.join();
  return 0;
}
