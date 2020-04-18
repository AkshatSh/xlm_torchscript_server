// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <csignal>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TSimpleServer.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>
#include "gen-cpp/Predictor.h"

#include "pistache/endpoint.h"
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <torch/script.h>

#include <glog/logging.h>
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
using json = nlohmann::json;

// Server objects need to be global so that the handler can kill them cleanly
unique_ptr<TSimpleServer> thriftServer;
unique_ptr<Http::Endpoint> restServer;
void shutdownHandler(int s) {
  if (thriftServer) {
    LOG(INFO) << "Shutting down Thrift server";
    thriftServer->stop();
  }
  if (restServer) {
    LOG(INFO) << "Shutting down REST proxy server";
    restServer->shutdown();
  }
  exit(0);
}

// Main handler for the predictor service
class PredictorHandler : virtual public PredictorIf {
 private:
  torch::jit::script::Module mModule;

  vector<string> mDummyVec;
  vector<vector<string>> mDummyVecVec;

  void tokenize(vector<string>& tokens, string& doc) {
    transform(doc.begin(), doc.end(), doc.begin(), ::tolower);
    size_t start = 0;
    size_t end = 0;
    for (size_t i = 0; i < doc.length(); i++) {
      if (isspace(doc.at(i))){
        end = i;
        if (end != start) {
          tokens.push_back(doc.substr(start, end - start));
        }

        start = i + 1;
      }
    }

    if (start < doc.length()) {
      tokens.push_back(doc.substr(start, doc.length() - start));
    }

    if (tokens.size() == 0) {
      // Add PAD_TOKEN in case of empty text
      tokens.push_back("<pad>");
    }
  }

 public:
  PredictorHandler(string& modelFile) {
    mModule = torch::jit::load(modelFile);
  }

  void predict(map<string, double>& _return, const string& doc) {
    LOG(INFO) << "Processing \"" << doc << "\"";
    // Pre-process: tokenize input doc
    vector<string> tokens;
    string docCopy = doc;
    tokenize(tokens, docCopy);

    if (VLOG_IS_ON(1)) {
      stringstream ss;
      ss << "[";
      copy(tokens.begin(), tokens.end(), ostream_iterator<string>(ss, ", "));
      ss.seekp(-1, ss.cur); ss << "]";
      VLOG(1) << "Tokens for \"" << doc << "\": " << ss.str();
    }

    // Prepare input for the model as a batch
    vector<vector<string>> batch{tokens};
    vector<torch::jit::IValue> inputs{
        mDummyVec, // texts in model.forward
        mDummyVecVec, // multi_texts in model.forward
        batch // tokens in model.forward
    };

    // Run the model
    auto output =
        mModule.forward(inputs).toGenericListRef().at(0).toGenericDict();

    // Extract and populate results into the response
    for (const auto& elem : output) {
      _return.insert({elem.key().toStringRef(), elem.value().toDouble()});
    }
    VLOG(1) << "Logits for \"" << doc << "\": " << _return;
  }
};

// Response Formatting for a specific API
class ResponseFormatter {
 public:
  static string format(const map<string, double>& scores, const string& text) {
    // Exponentiate
    map<string, double> expScores;
    transform(scores.begin(), scores.end(), inserter(expScores, expScores.begin()),
              [](const auto& p) {
                return make_pair(p.first, exp(p.second));
              });

    // Sum up
    double sum = accumulate(begin(expScores), end(expScores), 0.,
                            [](double previous, const auto& p) { return previous + p.second; });

    // Normalize (end up with softmax)
    map<string, double> normScores;
    transform(expScores.begin(), expScores.end(), inserter(normScores, normScores.begin()),
              [sum](const auto& p) {
                return make_pair(p.first, p.second / sum);
              });

    // Sort in descending order
    vector<pair<string, double>> sortedScores = sortMapByValue(normScores);
    VLOG(1) << "Normalized scores for \"" << text << "\": " << normScores;

    // Reformat into name / confidence pairs. Strip "intent:" prefix
    json ir = json::array();
    for (const auto& p : sortedScores) {
      ir.push_back({{mName, stripPrefix(p.first, mIntentPrefix)},
                    {mConfidence, p.second}});
    }

    json j;
    j[mText] = text;
    j[mIntentRanking] = ir;
    if (!ir.empty()) {
      j[mIntent] = ir.at(0);
    } else {
      j[mIntent] = nullptr;
    }
    j[mEntities] = json::array();

    LOG(INFO) << "Processed \"" << text << "\", predicted " << (j[mIntent] != nullptr ? j[mIntent].dump() : "no intents");
    return j.dump(2 /*indentation*/);
  }

  static const string mName;
  static const string mConfidence;
  static const string mIntentPrefix;
  static const string mText;
  static const string mIntentRanking;
  static const string mIntent;
  static const string mEntities;

  template <typename A, typename B>
  static vector<pair<A, B>> sortMapByValue(const map<A, B>& src) {
    vector<pair<A, B>> v{make_move_iterator(begin(src)),
                         make_move_iterator(end(src))};

    sort(begin(v), end(v),
         [](auto lhs, auto rhs) { return lhs.second > rhs.second; });  // descending order

    return v;
  }

  static string stripPrefix(const string& doc, const string& prefix) {
    if (doc.length() >= prefix.length()) {
      auto res = std::mismatch(prefix.begin(), prefix.end(), doc.begin());
      if (res.first == prefix.end()) {
        return doc.substr(prefix.length());
      }
    }
    return doc;
  }
};

const string ResponseFormatter::mName = "name";
const string ResponseFormatter::mConfidence = "confidence";
const string ResponseFormatter::mIntentPrefix = "intent:";
const string ResponseFormatter::mText = "text";
const string ResponseFormatter::mIntentRanking = "intent_ranking";
const string ResponseFormatter::mIntent = "intent";
const string ResponseFormatter::mEntities = "entities";

// REST proxy for the predictor Thrift service (not covered in tutorial)
class RestProxyHandler : public Http::Handler {
 private:
  shared_ptr<TTransport> mTransport;
  shared_ptr<PredictorClient> mPredictorClient;
  shared_ptr<ResponseFormatter> mResponseFormatter;

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
    if (!mTransport->isOpen()) {
      mTransport->open();
    }

    auto headers = request.headers();

    shared_ptr<Http::Header::ContentType> contentType;
    try {
      contentType = headers.get<Http::Header::ContentType>();
    } catch (runtime_error) {
      response.send(Http::Code::Bad_Request,
                    "Expected HTTP header Content-Type: application/json\n");
      return;
    }

    auto mediaType = contentType->mime();
    if (mediaType != MIME(Application, Json)) {
      response.send(Http::Code::Bad_Request,
                    "Expected HTTP header Content-Type: application/json, found " + mediaType.toString() + "\n");
      return;
    }

    const json requestBody = json::parse(request.body());
    string text;

    try {
      text = requestBody.at(mTextParam);
    } catch (json::out_of_range) {
      response.send(Http::Code::Bad_Request,
                    "Missing json parameter: " + mTextParam + "\n");
      return;
    }

    map<string, double> scores;
    mPredictorClient->predict(scores, text);
    response.send(Http::Code::Ok, mResponseFormatter->format(scores, text));
  }

  static const string mTextParam;
};

const string RestProxyHandler::mTextParam = "text";

int main(int argc, char **argv) {
  // Parse command line args
  if (argc < 2) {
    cerr << "Usage:" << endl;
    cerr << "./server <model file>" << endl;
    return 1;
  }

  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;
  string modelFile = argv[1];

  // Handle shutdown events
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = shutdownHandler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  // Initialize predictor thrift service
  shared_ptr<PredictorHandler> handler(new PredictorHandler(modelFile));
  shared_ptr<TProcessor> processor(new PredictorProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(FLAGS_port_thrift));
  shared_ptr<TTransportFactory> transportFactory(
      new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());
  thriftServer = make_unique<TSimpleServer>(
      processor, serverTransport, transportFactory, protocolFactory);
  thread thriftThread([&]() { thriftServer->serve(); });
  LOG(INFO) << "Thrift server running at port: " << FLAGS_port_thrift;

  if (FLAGS_rest) {
    // Initialize Thrift client used to foward requests from REST
    shared_ptr<TTransport> socket(new TSocket("127.0.0.1", FLAGS_port_thrift));
    shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
    shared_ptr<PredictorClient> predictorClient(new PredictorClient(protocol));

    // Initialize REST proxy
    Address addr(Ipv4::any(), Port(FLAGS_port_rest));
    auto opts = Http::Endpoint::options().threads(1);
    restServer = make_unique<Http::Endpoint>(addr);
    restServer->init(opts);
    restServer->setHandler(
        make_shared<RestProxyHandler>(transport, predictorClient));
    thread restThread([&]() { restServer->serve(); });

    LOG(INFO) << "REST proxy server running at port: " << FLAGS_port_rest;
    restThread.join();
  }

  thriftThread.join();
  return 0;
}
