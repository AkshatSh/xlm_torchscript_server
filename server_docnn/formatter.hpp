
#include <string>
#include <nlohmann/json.hpp>
#include <glog/logging.h>

using namespace std;
using json = nlohmann::json;

class Formatter {
 public:
  string formatRequest(const string& requestBody) {
    const json jsonRequest = json::parse(requestBody);
    
    string text;
    try {
      text = jsonRequest.at(mTextParam);
    } catch (json::out_of_range e) {
      throw out_of_range(e.what());
    }
    
    return text;
  }

  string formatResponse(const map<string, double>& scores, const string& text) {
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

  static const string mTextParam;
  static const string mName;
  static const string mConfidence;
  static const string mIntentPrefix;
  static const string mText;
  static const string mIntentRanking;
  static const string mIntent;
  static const string mEntities;

  template <typename A, typename B>
  vector<pair<A, B>> sortMapByValue(const map<A, B>& src) {
    vector<pair<A, B>> v{make_move_iterator(begin(src)),
                         make_move_iterator(end(src))};

    sort(begin(v), end(v),
         [](auto lhs, auto rhs) { return lhs.second > rhs.second; });  // descending order

    return v;
  }

  string stripPrefix(const string& doc, const string& prefix) {
    if (doc.length() >= prefix.length()) {
      auto res = std::mismatch(prefix.begin(), prefix.end(), doc.begin());
      if (res.first == prefix.end()) {
        return doc.substr(prefix.length());
      }
    }
    return doc;
  }
};

const string Formatter::mTextParam = "text";
const string Formatter::mName = "name";
const string Formatter::mConfidence = "confidence";
const string Formatter::mIntentPrefix = "intent:";
const string Formatter::mText = "text";
const string Formatter::mIntentRanking = "intent_ranking";
const string Formatter::mIntent = "intent";
const string Formatter::mEntities = "entities";