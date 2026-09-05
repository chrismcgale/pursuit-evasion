// pe_run: evaluate a controller (scripted / rl / bt / bt_safe) on the seeded
// scenario battery and report the same metrics as the Python harness
// (src/pursuit_evasion/eval/scenarios.py), for cross-runtime validation.
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <memory>
#include <string>
#include <vector>

#include "pe/games.hpp"
#include "pe/gate.hpp"
#include "pe/onnx_policy.hpp"
#include "pe/safety.hpp"
#include "pe/scripted.hpp"
#include "pe/sim.hpp"

using namespace pe;

namespace {

struct Args {
  std::string game = "tag";           // tag | assault | escort
  std::string side = "pursuers";
  std::string controller = "bt";      // scripted | rl | bt | bt_safe
  std::string arena;                   // defaults per game
  std::string onnx;                    // defaults per game
  std::string tree;                    // defaults per game + side
  std::string log;                     // optional JSON output
  std::string starts_file;             // optional: identical starts to Python
  int episodes = 100;
  uint64_t seed0 = 10000;
  bool groot = false;
};

// Which gate profile a side runs in a given game — mirrors
// gating.py:default_profile. The objective games route to different trees
// because the regime where the scripted law is weak is a *different* regime
// (see the ablation table at the top of trees/gate_defenders.xml).
std::string default_tree(const std::string& game, bool pursuer_side) {
  if (game == "tag")
    return pursuer_side ? "trees/gate_pursuers.xml" : "trees/gate_evaders.xml";
  return pursuer_side ? "trees/gate_defenders.xml" : "trees/gate_attackers.xml";
}

// Model stem convention shared with eval/games.py:_model_stems.
std::string default_onnx(const std::string& game, bool pursuer_side) {
  if (game == "tag")
    return pursuer_side ? "../models/pursuer.onnx" : "../models/evader.onnx";
  return "../models/" + game + "_dagger.onnx";
}

// Load episodes of explicit starts: one line per episode, 12 whitespace-separated
// floats = pursuer0 xyz, pursuer1 xyz, evader0 xyz, evader1 xyz.
std::vector<Starts> load_starts_file(const std::string& path) {
  std::vector<Starts> out;
  std::ifstream f(path);
  double v[12];
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    bool ok = true;
    for (double& x : v)
      if (!(ss >> x)) { ok = false; break; }
    if (!ok) continue;
    out.push_back({{"pursuer0", {v[0], v[1], v[2]}},
                   {"pursuer1", {v[3], v[4], v[5]}},
                   {"evader0", {v[6], v[7], v[8]}},
                   {"evader1", {v[9], v[10], v[11]}}});
  }
  return out;
}

std::string arg_val(int argc, char** argv, const std::string& key, const std::string& def) {
  for (int i = 1; i < argc - 1; ++i)
    if (key == argv[i]) return argv[i + 1];
  return def;
}
bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i)
    if (key == argv[i]) return true;
  return false;
}

struct EpRec {
  uint64_t seed;
  bool win;              // pursuer-slot team achieved its objective
  int n_captured;
  int n_spent;           // of those, missiles that burned out rather than were hit
  int steps_to_all;      // -1 if not a win
  double time_to_first;  // -1 if none
  double min_sep;
  int geofence_viol;
  int speed_viol;
  // objective games
  bool breach;
  double asset_margin;   // closest any live attacker ever got to the asset
};

ShieldedController* as_shield(Controller* c) {
  return dynamic_cast<ShieldedController*>(c);
}

// Fixed start configurations shared with the Python parity check. Running
// scripted-vs-scripted from identical starts must match Python bit-for-bit
// (same libmujoco, same ctrl) — isolating logic bugs from RNG differences.
// Tag-only: the objective games use --starts-file for the same purpose.
std::vector<Starts> parity_starts() {
  return {
      {{"pursuer0", {-8, 1, 3}}, {"pursuer1", {-8, -1, 3}}, {"evader0", {8, 1, 3}}, {"evader1", {8, -1, 4}}},
      {{"pursuer0", {-6, 3, 2}}, {"pursuer1", {-7, -2, 5}}, {"evader0", {6, -3, 3}}, {"evader1", {7, 2, 4}}},
      {{"pursuer0", {-9, 0, 4}}, {"pursuer1", {-5, 4, 2}}, {"evader0", {5, 0, 5}}, {"evader1", {8, -4, 3}}},
      {{"pursuer0", {-4, -4, 3}}, {"pursuer1", {-8, 2, 6}}, {"evader0", {7, 3, 2}}, {"evader1", {4, -2, 5}}},
      {{"pursuer0", {-7, -3, 5}}, {"pursuer1", {-6, 1, 3}}, {"evader0", {6, 4, 4}}, {"evader1", {9, -1, 2}}},
  };
}

int run_parity(Sim& sim) {
  InterceptPursuers P;
  FieldEvaders E;
  auto starts = parity_starts();
  for (size_t k = 0; k < starts.size(); ++k) {
    sim.reset_with_starts(starts[k]);
    double checksum = 0.0;
    StepInfo info;
    int step = 0;
    while (true) {
      TeamView pv = sim.view(Team::Pursuers), ev = sim.view(Team::Evaders);
      info = sim.step(P.act(pv), E.act(ev));
      checksum += info.min_dist;
      ++step;
      if (info.terminated || info.truncated) break;
    }
    std::printf("[cpp-parity] cfg=%zu win=%d steps=%d checksum=%.6f\n", k,
                info.all_captured ? 1 : 0, info.steps, checksum);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  a.game = arg_val(argc, argv, "--game", a.game);
  a.side = arg_val(argc, argv, "--side", a.side);
  a.controller = arg_val(argc, argv, "--controller", a.controller);
  a.episodes = std::stoi(arg_val(argc, argv, "--episodes", std::to_string(a.episodes)));
  a.seed0 = std::stoull(arg_val(argc, argv, "--seed", std::to_string(a.seed0)));
  a.log = arg_val(argc, argv, "--log", "");
  a.starts_file = arg_val(argc, argv, "--starts-file", "");
  a.groot = has_flag(argc, argv, "--groot");
  bool test_pursuers = (a.side == "pursuers");

  const GameSpec spec = make_game(a.game);   // throws on an unknown key
  a.arena = arg_val(argc, argv, "--arena", "assets/" + arena_asset_name(a.game));
  a.onnx = arg_val(argc, argv, "--onnx", default_onnx(a.game, test_pursuers));
  a.tree = arg_val(argc, argv, "--tree", default_tree(a.game, test_pursuers));

  Sim sim(a.arena, spec);
  if (has_flag(argc, argv, "--parity")) {
    if (a.game != "tag") {
      std::fprintf(stderr, "[cpp] --parity is tag-only; use --starts-file for %s\n",
                   a.game.c_str());
      return 2;
    }
    return run_parity(sim);
  }
  Team test_team = test_pursuers ? Team::Pursuers : Team::Evaders;

  // --- build the controller under test + its opponent -----------------------
  // The scripted matchup is per-game: tag runs intercept-vs-field, the objective
  // games run guard-defenders-vs-missile-raid.
  std::vector<std::unique_ptr<Controller>> owned;
  std::unique_ptr<OnnxPolicy> policy;
  auto keep = [&](Controller* c) { owned.emplace_back(c); return c; };

  ControllerPair pair = default_controllers(a.game);
  Controller* scripted_test = keep((test_pursuers ? pair.pursuer : pair.evader).release());
  Controller* tested = nullptr;
  if (a.controller == "scripted") {
    tested = scripted_test;
  } else {
    if (a.controller == "rl" || a.controller == "bt" || a.controller == "bt_safe") {
      policy = std::make_unique<OnnxPolicy>(a.onnx);
    }
    if (a.controller == "rl") {
      tested = keep(new RLController(policy.get(), "rl"));
    } else {
      Controller* rl = keep(new RLController(policy.get(), "rl"));
      Controller* gate =
          keep(new GatedController(test_team, scripted_test, rl, a.tree, {}, a.groot));
      tested = (a.controller == "bt_safe") ? keep(new ShieldedController(gate)) : gate;
    }
  }

  // opponent is always the scripted baseline of the other side
  ControllerPair opp_pair = default_controllers(a.game);
  Controller* opponent =
      keep((test_pursuers ? opp_pair.evader : opp_pair.pursuer).release());

  Controller* pursuer_ctrl = test_pursuers ? tested : opponent;
  Controller* evader_ctrl = test_pursuers ? opponent : tested;

  const double dt = sim.dt();
  std::vector<Starts> ep_starts;
  if (!a.starts_file.empty()) {
    ep_starts = load_starts_file(a.starts_file);
    std::printf("[cpp] loaded %zu explicit starts from %s\n", ep_starts.size(),
                a.starts_file.c_str());
  }
  int n_eps = ep_starts.empty() ? a.episodes : (int)ep_starts.size();
  std::vector<EpRec> recs;
  recs.reserve(n_eps);

  // GatedController::reset() zeroes its own counters at the top of every episode,
  // so accumulate here — otherwise the reported mode split is just the last one.
  Controller* unwrapped = as_shield(tested) ? as_shield(tested)->inner : tested;
  auto* gate_ctrl = dynamic_cast<GatedController*>(unwrapped);
  long ticks_scripted = 0, ticks_rl = 0;

  for (int e = 0; e < n_eps; ++e) {
    uint64_t seed = a.seed0 + e;
    if (ep_starts.empty())
      sim.reset(seed);
    else
      sim.reset_with_starts(ep_starts[e]);
    pursuer_ctrl->reset();
    evader_ctrl->reset();
    ShieldedController* shield = as_shield(tested);

    double min_sep = std::numeric_limits<double>::infinity();
    double asset_margin = std::numeric_limits<double>::infinity();
    double time_to_first = -1.0;
    int prev_cap = 0;
    StepInfo info;
    while (true) {
      TeamView pv = sim.view(Team::Pursuers);
      TeamView ev = sim.view(Team::Evaders);
      auto pa = pursuer_ctrl->act(pv);
      auto ea = evader_ctrl->act(ev);
      info = sim.step(pa, ea);
      if (info.n_captured < 2) min_sep = std::min(min_sep, info.min_dist);
      if (std::isfinite(info.asset_dist))
        asset_margin = std::min(asset_margin, info.asset_dist);
      if (time_to_first < 0 && info.n_captured > prev_cap)
        time_to_first = info.steps * dt;
      prev_cap = info.n_captured;
      if (info.terminated || info.truncated) break;
    }
    EpRec r;
    r.seed = seed;
    // Always the pursuer-slot objective, whichever side is under test — mirrors
    // scenarios.py (`win=r.info["pursuer_win"]`), so the two runtimes' win rates
    // are directly comparable. In tag that is "both evaders tagged"; in the
    // objective games a timeout is a *defender* win, so the sim decides it.
    r.win = info.pursuer_win;
    r.n_captured = info.n_captured;
    r.n_spent = info.n_spent;
    r.steps_to_all = info.all_captured ? info.steps : -1;
    r.time_to_first = time_to_first;
    r.min_sep = std::isinf(min_sep) ? 0.0 : min_sep;
    r.geofence_viol = shield ? shield->filter.n_geofence : 0;
    r.speed_viol = shield ? shield->filter.n_speed : 0;
    r.breach = info.breach;
    r.asset_margin = std::isinf(asset_margin) ? 0.0 : asset_margin;
    recs.push_back(r);
    if (gate_ctrl) {
      ticks_scripted += gate_ctrl->mode_counts["scripted"];
      ticks_rl += gate_ctrl->mode_counts["rl"];
    }
  }

  // --- aggregate ------------------------------------------------------------
  int wins = 0, caps = 0, geo = 0, spd = 0, nwins = 0, breaches = 0, spent = 0;
  double sum_steps = 0, sum_minsep = 0, sum_first = 0, sum_margin = 0;
  int nfirst = 0;
  for (auto& r : recs) {
    wins += r.win;
    caps += r.n_captured;
    spent += r.n_spent;
    geo += r.geofence_viol;
    spd += r.speed_viol;
    sum_minsep += r.min_sep;
    breaches += r.breach;
    sum_margin += r.asset_margin;
    // mean_steps_to_win averages over *wins* in tag; steps_to_all is -1 when the
    // objective was met without a clean sweep, so skip those
    if (r.win && r.steps_to_all >= 0) { sum_steps += r.steps_to_all; ++nwins; }
    if (r.time_to_first >= 0) { sum_first += r.time_to_first; ++nfirst; }
  }
  int n = (int)recs.size();
  double win_rate = n ? (double)wins / n : 0;
  double mean_caps = n ? (double)caps / n : 0;
  double mean_steps = nwins ? sum_steps / nwins : NAN;
  double mean_minsep = n ? sum_minsep / n : 0;
  double mean_first = nfirst ? sum_first / nfirst : NAN;
  double breach_rate = n ? (double)breaches / n : 0;
  double spent_rate = n ? (double)spent / n : 0;
  double mean_margin = n ? sum_margin / n : 0;

  char extra[128] = {0};
  if (spec.has_asset)
    std::snprintf(extra, sizeof(extra), " breach=%.2f assetmargin=%.1f spent=%.2f",
                  breach_rate, mean_margin, spent_rate);

  std::printf("[cpp] game=%s side=%s controller=%-8s n=%d | win=%.2f caps=%.2f "
              "steps2win=%s minsep=%.2f t2first=%.2fs viol(geo/spd)=%d/%d%s\n",
              a.game.c_str(), a.side.c_str(), a.controller.c_str(), n, win_rate,
              mean_caps,
              // round, not truncate — scenarios.py prints "%.0f" and the two
              // runtimes' rows are meant to be diffable verbatim
              std::isnan(mean_steps) ? "  -" : std::to_string(std::lround(mean_steps)).c_str(),
              mean_minsep, mean_first, geo, spd, extra);
  // which regimes the tree actually routed to, over the whole batch — the number
  // that makes the gate argument checkable rather than asserted (see the
  // ablation table in trees/gate_defenders.xml)
  if (gate_ctrl) {
    const long tot = ticks_scripted + ticks_rl;
    std::printf("[cpp] gate=%s ticks: scripted=%ld rl=%ld (rl_share=%.2f)\n",
                a.tree.c_str(), ticks_scripted, ticks_rl,
                tot ? (double)ticks_rl / tot : 0.0);
  }

  if (!a.log.empty()) {
    std::ofstream f(a.log);
    f << "{\n  \"game\": \"" << a.game << "\", \"side\": \"" << a.side
      << "\", \"controller\": \"" << a.controller << "\", \"runtime\": \"cpp\",\n";
    f << "  \"aggregate\": {\"n\": " << n << ", \"win_rate\": " << win_rate
      << ", \"mean_captures\": " << mean_caps << ", \"mean_steps_to_win\": "
      << (std::isnan(mean_steps) ? -1 : mean_steps) << ", \"mean_min_separation\": "
      << mean_minsep << ", \"geofence_viol\": " << geo << ", \"speed_viol\": " << spd
      << ", \"breach_rate\": " << breach_rate << ", \"spent_rate\": " << spent_rate
      << ", \"mean_min_asset_dist\": " << (spec.has_asset ? mean_margin : -1)
      << "},\n  \"episodes\": [\n";
    for (size_t i = 0; i < recs.size(); ++i) {
      auto& r = recs[i];
      f << "    {\"seed\": " << r.seed << ", \"win\": " << (r.win ? "true" : "false")
        << ", \"n_captured\": " << r.n_captured << ", \"n_spent\": " << r.n_spent
        << ", \"steps_to_all\": " << r.steps_to_all
        << ", \"min_separation\": " << r.min_sep << ", \"geofence_viol\": " << r.geofence_viol
        << ", \"speed_viol\": " << r.speed_viol
        << ", \"breach\": " << (r.breach ? "true" : "false")
        << ", \"min_asset_dist\": " << (spec.has_asset ? r.asset_margin : -1) << "}"
        << (i + 1 < recs.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
    std::printf("[cpp] wrote %s\n", a.log.c_str());
  }
  return 0;
}
