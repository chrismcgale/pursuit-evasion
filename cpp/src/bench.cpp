// pe_bench: per-tick wall-clock breakdown of the ground-station control stack
// (ROADMAP #6, feeds #18: can one process hold 4 aircraft at 100+ Hz?).
//
// Times, per control tick, every stage the production stack runs for a team of
// two aircraft: the scripted law, ONNX inference (team-joint policy, so once
// per team), feature extraction + BehaviorTree.CPP tick (once per aircraft) and
// the safety filter. Physics is timed separately: on hardware it is the world,
// not ground-station work.
//
//   ./build/pe_bench --game tag --episodes 20                 # 2 aircraft
//   ./build/pe_bench --game tag --episodes 20 --aircraft 4    # both teams gated
//   ./build/pe_bench --game assault --replicas 4              # 8 aircraft, synthetic
//
// --aircraft 4 flies BOTH teams through a gated+shielded stack in one process
// (tag only: the evader side needs gate_evaders.xml and an evader policy, see
// --onnx-evader). --replicas K ticks K independent copies of the pursuer-side
// stack on the same view each tick: 2K aircraft of compute, for the scaling
// slope. Copies do not fly; one copy's actions drive the sim.
//
// The numbers depend on the host. Read them as ratios and breakdowns, not as a
// budget for the ground station.
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "pe/games.hpp"
#include "pe/gate.hpp"
#include "pe/onnx_policy.hpp"
#include "pe/safety.hpp"
#include "pe/scripted.hpp"
#include "pe/sim.hpp"
#include "pe/timing.hpp"

using namespace pe;

namespace {

std::string arg_val(int argc, char** argv, const std::string& key, const std::string& def) {
  for (int i = 1; i < argc - 1; ++i)
    if (key == argv[i]) return argv[i + 1];
  return def;
}

// One team's production stack: Shielded(Gated(scripted, ONNX policy)).
struct Stack {
  ControllerPair pair;
  std::unique_ptr<OnnxPolicy> policy;
  std::unique_ptr<RLController> rl;
  std::unique_ptr<GatedController> gate;
  TeamSafetyFilter filter;
  Team team;
  std::string prefix;

  Stack(const std::string& game, Team t, const std::string& onnx, const std::string& tree,
        TimingSink* sink, std::string pre)
      : pair(default_controllers(game)), team(t), prefix(std::move(pre)) {
    policy = std::make_unique<OnnxPolicy>(onnx);
    rl = std::make_unique<RLController>(policy.get());
    Controller* law = (t == Team::Pursuers ? pair.pursuer : pair.evader).get();
    gate = std::make_unique<GatedController>(t, law, rl.get(), tree);
    gate->timing = sink;
    gate->timing_prefix = prefix;
  }
  void reset() {
    gate->reset();
    filter.reset();
  }
  std::vector<double> act(const TeamView& v, TimingSink& sink) {
    auto t0 = Clock::now();
    auto a = gate->act(v);
    auto t1 = Clock::now();
    auto out = filter.apply(v, a);
    sink.add(prefix + "safety_filter/team", us_since(t1));
    sink.add(prefix + "stack_total/team", us_since(t0));
    return out;
  }
};

}  // namespace

int main(int argc, char** argv) {
  const std::string game = arg_val(argc, argv, "--game", "tag");
  const int episodes = std::stoi(arg_val(argc, argv, "--episodes", "20"));
  const uint64_t seed0 = std::stoull(arg_val(argc, argv, "--seed", "20000"));
  const int aircraft = std::stoi(arg_val(argc, argv, "--aircraft", "2"));
  const int replicas = std::stoi(arg_val(argc, argv, "--replicas", "1"));
  const std::string csv = arg_val(argc, argv, "--csv", "");
  const bool tag = game == "tag";
  const std::string onnx_p = arg_val(argc, argv, "--onnx",
                                     tag ? "../models/pursuer_dagger.onnx"
                                         : "../models/" + game + "_dagger.onnx");
  const std::string onnx_e = arg_val(argc, argv, "--onnx-evader", "../models/evader.onnx");
  const std::string tree_p = tag ? "trees/gate_pursuers.xml" : "trees/gate_defenders.xml";
  const std::string tree_e = tag ? "trees/gate_evaders.xml" : "trees/gate_attackers.xml";
  if (aircraft != 2 && aircraft != 4) {
    std::fprintf(stderr, "--aircraft must be 2 or 4\n");
    return 2;
  }

  const GameSpec spec = make_game(game);
  Sim sim("assets/" + arena_asset_name(game), spec);
  TimingSink sink;

  std::vector<std::unique_ptr<Stack>> pstacks;
  for (int k = 0; k < replicas; ++k)
    pstacks.push_back(std::make_unique<Stack>(
        game, Team::Pursuers, onnx_p, tree_p, &sink,
        replicas > 1 ? "P" + std::to_string(k) + "/" : "P/"));
  std::unique_ptr<Stack> estack;
  ControllerPair opp = default_controllers(game);
  if (aircraft == 4)
    estack = std::make_unique<Stack>(game, Team::Evaders, onnx_e, tree_e, &sink, "E/");

  long ticks = 0;
  for (int e = 0; e < episodes; ++e) {
    sim.reset(seed0 + e);
    for (auto& s : pstacks) s->reset();
    if (estack) estack->reset();
    opp.evader->reset();
    while (true) {
      TeamView pv = sim.view(Team::Pursuers), ev = sim.view(Team::Evaders);
      auto t0 = Clock::now();
      std::vector<double> pa;
      for (size_t k = 0; k < pstacks.size(); ++k) {
        auto a = pstacks[k]->act(pv, sink);
        if (k == 0) pa = a;
      }
      std::vector<double> ea = estack ? estack->act(ev, sink) : std::vector<double>{};
      sink.add("ground_station_tick/all_aircraft", us_since(t0));
      if (!estack) ea = opp.evader->act(ev);   // scripted opponent: not timed
      auto t1 = Clock::now();
      StepInfo info = sim.step(pa, ea);
      sink.add("physics_step (reference)", us_since(t1));
      ++ticks;
      if (info.terminated || info.truncated) break;
    }
  }
  const int n_aircraft = aircraft == 4 ? 4 : 2 * replicas;
  char title[256];
  std::snprintf(title, sizeof title,
                "pe_bench game=%s aircraft=%d (replicas=%d) episodes=%d ticks=%ld "
                "control_period=%.0f ms",
                game.c_str(), n_aircraft, replicas, episodes, ticks, sim.dt() * 1000);
  sink.print_table(title);
  const double p99 = TimingSink::pct(sink.us["ground_station_tick/all_aircraft"], 0.99);
  std::printf("\nground-station tick p99 = %.1f us for %d aircraft = %.2f%% of a 10 ms "
              "(100 Hz) period; %.1f us per aircraft\n",
              p99, n_aircraft, p99 / 100.0, p99 / n_aircraft);
  if (!csv.empty()) sink.write_histogram_csv(csv);
  return 0;
}
