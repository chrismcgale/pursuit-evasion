#pragma once
// Mirror of src/pursuit_evasion/env/link.py: the sim-to-real link as a wrapper
// around a controller (state latency, noise, occlusion, identity swaps, uplink
// latency, packet loss, motor lag, battery sag). PERFECT bypasses exactly.
//
// Every random outcome the link uses is a pure function of (seed, stream, tick)
// — nothing depends on the view or the command — so the deterministic half
// (delay interpolation, hold, swap, uplink) is mirrored here line for line and
// the random half comes from a DrawSource:
//   * TapeSource replays a tape written by Python (link.py:write_tape). Exact:
//     degraded C++ runs reproduce Python's bit for bit, as --starts-file does
//     for spawns.
//   * NativeSource draws from std::mt19937_64 with the same distributions.
//     Statistically equivalent, NOT paired with Python (numpy's PCG64 +
//     ziggurat normal are a different stream), for standalone C++ evals.
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "pe/controller.hpp"
#include "pe/sim.hpp"

namespace pe {

struct LinkProfile {
  std::string name = "perfect";
  double obs_latency_s = 0.0, pos_noise_m = 0.0, vel_noise_mps = 0.0;
  double dropout_p = 0.0, dropout_burst = 1.0, swap_p = 0.0, swap_burst = 1.0;
  double cmd_latency_s = 0.0, motor_tau_s = 0.0, packet_loss_p = 0.0;
  double thrust_scale = 1.0, sag_frac_per_min = 0.0;
  bool is_perfect() const {
    return obs_latency_s == 0.0 && pos_noise_m == 0.0 && vel_noise_mps == 0.0 &&
           dropout_p == 0.0 && swap_p == 0.0 && cmd_latency_s == 0.0 &&
           motor_tau_s == 0.0 && packet_loss_p == 0.0 && thrust_scale == 1.0 &&
           sag_frac_per_min == 0.0;
  }
};

// link.py:PRESETS — perfect | vicon_lab | vicon_busy | field_degraded. Throws on
// an unknown name.
LinkProfile link_preset(const std::string& name);

// link.py:TickDraws
struct TickDraws {
  std::vector<Vec3> pos_self, pos_opp, vel_self, vel_opp;   // empty when that noise is off
  std::vector<char> drop_self, drop_opp;
  int swap_i = -1, swap_j = -1;
  bool lost_packet = false;
};

class DrawSource {
 public:
  virtual ~DrawSource() = default;
  // Called on every LinkedController reset with the stream index (resets so far).
  virtual void begin_stream(int stream, int n_self, int n_opp) = 0;
  virtual TickDraws next() = 0;
};

class TapeSource : public DrawSource {
 public:
  TapeSource(const std::string& path, const LinkProfile& p);
  void begin_stream(int stream, int n_self, int n_opp) override;
  TickDraws next() override;

 private:
  LinkProfile p_;
  int n_self_ = 2, n_opp_ = 2;
  std::map<int, std::vector<TickDraws>> streams_;
  const std::vector<TickDraws>* cur_ = nullptr;
  int stream_ = -1;
  size_t t_ = 0;
};

class NativeSource : public DrawSource {
 public:
  NativeSource(const LinkProfile& p, uint64_t seed) : p_(p), seed_(seed) {}
  void begin_stream(int stream, int n_self, int n_opp) override;
  TickDraws next() override;

 private:
  struct Burst {
    double p_on, p_stay;
    bool on = false;
  };
  bool tick(Burst& b);
  LinkProfile p_;
  uint64_t seed_;
  std::mt19937_64 rng_;
  int n_self_ = 2, n_opp_ = 2, t_ = 0;
  std::vector<Burst> self_drop_, opp_drop_;
  Burst swap_{0, 0};
  int si_ = -1, sj_ = -1;
};

class LinkedController : public Controller {
 public:
  // dt is the control period. The source may be null only for a perfect link.
  LinkedController(Controller* inner, LinkProfile profile, std::unique_ptr<DrawSource> src,
                   double dt);
  void reset() override;
  std::vector<double> act(const TeamView& view) override;

  Controller* inner;
  LinkProfile profile;
  long n_dropout = 0, n_swap = 0, n_lost_packet = 0, ticks = 0;

 private:
  TeamView delayed(const TeamView& view) const;
  TeamView corrupt(const TeamView& view);
  std::vector<double> uplink(const std::vector<double>& cmd, bool lost);

  std::unique_ptr<DrawSource> src_;
  double dt_;
  int ep_ = -1;
  size_t hist_ = 2, cmd_hist_ = 1;
  std::vector<TeamView> views_;
  std::vector<std::vector<double>> cmds_;
  bool has_held_ = false;
  std::vector<Vec3> held_sp_, held_sv_, held_op_, held_ov_;
  std::vector<double> last_cmd_, thrust_;
  bool has_last_cmd_ = false, has_thrust_ = false, prev_swap_ = false;
  bool started_stream_ = false;
  TickDraws draws_;
};

}  // namespace pe
