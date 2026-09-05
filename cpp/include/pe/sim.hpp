#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "pe/config.hpp"
#include "pe/dynamics.hpp"
#include "pe/games.hpp"
#include "pe/vec3.hpp"

struct mjModel_;
struct mjData_;

namespace pe {

// Mirrors src/pursuit_evasion/env/core.py:TeamView
struct TeamView {
  Team team;
  std::vector<Vec3> self_pos, self_vel;
  std::vector<char> self_alive;
  std::vector<Vec3> opp_pos, opp_vel;
  std::vector<char> opp_alive;
  double time_frac = 0.0;
  const ArenaConfig* arena = nullptr;
  double vmax = 0.0, opp_vmax = 0.0;
  // --- objective games; the defaults leave `tag` controllers untouched -------
  std::string game = "tag";
  const DynamicsProfile* self_dyn = nullptr;
  const DynamicsProfile* opp_dyn = nullptr;
  std::vector<double> self_fuel;   // remaining burn fraction in [0,1], per self agent
  std::vector<double> opp_fuel;
  bool has_asset = false;
  Vec3 asset_pos{0, 0, 0};
  Vec3 asset_vel{0, 0, 0};
  double asset_radius = 0.0;

  int n_self() const { return (int)self_pos.size(); }
  int n_opp() const { return (int)opp_pos.size(); }
};

struct StepInfo {
  std::vector<char> captured;
  int n_captured = 0;
  int n_spent = 0;              // of those, missiles that burned out rather than hit
  double min_dist = 0.0;
  bool all_captured = false;
  bool timeout = false;
  int steps = 0;
  bool terminated = false;
  bool truncated = false;
  // objective games
  bool breach = false;          // an attacker reached the asset
  bool asset_arrived = false;   // the escort convoy made its goal
  double asset_dist = 0.0;      // nearest live attacker -> asset (NaN if no asset)
  bool pursuer_win = false;     // the pursuer-slot team achieved its objective
};

using Starts = std::map<std::string, Vec3>;

// Mirrors src/pursuit_evasion/env/core.py:PursuitEvasionCore (dynamics + captures).
// Loads the same arena.xml and links the same libmujoco as the Python env.
class Sim {
 public:
  // `game` supplies the objective, dynamics profiles and spawn geometry; the
  // arena/episode default to that game's own and can be overridden for sweeps.
  Sim(const std::string& xml_path, GameSpec game = game_tag(), uint64_t seed = 0);
  Sim(const std::string& xml_path, ArenaConfig arena, EpisodeConfig ep,
      uint64_t seed = 0);
  ~Sim();

  void reset(uint64_t seed);                 // sample random starts
  void reset_with_starts(const Starts& s);   // explicit starts (replay / parity)
  StepInfo step(const std::vector<double>& pursuer_action,
                const std::vector<double>& evader_action);

  TeamView view(Team team) const;
  Vec3 pos(const std::string& name) const;
  Vec3 vel(const std::string& name) const;
  Starts sample_starts();
  const Starts& last_starts() const { return last_starts_; }

  double dt() const { return arena_.timestep * ep_.control_repeat; }
  Vec3 asset_position() const;
  Vec3 asset_velocity() const;

  const GameSpec& game() const { return game_; }
  const ArenaConfig& arena() const { return arena_; }
  const EpisodeConfig& episode() const { return ep_; }
  const std::vector<std::string>& pursuers() const { return pursuers_; }
  const std::vector<std::string>& evaders() const { return evaders_; }

 private:
  void init(const std::string& xml_path);
  void build_addr_cache();
  int update_captures();
  int update_spent();
  bool check_breach();
  bool check_arrived();
  void sync_asset();
  double nearest_dist() const;
  double nearest_attacker_asset_dist() const;
  std::vector<double> fuel_frac() const;
  void apply(const std::string& name, const double* ctrl);

  GameSpec game_;
  ArenaConfig arena_;
  EpisodeConfig ep_;
  mjModel_* model_ = nullptr;
  mjData_* data_ = nullptr;
  std::vector<std::string> pursuers_, evaders_;
  std::map<std::string, std::array<int, 3>> qadr_;
  std::map<std::string, std::array<int, 3>> act_ids_;
  std::vector<char> captured_;
  std::vector<char> spent_;
  std::vector<int> burn_left_;
  bool breached_ = false;
  bool asset_arrived_ = false;
  int steps_ = 0;
  std::mt19937_64 rng_;
  Starts last_starts_;
};

}  // namespace pe
