#include "pe/sim.hpp"

#include <mujoco/mujoco.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace pe {

Sim::Sim(const std::string& xml_path, GameSpec game, uint64_t seed)
    : game_(std::move(game)), rng_(seed) {
  arena_ = game_.arena;
  ep_ = game_.episode;
  init(xml_path);
}

// Back-compat overload: explicit arena/episode on the tag game (sweeps, tests).
Sim::Sim(const std::string& xml_path, ArenaConfig arena, EpisodeConfig ep, uint64_t seed)
    : game_(game_tag()), arena_(arena), ep_(ep), rng_(seed) {
  game_.arena = arena;
  game_.episode = ep;
  init(xml_path);
}

void Sim::init(const std::string& xml_path) {
  pursuers_ = pursuer_names(arena_);
  evaders_ = evader_names(arena_);
  char err[1024] = {0};
  model_ = mj_loadXML(xml_path.c_str(), nullptr, err, sizeof(err));
  if (!model_) throw std::runtime_error(std::string("mj_loadXML failed: ") + err);
  data_ = mj_makeData(model_);
  build_addr_cache();
  captured_.assign(evaders_.size(), 0);
  spent_.assign(evaders_.size(), 0);
  burn_left_.assign(evaders_.size(), arena_.evader_dyn.burn_steps);
}

Sim::~Sim() {
  if (data_) mj_deleteData(data_);
  if (model_) mj_deleteModel(model_);
}

void Sim::build_addr_cache() {
  const char* ax[3] = {"x", "y", "z"};
  auto cache = [&](const std::string& name) {
    std::array<int, 3> q{}, a{};
    for (int k = 0; k < 3; ++k) {
      std::string jn = name + "_" + ax[k];
      int jid = mj_name2id(model_, mjOBJ_JOINT, jn.c_str());
      int aid = mj_name2id(model_, mjOBJ_ACTUATOR, jn.c_str());
      if (jid < 0 || aid < 0)
        throw std::runtime_error("missing joint/actuator: " + jn);
      q[k] = model_->jnt_qposadr[jid];
      a[k] = aid;
    }
    qadr_[name] = q;
    act_ids_[name] = a;
  };
  for (auto& n : pursuers_) cache(n);
  for (auto& n : evaders_) cache(n);
}

Vec3 Sim::pos(const std::string& name) const {
  const auto& q = qadr_.at(name);
  return {data_->qpos[q[0]], data_->qpos[q[1]], data_->qpos[q[2]]};
}
Vec3 Sim::vel(const std::string& name) const {
  const auto& q = qadr_.at(name);
  return {data_->qvel[q[0]], data_->qvel[q[1]], data_->qvel[q[2]]};
}

void Sim::apply(const std::string& name, const double* ctrl) {
  const auto& a = act_ids_.at(name);
  for (int k = 0; k < 3; ++k) data_->ctrl[a[k]] = clamp(ctrl[k], -1.0, 1.0);
}

// --------------------------------------------------------------------- asset

Vec3 Sim::asset_position() const {
  return game_.has_asset ? game_.asset.position_at(steps_ * dt()) : Vec3{0, 0, 0};
}
Vec3 Sim::asset_velocity() const {
  return game_.has_asset ? game_.asset.velocity_at(steps_ * dt()) : Vec3{0, 0, 0};
}

// The asset is a mocap body: kinematic, no dynamics, pose written directly.
void Sim::sync_asset() {
  if (!game_.has_asset || model_->nmocap <= 0) return;
  Vec3 p = asset_position();
  data_->mocap_pos[0] = p.x;
  data_->mocap_pos[1] = p.y;
  data_->mocap_pos[2] = p.z;
}

// -------------------------------------------------------------------- spawns
// NOTE: this mirrors games.py:sample_starts *geometrically*, but numpy's
// Generator and std::mt19937_64 are different streams, so the sampled values
// differ. Exact cross-runtime episodes come from --starts-file, not from here.

Starts Sim::sample_starts() {
  Starts s;
  std::uniform_real_distribution<double> zd(ep_.spawn_z_lo, ep_.spawn_z_hi);
  std::uniform_real_distribution<double> rad(ep_.spawn_radius * 0.6, ep_.spawn_radius);

  if (game_.spawn_kind == "opposed") {
    std::uniform_real_distribution<double> ang(-0.6, 0.6);
    for (auto& n : pursuers_) {
      double a = ang(rng_), r = rad(rng_), z = zd(rng_);
      s[n] = {-r * std::cos(a), r * std::sin(a), z};
    }
    for (auto& n : evaders_) {
      double a = ang(rng_), r = rad(rng_), z = zd(rng_);
      s[n] = {r * std::cos(a), r * std::sin(a), z};
    }
    return s;
  }

  // objective games: attackers run in from long range on separated bearings,
  // defenders hold a combat air patrol near the thing they are protecting.
  const Vec3 asset = game_.asset.start;
  const int n_att = std::max<int>((int)evaders_.size(), 1);
  const bool ambush = (game_.spawn_kind == "ambush");
  double base = 0.0, arc = 0.0;
  if (ambush) {
    // escort: lie in wait downrange, inside a forward arc about the convoy's
    // transit heading, so it is a head-on merge and not an uncatchable tail chase
    Vec3 leg = game_.asset.goal - game_.asset.start;
    base = std::atan2(leg.y, leg.x);
    arc = 70.0 * M_PI / 180.0;
  } else {
    std::uniform_real_distribution<double> any(-M_PI, M_PI);
    base = any(rng_);   // threats from any bearing
  }
  std::uniform_real_distribution<double> jitter(-0.35, 0.35);
  std::uniform_real_distribution<double> arad(ep_.spawn_radius * 0.85, ep_.spawn_radius);
  for (int i = 0; i < (int)evaders_.size(); ++i) {
    double off;
    if (!ambush) {
      off = 2.0 * M_PI * i / n_att;
    } else {
      off = n_att > 1 ? arc * (2.0 * i / (n_att - 1) - 1.0) : 0.0;
    }
    double a = base + off + jitter(rng_), r = arad(rng_), z = zd(rng_);
    s[evaders_[i]] = {asset.x + r * std::cos(a), asset.y + r * std::sin(a), z};
  }
  std::uniform_real_distribution<double> pang(-M_PI, M_PI);
  std::uniform_real_distribution<double> prad(5.0, 9.0);
  std::uniform_real_distribution<double> pz(2.0, 8.0);
  for (auto& n : pursuers_) {
    double a = pang(rng_), r = prad(rng_), z = pz(rng_);
    s[n] = {asset.x + r * std::cos(a), asset.y + r * std::sin(a), z};
  }

  // keep everything inside the joint ranges
  const double L = arena_.half_extent, zlo = arena_.z_min, zhi = arena_.z_max;
  for (auto& [k, v] : s) {
    v.x = clamp(v.x, -L + 1.0, L - 1.0);
    v.y = clamp(v.y, -L + 1.0, L - 1.0);
    v.z = clamp(v.z, zlo + 0.5, zhi - 0.5);
  }
  return s;
}

void Sim::reset(uint64_t seed) {
  rng_.seed(seed);
  reset_with_starts(sample_starts());
}

void Sim::reset_with_starts(const Starts& s) {
  mj_resetData(model_, data_);
  for (auto& [name, xyz] : s) {
    const auto& q = qadr_.at(name);
    for (int k = 0; k < 3; ++k) data_->qpos[q[k]] = xyz[k];
  }
  for (int i = 0; i < model_->nv; ++i) data_->qvel[i] = 0.0;
  steps_ = 0;
  captured_.assign(evaders_.size(), 0);
  spent_.assign(evaders_.size(), 0);
  burn_left_.assign(evaders_.size(), arena_.evader_dyn.burn_steps);
  breached_ = false;
  asset_arrived_ = false;
  sync_asset();
  mj_forward(model_, data_);
  last_starts_ = s;
}

// ----------------------------------------------------------------- mechanics

int Sim::update_captures() {
  int newly = 0;
  std::vector<Vec3> pp;
  for (auto& n : pursuers_) pp.push_back(pos(n));
  for (size_t j = 0; j < evaders_.size(); ++j) {
    if (captured_[j]) continue;
    Vec3 ep = pos(evaders_[j]);
    double dmin = 1e30;
    for (auto& p : pp) dmin = std::min(dmin, norm(p - ep));
    if (dmin < ep_.capture_radius) {
      captured_[j] = 1;
      ++newly;
    }
  }
  return newly;
}

// A burned-out missile that has decayed below stall speed is out of the fight.
int Sim::update_spent() {
  int newly = 0;
  for (size_t j = 0; j < evaders_.size(); ++j) {
    if (captured_[j]) continue;
    if (is_spent(arena_.evader_dyn, vel(evaders_[j]), burn_left_[j])) {
      captured_[j] = 1;
      spent_[j] = 1;
      ++newly;
    }
  }
  return newly;
}

// Any live attacker inside the asset's breach radius ends it (attacker win).
bool Sim::check_breach() {
  if (!game_.has_asset || breached_) return breached_;
  const Vec3 a = asset_position();
  for (size_t j = 0; j < evaders_.size(); ++j) {
    if (captured_[j]) continue;
    if (norm(pos(evaders_[j]) - a) < game_.asset.radius) {
      breached_ = true;
      break;
    }
  }
  return breached_;
}

bool Sim::check_arrived() {
  if (!game_.has_asset || breached_) return false;
  asset_arrived_ = game_.asset.arrived_at(steps_ * dt());
  return asset_arrived_;
}

double Sim::nearest_dist() const {
  std::vector<Vec3> pp;
  for (auto& n : pursuers_) pp.push_back(pos(n));
  double dmin = 1e30;
  bool any = false;
  for (size_t j = 0; j < evaders_.size(); ++j) {
    if (captured_[j]) continue;
    any = true;
    Vec3 ep = pos(evaders_[j]);
    for (auto& p : pp) dmin = std::min(dmin, norm(p - ep));
  }
  return any ? dmin : 0.0;
}

double Sim::nearest_attacker_asset_dist() const {
  if (!game_.has_asset) return std::numeric_limits<double>::quiet_NaN();
  const Vec3 a = asset_position();
  double dmin = std::numeric_limits<double>::infinity();
  for (size_t j = 0; j < evaders_.size(); ++j) {
    if (captured_[j]) continue;
    dmin = std::min(dmin, norm(pos(evaders_[j]) - a));
  }
  return dmin;
}

std::vector<double> Sim::fuel_frac() const {
  const int n = arena_.evader_dyn.burn_steps;
  std::vector<double> f(evaders_.size(), 1.0);
  if (n <= 0) return f;
  for (size_t j = 0; j < evaders_.size(); ++j)
    f[j] = clamp((double)burn_left_[j] / (double)n, 0.0, 1.0);
  return f;
}

// ---------------------------------------------------------------------- step

StepInfo Sim::step(const std::vector<double>& pa, const std::vector<double>& ea) {
  // Airframe transform: a controller command is a *desire*; what the vehicle can
  // actually push is decided by its DynamicsProfile.
  for (size_t i = 0; i < pursuers_.size(); ++i) {
    Vec3 c{pa[i * 3], pa[i * 3 + 1], pa[i * 3 + 2]};
    Vec3 u = apply_dynamics(arena_.pursuer_dyn, vel(pursuers_[i]), c, 1);
    double ctrl[3] = {u.x, u.y, u.z};
    apply(pursuers_[i], ctrl);
  }
  for (size_t i = 0; i < evaders_.size(); ++i) {
    double ctrl[3] = {0, 0, 0};
    if (!captured_[i]) {   // neutralised: freeze
      Vec3 c{ea[i * 3], ea[i * 3 + 1], ea[i * 3 + 2]};
      Vec3 u = apply_dynamics(arena_.evader_dyn, vel(evaders_[i]), c, burn_left_[i]);
      ctrl[0] = u.x;
      ctrl[1] = u.y;
      ctrl[2] = u.z;
    }
    apply(evaders_[i], ctrl);
  }

  for (int k = 0; k < ep_.control_repeat; ++k) {
    mj_step(model_, data_);
    // A defender is SLOWER than a missile, so it can never convert to a stern
    // chase — nearly every engagement is a high-offset crossing pass, and the
    // chord cut through the capture sphere is far shorter than its diameter.
    // Those grazing chords fit between two consecutive control-step samples.
    //
    // Note it is NOT simply "missiles are fast": per control step tag closes
    // 1.43 capture radii and assault only 0.79. Tag stays once-per-step because
    // its 40-50% scripted baseline is calibrated on that (CLAUDE.md).
    if (game_.capture_substeps) update_captures();
  }

  ++steps_;
  for (auto& b : burn_left_) b = std::max(b - 1, -1);
  sync_asset();

  if (!game_.capture_substeps) update_captures();
  update_spent();

  const bool breach = check_breach();
  const bool arrived = check_arrived();

  StepInfo info;
  info.captured = captured_;
  info.n_captured = 0;
  for (char c : captured_) info.n_captured += c ? 1 : 0;
  info.n_spent = 0;
  for (char c : spent_) info.n_spent += c ? 1 : 0;
  info.min_dist = nearest_dist();
  info.all_captured = (info.n_captured == (int)evaders_.size());
  info.timeout = steps_ >= ep_.max_steps;
  info.steps = steps_;
  info.terminated = info.all_captured || breach || arrived;
  info.truncated = info.timeout && !info.terminated;
  info.breach = breach;
  info.asset_arrived = arrived;
  info.asset_dist = nearest_attacker_asset_dist();
  info.pursuer_win = info.all_captured || arrived ||
                     (info.truncated && game_.timeout_winner == Team::Pursuers);
  return info;
}

// ----------------------------------------------------------------- team view

TeamView Sim::view(Team team) const {
  TeamView v;
  v.team = team;
  v.arena = &arena_;
  v.game = game_.key;
  const std::vector<double> fuel = fuel_frac();
  const std::vector<std::string>*self, *opp;
  if (team == Team::Pursuers) {
    self = &pursuers_;
    opp = &evaders_;
    v.vmax = arena_.pursuer_vmax();
    v.opp_vmax = arena_.evader_vmax();
    v.self_dyn = &arena_.pursuer_dyn;
    v.opp_dyn = &arena_.evader_dyn;
    v.self_fuel.assign(pursuers_.size(), 1.0);
    v.opp_fuel = fuel;
  } else {
    self = &evaders_;
    opp = &pursuers_;
    v.vmax = arena_.evader_vmax();
    v.opp_vmax = arena_.pursuer_vmax();
    v.self_dyn = &arena_.evader_dyn;
    v.opp_dyn = &arena_.pursuer_dyn;
    v.self_fuel = fuel;
    v.opp_fuel.assign(pursuers_.size(), 1.0);
  }
  for (size_t i = 0; i < self->size(); ++i) {
    v.self_pos.push_back(pos((*self)[i]));
    v.self_vel.push_back(vel((*self)[i]));
    // pursuers always alive; evaders alive iff not captured
    v.self_alive.push_back(team == Team::Pursuers ? 1 : !captured_[i]);
  }
  for (size_t j = 0; j < opp->size(); ++j) {
    v.opp_pos.push_back(pos((*opp)[j]));
    v.opp_vel.push_back(vel((*opp)[j]));
    v.opp_alive.push_back(team == Team::Pursuers ? !captured_[j] : 1);
  }
  v.time_frac = (double)steps_ / ep_.max_steps;
  v.has_asset = game_.has_asset;
  if (game_.has_asset) {
    v.asset_pos = asset_position();
    v.asset_vel = asset_velocity();
    v.asset_radius = game_.asset.radius;
  }
  return v;
}

}  // namespace pe
