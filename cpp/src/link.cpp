#include "pe/link.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pe {

LinkProfile link_preset(const std::string& name) {
  LinkProfile p;
  if (name == "perfect") return p;
  // VICON_LAB
  p.name = "vicon_lab";
  p.obs_latency_s = 0.008; p.pos_noise_m = 0.0005; p.vel_noise_mps = 0.03;
  p.dropout_p = 0.005; p.dropout_burst = 3.0; p.swap_p = 0.0005; p.swap_burst = 15.0;
  p.cmd_latency_s = 0.006; p.motor_tau_s = 0.020; p.packet_loss_p = 0.005;
  p.thrust_scale = 1.0; p.sag_frac_per_min = 0.06;
  if (name == "vicon_lab") return p;
  if (name == "vicon_busy") {   // replace(VICON_LAB, ...)
    p.name = "vicon_busy";
    p.obs_latency_s = 0.015; p.vel_noise_mps = 0.06; p.dropout_p = 0.02;
    p.dropout_burst = 5.0; p.swap_p = 0.004; p.swap_burst = 25.0;
    p.packet_loss_p = 0.02; p.sag_frac_per_min = 0.12;
    return p;
  }
  if (name == "field_degraded") {
    LinkProfile f;
    f.name = "field_degraded";
    f.obs_latency_s = 0.040; f.pos_noise_m = 0.15; f.vel_noise_mps = 0.25;
    f.dropout_p = 0.05; f.dropout_burst = 8.0; f.swap_p = 0.002; f.swap_burst = 20.0;
    f.cmd_latency_s = 0.020; f.motor_tau_s = 0.030; f.packet_loss_p = 0.05;
    f.thrust_scale = 0.9; f.sag_frac_per_min = 0.15;
    return f;
  }
  throw std::runtime_error("unknown link preset: " + name);
}

// ------------------------------------------------------------------ tape
TapeSource::TapeSource(const std::string& path, const LinkProfile& p) : p_(p) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open link tape " + path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') {
      // header carries n_self / n_opp
      auto get = [&](const std::string& k, int def) {
        auto pos = line.find(k + "=");
        return pos == std::string::npos ? def : std::stoi(line.substr(pos + k.size() + 1));
      };
      n_self_ = get("n_self", n_self_);
      n_opp_ = get("n_opp", n_opp_);
      continue;
    }
    std::istringstream ss(line);
    int k, t;
    ss >> k >> t;
    TickDraws d;
    auto vecs = [&](std::vector<Vec3>& v, int n, bool keep) {
      std::vector<Vec3> tmp(n);
      for (int i = 0; i < n; ++i) {
        std::string a, b, c;
        ss >> a >> b >> c;
        tmp[i] = {std::strtod(a.c_str(), nullptr), std::strtod(b.c_str(), nullptr),
                  std::strtod(c.c_str(), nullptr)};
      }
      if (keep) v = tmp;
    };
    vecs(d.pos_self, n_self_, p_.pos_noise_m > 0);
    vecs(d.pos_opp, n_opp_, p_.pos_noise_m > 0);
    vecs(d.vel_self, n_self_, p_.vel_noise_mps > 0);
    vecs(d.vel_opp, n_opp_, p_.vel_noise_mps > 0);
    d.drop_self.resize(n_self_);
    d.drop_opp.resize(n_opp_);
    for (auto& x : d.drop_self) { int b; ss >> b; x = (char)b; }
    for (auto& x : d.drop_opp) { int b; ss >> b; x = (char)b; }
    int lost;
    ss >> d.swap_i >> d.swap_j >> lost;
    d.lost_packet = lost != 0;
    if (!ss) throw std::runtime_error("malformed link tape line: " + line);
    auto& s = streams_[k];
    if ((int)s.size() != t) throw std::runtime_error("link tape ticks out of order");
    s.push_back(std::move(d));
  }
}

void TapeSource::begin_stream(int stream, int n_self, int n_opp) {
  stream_ = stream;
  t_ = 0;
  auto it = streams_.find(stream);
  cur_ = it == streams_.end() ? nullptr : &it->second;
  if (n_self != n_self_ || n_opp != n_opp_)
    throw std::runtime_error("link tape team sizes do not match the view");
}

TickDraws TapeSource::next() {
  if (!cur_ || t_ >= cur_->size())
    throw std::runtime_error("link tape exhausted at stream " + std::to_string(stream_) +
                             " tick " + std::to_string(t_));
  return (*cur_)[t_++];
}

// ---------------------------------------------------------------- native
void NativeSource::begin_stream(int stream, int n_self, int n_opp) {
  std::seed_seq seq{(uint32_t)(seed_ & 0xffffffffu), (uint32_t)(seed_ >> 32), (uint32_t)stream};
  rng_.seed(seq);
  n_self_ = n_self;
  n_opp_ = n_opp;
  t_ = 0;
  auto mk = [](double p_on, double mean_len) {
    return Burst{p_on, mean_len <= 1.0 ? 0.0 : 1.0 - 1.0 / mean_len, false};
  };
  self_drop_.assign(n_self, mk(p_.dropout_p, p_.dropout_burst));
  opp_drop_.assign(n_opp, mk(p_.dropout_p, p_.dropout_burst));
  swap_ = mk(p_.swap_p, p_.swap_burst);
  si_ = sj_ = -1;
}

bool NativeSource::tick(Burst& b) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  b.on = b.on ? u(rng_) < b.p_stay : u(rng_) < b.p_on;
  return b.on;
}

TickDraws NativeSource::next() {
  TickDraws d;
  auto noise = [&](std::vector<Vec3>& v, int n, double sigma) {
    if (sigma <= 0) return;
    std::normal_distribution<double> g(0.0, sigma);
    v.resize(n);
    for (auto& x : v) x = {g(rng_), g(rng_), g(rng_)};
  };
  noise(d.pos_self, n_self_, p_.pos_noise_m);
  noise(d.pos_opp, n_opp_, p_.pos_noise_m);
  noise(d.vel_self, n_self_, p_.vel_noise_mps);
  noise(d.vel_opp, n_opp_, p_.vel_noise_mps);
  for (auto& b : self_drop_) d.drop_self.push_back(tick(b));
  for (auto& b : opp_drop_) d.drop_opp.push_back(tick(b));
  if (n_opp_ >= 2) {
    const bool was_on = swap_.on;
    if (tick(swap_)) {
      if (!was_on) {
        std::uniform_int_distribution<int> a(0, n_opp_ - 1), b(0, n_opp_ - 2);
        si_ = a(rng_);
        sj_ = b(rng_);
        if (sj_ >= si_) ++sj_;
      }
      d.swap_i = si_;
      d.swap_j = sj_;
    } else {
      si_ = sj_ = -1;
    }
  }
  std::uniform_real_distribution<double> u(0.0, 1.0);
  d.lost_packet = p_.packet_loss_p > 0 && t_ > 0 && u(rng_) < p_.packet_loss_p;
  ++t_;
  return d;
}

// ------------------------------------------------------------ controller
LinkedController::LinkedController(Controller* in, LinkProfile prof,
                                   std::unique_ptr<DrawSource> src, double dt)
    : inner(in), profile(std::move(prof)), src_(std::move(src)), dt_(dt) {
  name = in->name + "@" + profile.name;
  if (!profile.is_perfect() && !src_)
    throw std::runtime_error("a degraded link needs a DrawSource");
  reset();
}

void LinkedController::reset() {
  inner->reset();
  ++ep_;
  views_.clear();
  cmds_.clear();
  has_held_ = has_last_cmd_ = has_thrust_ = prev_swap_ = false;
  n_dropout = n_swap = n_lost_packet = ticks = 0;
  // link.py: ceil(latency ticks) + 1 views, round(cmd latency ticks) + 1 commands
  hist_ = (size_t)std::max(2, (int)std::ceil(profile.obs_latency_s / dt_) + 1);
  cmd_hist_ = profile.cmd_latency_s > 0
                  ? (size_t)std::nearbyint(profile.cmd_latency_s / dt_) + 1 : 1;
  started_stream_ = false;
}

TeamView LinkedController::delayed(const TeamView& view) const {
  const double d = profile.obs_latency_s / dt_;
  if (d <= 0) return view;
  const int n = (int)views_.size();
  const int lo = std::min((int)std::floor(d), n - 1);
  const int hi = std::min((int)std::ceil(d), n - 1);
  const double frac = d - std::floor(d);
  const TeamView& a = views_[n - 1 - lo];
  const TeamView& b = views_[n - 1 - hi];
  if (lo == hi) return a;
  const double w = 1.0 - frac;
  auto mix = [&](Vec3 x, Vec3 y) { return w * x + (1.0 - w) * y; };
  TeamView out = a;   // discrete fields (alive, fuel, time) from the older sample
  for (size_t i = 0; i < out.self_pos.size(); ++i) {
    out.self_pos[i] = mix(a.self_pos[i], b.self_pos[i]);
    out.self_vel[i] = mix(a.self_vel[i], b.self_vel[i]);
  }
  for (size_t j = 0; j < out.opp_pos.size(); ++j) {
    out.opp_pos[j] = mix(a.opp_pos[j], b.opp_pos[j]);
    out.opp_vel[j] = mix(a.opp_vel[j], b.opp_vel[j]);
  }
  if (a.has_asset) {
    out.asset_pos = mix(a.asset_pos, b.asset_pos);
    out.asset_vel = mix(a.asset_vel, b.asset_vel);
  }
  return out;
}

TeamView LinkedController::corrupt(const TeamView& view) {
  const int ns = view.n_self(), no = (int)view.opp_pos.size();
  if (!started_stream_) {   // link.py builds its stream on the first view
    src_->begin_stream(ep_, ns, no);
    started_stream_ = true;
  }
  draws_ = src_->next();
  const TickDraws& d = draws_;
  TeamView e = view;
  if (!d.pos_self.empty()) {
    for (int i = 0; i < ns; ++i) e.self_pos[i] = e.self_pos[i] + d.pos_self[i];
    for (int j = 0; j < no; ++j) e.opp_pos[j] = e.opp_pos[j] + d.pos_opp[j];
  }
  if (!d.vel_self.empty()) {
    for (int i = 0; i < ns; ++i) e.self_vel[i] = e.self_vel[i] + d.vel_self[i];
    for (int j = 0; j < no; ++j) e.opp_vel[j] = e.opp_vel[j] + d.vel_opp[j];
  }
  if (has_held_) {
    for (int i = 0; i < ns; ++i)
      if (d.drop_self[i]) { e.self_pos[i] = held_sp_[i]; e.self_vel[i] = held_sv_[i]; ++n_dropout; }
    for (int j = 0; j < no; ++j)
      if (d.drop_opp[j]) { e.opp_pos[j] = held_op_[j]; e.opp_vel[j] = held_ov_[j]; ++n_dropout; }
  }
  if (d.swap_i >= 0) {
    if (!prev_swap_) ++n_swap;
    std::swap(e.opp_pos[d.swap_i], e.opp_pos[d.swap_j]);
    std::swap(e.opp_vel[d.swap_i], e.opp_vel[d.swap_j]);
  }
  prev_swap_ = d.swap_i >= 0;
  held_sp_ = e.self_pos; held_sv_ = e.self_vel; held_op_ = e.opp_pos; held_ov_ = e.opp_vel;
  has_held_ = true;
  return e;
}

std::vector<double> LinkedController::uplink(const std::vector<double>& cmd, bool lost) {
  cmds_.push_back(cmd);
  if (cmds_.size() > cmd_hist_) cmds_.erase(cmds_.begin());
  std::vector<double> sent;
  if (profile.cmd_latency_s > 0) {
    // Python round(): half to even, which is nearbyint's default mode
    const long k = (long)std::nearbyint(profile.cmd_latency_s / dt_);
    const long idx = std::max(0L, (long)cmds_.size() - 1 - k);
    sent = cmds_[idx];
  } else {
    sent = cmds_.back();
  }
  if (lost && has_last_cmd_) {
    ++n_lost_packet;
    sent = last_cmd_;
  }
  last_cmd_ = sent;
  has_last_cmd_ = true;
  std::vector<double> out = sent;
  if (profile.motor_tau_s > 0) {
    const double alpha = 1.0 - std::exp(-dt_ / profile.motor_tau_s);
    if (!has_thrust_) {
      thrust_.assign(sent.size(), 0.0);
      has_thrust_ = true;
    }
    for (size_t k = 0; k < sent.size(); ++k) thrust_[k] = thrust_[k] + alpha * (sent[k] - thrust_[k]);
    out = thrust_;
  }
  double scale = profile.thrust_scale;
  if (profile.sag_frac_per_min > 0)
    scale *= std::max(0.0, 1.0 - profile.sag_frac_per_min * (ticks * dt_) / 60.0);
  for (double& x : out) {
    x = x * scale;
    x = x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x);
  }
  return out;
}

std::vector<double> LinkedController::act(const TeamView& view) {
  if (profile.is_perfect()) return inner->act(view);   // bypass: bit-identical
  views_.push_back(view);
  if (views_.size() > hist_) views_.erase(views_.begin());
  TeamView est = corrupt(delayed(view));
  std::vector<double> cmd = inner->act(est);
  ++ticks;
  return uplink(cmd, draws_.lost_packet);
}

}  // namespace pe
