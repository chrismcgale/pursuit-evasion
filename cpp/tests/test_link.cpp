// Unit tests for the link mirror (pe/link.hpp). Cross-runtime exactness is
// checked by `pe_run --parity --link <preset> --link-tape` in CI; these pin the
// local behaviour, mirroring tests/test_link.py.
#include <doctest/doctest.h>

#include <cstdio>
#include <fstream>

#include "pe/link.hpp"

using namespace pe;

namespace {
// Records the estimate it is handed and replies with a fixed command.
struct Probe : Controller {
  std::vector<TeamView> seen;
  std::vector<double> cmd = std::vector<double>(6, 0.5);
  std::vector<double> act(const TeamView& v) override {
    seen.push_back(v);
    return cmd;
  }
};
ArenaConfig ARENA;
TeamView view_at(double x) {
  TeamView v;
  v.arena = &ARENA;
  v.self_pos = {{x, 0, 3}, {x, 1, 3}};
  v.self_vel = {{1, 0, 0}, {1, 0, 0}};
  v.self_alive = {1, 1};
  v.opp_pos = {{5, 0, 3}, {6, 0, 3}};
  v.opp_vel = {{0, 0, 0}, {0, 0, 0}};
  v.opp_alive = {1, 1};
  return v;
}
}  // namespace

TEST_CASE("link: PERFECT is a bypass (same estimate, same command)") {
  Probe p;
  LinkedController L(&p, link_preset("perfect"), nullptr, 0.06);
  auto v = view_at(1.0);
  auto out = L.act(v);
  CHECK(out == p.cmd);
  CHECK(p.seen.back().self_pos[0].x == 1.0);
}

TEST_CASE("link: presets mirror link.py and only perfect is perfect") {
  CHECK(link_preset("perfect").is_perfect());
  for (auto n : {"vicon_lab", "vicon_busy", "field_degraded"}) CHECK_FALSE(link_preset(n).is_perfect());
  CHECK(link_preset("vicon_lab").obs_latency_s == 0.008);
  CHECK(link_preset("vicon_busy").swap_p == 0.004);
  CHECK(link_preset("field_degraded").thrust_scale == 0.9);
  CHECK_THROWS(link_preset("nope"));
}

TEST_CASE("link: sub-tick latency interpolates between the last two views") {
  LinkProfile prof;
  prof.name = "lat";
  prof.obs_latency_s = 0.03;   // half a tick at dt = 60 ms
  Probe p;
  LinkedController L(&p, prof, std::make_unique<NativeSource>(prof, 1), 0.06);
  L.act(view_at(0.0));
  L.act(view_at(1.0));
  CHECK(p.seen.back().self_pos[0].x == doctest::Approx(0.5));
}

TEST_CASE("link: a tape swap exchanges opponent labels, positions exact") {
  const char* path = "/tmp/pe_test_tape.txt";
  {
    std::ofstream f(path);
    f << "# pe-link-tape v1 profile=x seed=1 n_self=2 n_opp=2 n_ticks=2\n";
    // stream 0 (no reset after construction here), no noise, no drops
    f << "0 0";
    for (int i = 0; i < 24; ++i) f << " 0";
    f << " 0 0 0 0 0 1 0\n";
    f << "0 1";
    for (int i = 0; i < 24; ++i) f << " 0";
    f << " 0 0 0 0 -1 -1 1\n";
  }
  LinkProfile prof;
  prof.name = "x";
  prof.swap_p = 0.5;
  prof.packet_loss_p = 0.5;
  Probe p;
  LinkedController L(&p, prof, std::make_unique<TapeSource>(path, prof), 0.06);
  L.act(view_at(0.0));
  CHECK(p.seen.back().opp_pos[0].x == 6.0);
  CHECK(p.seen.back().opp_pos[1].x == 5.0);
  CHECK(L.n_swap == 1);
  p.cmd = std::vector<double>(6, -0.25);
  auto out = L.act(view_at(0.0));     // tick 1: swap over, packet lost
  CHECK(p.seen.back().opp_pos[0].x == 5.0);
  CHECK(out[0] == 0.5);               // FC holds the previous frame
  CHECK(L.n_lost_packet == 1);
  std::remove(path);
}

TEST_CASE("link: an exhausted tape is an error, not silent perfection") {
  const char* path = "/tmp/pe_test_tape_short.txt";
  {
    std::ofstream f(path);
    f << "# pe-link-tape v1 profile=x seed=1 n_self=2 n_opp=2 n_ticks=1\n0 0";
    for (int i = 0; i < 24; ++i) f << " 0";
    f << " 0 0 0 0 -1 -1 0\n";
  }
  LinkProfile prof;
  prof.name = "x";
  prof.dropout_p = 0.1;
  Probe p;
  LinkedController L(&p, prof, std::make_unique<TapeSource>(path, prof), 0.06);
  L.act(view_at(0.0));
  CHECK_THROWS(L.act(view_at(0.0)));
  std::remove(path);
}

TEST_CASE("link: history is sized by the latency, not capped at 64 (REVIEW #14)") {
  LinkProfile prof;
  prof.name = "long";
  prof.obs_latency_s = 0.06 * 80;   // 80 ticks
  Probe p;
  LinkedController L(&p, prof, std::make_unique<NativeSource>(prof, 1), 0.06);
  for (int k = 0; k <= 90; ++k) L.act(view_at((double)k));
  CHECK(p.seen.back().self_pos[0].x == doctest::Approx(10.0));   // 90 - 80
}
