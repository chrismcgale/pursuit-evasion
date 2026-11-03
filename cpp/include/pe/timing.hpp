#pragma once
// Per-stage wall-clock timing for the ground-station question (ROADMAP #6/#18):
// how long does one control tick of BT + policy + shield take per aircraft?
// Collection only; nothing here changes what any controller computes.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace pe {

using Clock = std::chrono::steady_clock;

inline double us_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

struct TimingSink {
  std::map<std::string, std::vector<double>> us;  // stage -> samples (microseconds)
  void add(const std::string& stage, double v) { us[stage].push_back(v); }

  static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = q * (v.size() - 1);
    const size_t lo = (size_t)std::floor(idx), hi = (size_t)std::ceil(idx);
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
  }

  // One row per stage: n, mean, p50, p90, p99, p99.9, max (us).
  void print_table(const char* title) const {
    std::printf("\n%s\n", title);
    std::printf("| stage | n | mean us | p50 | p90 | p99 | p99.9 | max |\n");
    std::printf("|---|---|---|---|---|---|---|---|\n");
    for (const auto& [k, v] : us) {
      double m = 0;
      for (double x : v) m += x;
      m /= std::max<size_t>(v.size(), 1);
      std::printf("| %s | %zu | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |\n", k.c_str(),
                  v.size(), m, pct(v, 0.5), pct(v, 0.9), pct(v, 0.99), pct(v, 0.999),
                  v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()));
    }
  }

  // Log2-bucketed histogram (bucket b holds [2^b, 2^(b+1)) us), CSV rows
  // stage,bucket_lo_us,count — enough to plot without shipping raw samples.
  void write_histogram_csv(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "stage,bucket_lo_us,count\n");
    for (const auto& [k, v] : us) {
      std::map<int, long> h;
      for (double x : v) h[x < 1.0 ? -1 : (int)std::floor(std::log2(x))] += 1;
      for (auto [b, c] : h) std::fprintf(f, "%s,%g,%ld\n", k.c_str(), b < 0 ? 0.0 : std::ldexp(1.0, b), c);
    }
    std::fclose(f);
  }
};

}  // namespace pe
