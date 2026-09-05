#pragma once
#include <cmath>

namespace pe {

// Minimal 3-vector so the C++ runtime has no linear-algebra dependency and the
// math reads the same as the numpy reference.
struct Vec3 {
  double x{0}, y{0}, z{0};
  Vec3() = default;
  Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
  double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
  double& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(double s, Vec3 a) { return a * s; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double norm(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Vec3 unit(Vec3 v, double eps = 1e-8) {
  double n = norm(v);
  return n > eps ? v * (1.0 / n) : Vec3{0, 0, 0};
}
inline double clamp(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace pe
