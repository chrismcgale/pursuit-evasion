#pragma once
#include <string>

#include "pe/vec3.hpp"

// Exact mirror of src/pursuit_evasion/env/dynamics.py.
//
// Every agent is a drag-limited point mass with three slide joints; the airframe
// is the map from a controller *command* to actuator input:
//
//   holonomic (drone) — the command IS the thrust vector: hover, instant
//                       reversal, zero turn radius.
//   missile           — thrust is pinned to the body axis (== velocity) while the
//                       motor burns; the command only buys lateral authority, and
//                       only up to `lat_authority` of full thrust. Turn radius is
//                       therefore v^2 / lat_accel. After `burn_steps` the motor
//                       cuts out; below `v_stall` the round is spent.
//
// apply_dynamics() is pure. Any change here MUST be made in dynamics.py too, and
// re-validated with the parity check (see CLAUDE.md).
namespace pe {

enum class DynKind { Holonomic, Missile };

// Mirrors dynamics.py:DynamicsProfile
struct DynamicsProfile {
  std::string name = "drone_pursuer";
  DynKind kind = DynKind::Holonomic;
  double gear = 21.0;            // peak thrust (N); v_max = gear / damping
  double damping = 4.0;          // joint damping == aerodynamic drag
  double lat_authority = 1.0;    // missile: max lateral cmd as a fraction of thrust
  int burn_steps = 0;            // missile: powered control steps (0 => unlimited)
  double v_stall = 0.0;          // missile: spent below this speed after burnout
  double v_align = 0.5;          // missile: speed below which there is no body axis

  double vmax() const { return gear / damping; }
  double lat_accel() const { return lat_authority * gear; }
  double turn_radius(double speed) const {
    double a = lat_accel();
    return a <= 0.0 ? 1e30 : speed * speed / a;
  }
  double turn_radius() const { return turn_radius(vmax()); }
};

// ---------------------------------------------------------------- presets
// The tag game's original balance (gears 21/19, damping 4 -> v_max 5.25/4.75).
// Do not retune: the committed results depend on these exact numbers.
inline DynamicsProfile drone_pursuer() {
  return {"drone_pursuer", DynKind::Holonomic, 21.0, 4.0, 1.0, 0, 0.0, 0.5};
}
inline DynamicsProfile drone_evader() {
  return {"drone_evader", DynKind::Holonomic, 19.0, 4.0, 1.0, 0, 0.0, 0.5};
}
// Air-defence roles: the interceptor is agile but slow, the missile ~1.5x faster
// and unable to turn inside ~6 m at top speed. That asymmetry is the whole game.
inline DynamicsProfile drone_interceptor() {
  return {"drone_interceptor", DynKind::Holonomic, 30.0, 4.0, 1.0, 0, 0.0, 0.5};
}
inline DynamicsProfile missile_attacker() {
  return {"missile", DynKind::Missile, 44.0, 4.0, 0.45, 120, 2.0, 0.5};
}
inline DynamicsProfile missile_long_burn() {
  return {"missile_long", DynKind::Missile, 44.0, 4.0, 0.45, 320, 2.0, 0.5};
}

inline Vec3 clip_norm(Vec3 v, double lim = 1.0) {
  double n = norm(v);
  return n > lim ? v * (lim / n) : v;
}

// Mirrors dynamics.py:apply_dynamics. Returns a vector with norm <= 1 (actuators
// are ctrlrange="-1 1", scaled by gear). `burn_left` is ignored for drones.
inline Vec3 apply_dynamics(const DynamicsProfile& dyn, Vec3 vel, Vec3 cmd,
                           int burn_left) {
  Vec3 c = clip_norm(cmd);
  if (dyn.kind == DynKind::Holonomic) return c;

  const bool powered = (dyn.burn_steps <= 0) || (burn_left > 0);
  const double speed = norm(vel);
  if (speed < dyn.v_align) {
    // Not yet flying: the launcher points it wherever it is commanded.
    return powered ? c : Vec3{0, 0, 0};
  }
  const Vec3 fwd = vel / speed;
  Vec3 lat = c - fwd * dot(c, fwd);
  lat = clip_norm(lat, dyn.lat_authority);
  Vec3 out = (powered ? fwd : Vec3{0, 0, 0}) + lat;
  return clip_norm(out);
}

// Mirrors dynamics.py:is_spent — burned out and decayed below stall speed.
inline bool is_spent(const DynamicsProfile& dyn, Vec3 vel, int burn_left) {
  if (dyn.kind != DynKind::Missile || dyn.burn_steps <= 0 || dyn.v_stall <= 0.0)
    return false;
  return burn_left <= 0 && norm(vel) < dyn.v_stall;
}

}  // namespace pe
