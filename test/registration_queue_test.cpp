#include "sonar_slam_cpp/registration_queue.hpp"
#include <cstdio>

int main()
{
  using namespace sonar_slam;
  const auto ping = [](int i) {
    PendingPing value;
    value.time.sec = 100 + i;
    value.epoch = 7;
    value.points = Matrix::Constant(120, 3, static_cast<float>(i));
    return value;
  };
  RegistrationQueue reliable(1, true);
  for (int i = 0; i < 12; ++i) reliable.push(ping(i));
  auto stats = reliable.stats();
  if (stats.pending != 12 || stats.resident != 1 || stats.spilled != 11 || stats.dropped) return 1;
  PendingPing value;
  for (int i = 0; i < 12; ++i) {
    if (!reliable.pop(value) || value.time.sec != 100 + i || value.epoch != 7 ||
        value.points.rows() != 120 || !value.points.isConstant(static_cast<float>(i))) return 1;
  }
  if (reliable.pop(value)) return 1;
  reliable.push(ping(13));
  reliable.push(ping(14));
  reliable.clear();
  if (reliable.stats().pending || reliable.pop(value)) return 1;
  RegistrationQueue live(2, false);
  for (int i = 0; i < 12; ++i) live.push(ping(i));
  if (live.stats().dropped != 10 || !live.pop(value) || value.time.sec != 110 ||
      !live.pop(value) || value.time.sec != 111 || live.pop(value)) return 1;
  std::puts("PASS: replay FIFO/spill/clear, bounded live queue and explicit drops");
}
