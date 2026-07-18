// Cross-device stamp-offset probe — the measuring half of the
// `<sensor>.stamp_offset` remedy (docs/SLAM_EFFECTIVENESS_AUDIT.md §2.2).
//
// Each sensor driver stamps from its own clock. A constant offset between two
// streams biases every sync pairing (v·δ, ω·δ pose error) or starves the
// synchronizer entirely past the slop — the no-match ERROR. This tool
// measures it instead of asking the operator to: subscribe to the streams
// being paired, record (local arrival time, header stamp) per message, and
// report per-stream median (stamp − arrival) plus the pairwise deltas and
// the exact config lines to set.
//
//   ros2 run sonar_slam_cpp stamp_probe --topics /oculus/sonar_image /dvl/data \
//        [--window 30]
//
// The FIRST topic is the reference; suggested offsets shift the others onto
// it. Notes:
//  - Works on any message whose FIRST field is std_msgs/Header (true for
//    every sensor stream this stack pairs); the stamp is parsed straight out
//    of the CDR buffer, so no type support is needed at build time.
//  - Arrival is a shared steady clock, so the measurement is valid live AND
//    under `ros2 bag play` at rate 1.0 (sim time does not matter).
//  - Sub-10 ms deltas are transport/serialization noise (a sonar image
//    arrives later than a DVL packet); the offsets that break pairing are
//    10s of ms and up.
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialized_message.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Stream
{
  std::string topic;
  rclcpp::GenericSubscription::SharedPtr sub;
  std::vector<std::pair<double, double>> samples;  // (arrival s, stamp s)
};

double steady_now()
{
  return std::chrono::duration<double>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

// header stamp out of a CDR buffer whose first field is std_msgs/Header:
// 4-byte encapsulation (byte1 bit0 = little-endian), int32 sec @4, uint32
// nanosec @8. Returns NaN when the buffer is too short.
double parse_header_stamp(const rcl_serialized_message_t& m)
{
  if (m.buffer_length < 12) return std::nan("");
  const uint8_t* b = m.buffer;
  const bool le = (b[1] & 0x01) != 0;
  auto rd32 = [&](std::size_t off) -> uint32_t {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i)
      v |= static_cast<uint32_t>(b[off + (le ? i : 3 - i)]) << (8 * i);
    return v;
  };
  const int32_t sec = static_cast<int32_t>(rd32(4));
  const uint32_t nsec = rd32(8);
  if (nsec >= 1000000000u) return std::nan("");  // not a header-first type
  return static_cast<double>(sec) + 1e-9 * static_cast<double>(nsec);
}

double median(std::vector<double> v)
{
  if (v.empty()) return std::nan("");
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  std::vector<std::string> topics;
  double window = 30.0;
  try {
    const std::vector<std::string> args =
      rclcpp::remove_ros_arguments(argc, argv);
    for (std::size_t i = 1; i < args.size(); ++i) {
      if (args[i] == "--topics") {
        while (i + 1 < args.size() && args[i + 1][0] == '/')
          topics.push_back(args[++i]);
      } else if (args[i] == "--window" && i + 1 < args.size()) {
        window = std::stod(args[++i]);
      } else {
        throw std::runtime_error("unknown argument " + args[i]);
      }
    }
    if (topics.size() < 2)
      throw std::runtime_error("need at least two --topics (first = reference)");
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "%s\nusage: stamp_probe --topics /ref /other [...] "
                 "[--window s]\n",
                 e.what());
    return 2;
  }

  auto node = std::make_shared<rclcpp::Node>("stamp_probe");
  std::vector<Stream> streams(topics.size());
  for (std::size_t i = 0; i < topics.size(); ++i) streams[i].topic = topics[i];

  // discover each topic's type, then attach a generic subscription
  auto try_subscribe = [&]() {
    const auto types = node->get_topic_names_and_types();
    for (auto& s : streams) {
      if (s.sub) continue;
      const auto it = types.find(s.topic);
      if (it == types.end() || it->second.empty()) continue;
      s.sub = node->create_generic_subscription(
        s.topic, it->second.front(), rclcpp::SensorDataQoS(),
        [&s](std::shared_ptr<rclcpp::SerializedMessage> msg) {
          const double stamp =
            parse_header_stamp(msg->get_rcl_serialized_message());
          if (std::isnan(stamp)) return;
          s.samples.emplace_back(steady_now(), stamp);
        });
      std::fprintf(stderr, "[stamp_probe] %s (%s)\n", s.topic.c_str(),
                   it->second.front().c_str());
    }
  };

  const double t0 = steady_now();
  auto timer = node->create_wall_timer(std::chrono::milliseconds(500), [&]() {
    try_subscribe();
    if (steady_now() - t0 > window) rclcpp::shutdown();
  });
  std::fprintf(stderr, "[stamp_probe] listening for %.0f s...\n", window);
  rclcpp::spin(node);

  // per-stream: median(stamp - arrival), spread, drift
  std::vector<double> med(streams.size(), std::nan(""));
  for (std::size_t i = 0; i < streams.size(); ++i) {
    const auto& sm = streams[i].samples;
    if (sm.size() < 5) {
      std::printf("%-40s  %zu msgs — too few to measure\n",
                  streams[i].topic.c_str(), sm.size());
      continue;
    }
    std::vector<double> d;
    d.reserve(sm.size());
    for (const auto& [arr, st] : sm) d.push_back(st - arr);
    med[i] = median(d);
    std::vector<double> ad;
    ad.reserve(d.size());
    for (double x : d) ad.push_back(std::fabs(x - med[i]));
    const double mad = median(ad);
    // drift: least-squares slope of (stamp - arrival) over arrival
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    const double n = static_cast<double>(d.size());
    for (std::size_t k = 0; k < d.size(); ++k) {
      const double x = sm[k].first - sm[0].first;
      sx += x;
      sy += d[k];
      sxx += x * x;
      sxy += x * d[k];
    }
    const double denom = n * sxx - sx * sx;
    const double slope = denom > 1e-9 ? (n * sxy - sx * sy) / denom : 0.0;
    std::printf("%-40s  %5zu msgs  stamp-arrival median %+.4f s  jitter "
                "(MAD) %.4f s  drift %+.2f ms/min\n",
                streams[i].topic.c_str(), sm.size(), med[i], mad,
                slope * 60.0 * 1e3);
    if (std::fabs(slope) * 60.0 > 5e-3)
      std::printf("  ^ WARNING: offset is DRIFTING — a constant stamp_offset "
                  "cannot fully fix this stream\n");
  }

  if (!std::isnan(med[0])) {
    std::printf("\nrelative to %s:\n", streams[0].topic.c_str());
    for (std::size_t i = 1; i < streams.size(); ++i) {
      if (std::isnan(med[i])) continue;
      const double delta = med[i] - med[0];
      std::printf("  %-38s  stamps %+.4f s vs reference", streams[i].topic.c_str(),
                  delta);
      if (std::fabs(delta) < 0.010) {
        std::printf("  — within transport noise, no offset needed\n");
      } else {
        std::printf("\n    set on the node(s) pairing these streams:  "
                    "<sensor>.stamp_offset: %.4f   # for the %s driver\n",
                    -delta, streams[i].topic.c_str());
      }
    }
  }
  return 0;
}
