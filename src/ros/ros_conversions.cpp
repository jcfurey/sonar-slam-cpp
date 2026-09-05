#include "sonar_slam_cpp/ros_conversions.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace sonar_slam {

gtsam::Pose3 r2g(const geometry_msgs::msg::Pose& pose)
{
  const auto& q = pose.orientation;
  const auto& p = pose.position;
  return gtsam::Pose3(gtsam::Rot3::Quaternion(q.w, q.x, q.y, q.z),
                      gtsam::Point3(p.x, p.y, p.z));
}

geometry_msgs::msg::Pose g2r(const gtsam::Pose3& pose)
{
  geometry_msgs::msg::Pose msg;
  msg.position.x = pose.x();
  msg.position.y = pose.y();
  msg.position.z = pose.z();
  const gtsam::Quaternion q = pose.rotation().toQuaternion();
  msg.orientation.x = q.x();
  msg.orientation.y = q.y();
  msg.orientation.z = q.z();
  msg.orientation.w = q.w();
  return msg;
}

geometry_msgs::msg::TransformStamped make_transform(
  const Eigen::Vector3d& translation, const Eigen::Quaterniond& rotation,
  const builtin_interfaces::msg::Time& stamp, const std::string& parent_frame,
  const std::string& child_frame)
{
  geometry_msgs::msg::TransformStamped t;
  t.header.stamp = stamp;
  t.header.frame_id = parent_frame;
  t.child_frame_id = child_frame;
  t.transform.translation.x = translation.x();
  t.transform.translation.y = translation.y();
  t.transform.translation.z = translation.z();
  t.transform.rotation.x = rotation.x();
  t.transform.rotation.y = rotation.y();
  t.transform.rotation.z = rotation.z();
  t.transform.rotation.w = rotation.w();
  return t;
}

sensor_msgs::msg::PointCloud2 make_cloud_xyz(const Matrix& points)
{
  return make_cloud({"x", "y", "z"}, points);
}

sensor_msgs::msg::PointCloud2 make_cloud(const std::vector<std::string>& fields,
                                         const Matrix& data)
{
  sensor_msgs::msg::PointCloud2 msg;
  msg.height = 1;
  msg.width = static_cast<uint32_t>(data.rows());
  msg.is_bigendian = false;
  msg.is_dense = false;

  msg.fields.reserve(fields.size());
  uint32_t offset = 0;
  for (const auto& name : fields) {
    sensor_msgs::msg::PointField f;
    f.name = name;
    f.offset = offset;
    f.datatype = sensor_msgs::msg::PointField::FLOAT32;
    f.count = 1;
    msg.fields.push_back(f);
    offset += 4;
  }
  msg.point_step = offset;
  msg.row_step = msg.point_step * msg.width;
  msg.data.resize(static_cast<std::size_t>(msg.row_step));

  // point_step is sized from fields.size(), so writing data.cols() floats
  // overruns the stride — and the buffer itself on the last point — whenever a
  // caller passes more columns than field names. Every in-tree caller matches
  // today (xyz/3, xyzi/4, traj/8), but this is a raw pointer write in a shared
  // helper, so bound it rather than rely on that staying true. Fewer columns
  // than fields is harmless: resize() zero-filled the remainder.
  const int ncols = std::min<int>(static_cast<int>(data.cols()),
                                  static_cast<int>(fields.size()));
  for (int i = 0; i < data.rows(); ++i) {
    float* dst = reinterpret_cast<float*>(msg.data.data() +
                                          static_cast<std::size_t>(i) * msg.point_step);
    for (int c = 0; c < ncols; ++c) dst[c] = data(i, c);
  }
  return msg;
}

Matrix cloud_to_xyz(const sensor_msgs::msg::PointCloud2& msg)
{
  // Reject inconsistent metadata before allocating or touching the payload.
  // A malformed cloud is an odometry-only frame, like an empty detection.
  if (msg.width == 0 || msg.height == 0 || msg.point_step < sizeof(float))
    return Matrix(0, 3);
  const std::uint64_t row_bytes = std::uint64_t(msg.width) * msg.point_step;
  const std::uint64_t payload_bytes = std::uint64_t(msg.height) * msg.row_step;
  const std::uint64_t n = std::uint64_t(msg.width) * msg.height;
  if (row_bytes > msg.row_step || payload_bytes > msg.data.size() ||
      n > std::uint64_t(std::numeric_limits<Eigen::Index>::max()) / 3)
    return Matrix(0, 3);

  std::array<std::uint32_t, 3> offsets{};
  const std::array<const char*, 3> names{{"x", "y", "z"}};
  for (std::size_t c = 0; c < names.size(); ++c) {
    bool found = false;
    for (const auto& f : msg.fields)
      if (f.name == names[c]) {
        if (found || f.datatype != sensor_msgs::msg::PointField::FLOAT32 ||
            f.count != 1 || f.offset > msg.point_step - sizeof(float))
          return Matrix(0, 3);
        offsets[c] = f.offset;
        found = true;
      }
    if (!found) return Matrix(0, 3);
  }

  Matrix out(static_cast<Eigen::Index>(n), 3);
  for (std::uint32_t r = 0; r < msg.height; ++r) {
    for (std::uint32_t p = 0; p < msg.width; ++p) {
      const auto* point = msg.data.data() + std::size_t(r) * msg.row_step +
                          std::size_t(p) * msg.point_step;
      for (std::size_t c = 0; c < offsets.size(); ++c) {
        const auto* bytes = point + offsets[c];
        std::uint32_t bits = 0;
        for (int b = 0; b < 4; ++b)
          bits |= std::uint32_t(bytes[b]) <<
                  (8 * (msg.is_bigendian ? 3 - b : b));
        float value;
        std::memcpy(&value, &bits, sizeof value);
        out(Eigen::Index(r) * msg.width + p, c) = value;
      }
    }
  }
  return out;
}

visualization_msgs::msg::Marker ros_constraints(
  const std::vector<ConstraintLink>& links)
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = "map";
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.ns = "constraints";
  marker.scale.x = 0.2;
  marker.color.g = 1.0;
  marker.color.a = 1.0;
  marker.pose.orientation.w = 1.0;  // valid identity quaternion (not 0,0,0,0)

  auto make_color = [](const std::string& name) {
    std_msgs::msg::ColorRGBA c;
    c.a = 1.0;
    if (name == "red") c.r = 1.0;
    else if (name == "green") c.g = 1.0;
    else if (name == "blue") c.b = 1.0;
    else { c.r = c.g = c.b = 1.0; }
    return c;
  };

  for (const auto& [pts, color] : links) {
    geometry_msgs::msg::Point p1, p2;
    p1.x = pts.first.x();
    p1.y = pts.first.y();
    p1.z = pts.first.z();
    p2.x = pts.second.x();
    p2.y = pts.second.y();
    p2.z = pts.second.z();
    marker.points.push_back(p1);
    marker.points.push_back(p2);
    marker.colors.push_back(make_color(color));
    marker.colors.push_back(make_color(color));
  }
  return marker;
}

}  // namespace sonar_slam
