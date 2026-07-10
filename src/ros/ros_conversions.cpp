#include "sonar_slam_cpp/ros_conversions.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <cstring>

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

  for (int i = 0; i < data.rows(); ++i) {
    float* dst = reinterpret_cast<float*>(msg.data.data() +
                                          static_cast<std::size_t>(i) * msg.point_step);
    for (int c = 0; c < data.cols(); ++c) dst[c] = data(i, c);
  }
  return msg;
}

Matrix cloud_to_xyz(const sensor_msgs::msg::PointCloud2& msg)
{
  const long n = static_cast<long>(msg.width) * msg.height;
  Matrix out(n, 3);
  sensor_msgs::PointCloud2ConstIterator<float> it_x(msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> it_y(msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> it_z(msg, "z");
  for (long i = 0; i < n; ++i, ++it_x, ++it_y, ++it_z) {
    out(i, 0) = *it_x;
    out(i, 1) = *it_y;
    out(i, 2) = *it_z;
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
