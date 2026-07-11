// Common node base: port of bruce_slam utils/io.py BruceNode. get_param
// accepts ROS 1 style "a/b" names (translated to "a.b"), auto-declares
// missing parameters with the supplied default, and converts numerics
// leniently (a YAML `65` satisfies a double parameter).
#pragma once

#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace sonar_slam {

class SlamNodeBase : public rclcpp::Node
{
public:
  explicit SlamNodeBase(const std::string& node_name,
                        const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
    : rclcpp::Node(node_name, rclcpp::NodeOptions(options)
                                .automatically_declare_parameters_from_overrides(true))
  {
  }

protected:
  static std::string normalize(const std::string& name)
  {
    std::string out = name;
    if (!out.empty() && out[0] == '~') out.erase(0, 1);
    for (auto& c : out)
      if (c == '/') c = '.';
    return out;
  }

  rclcpp::Parameter require_param(const std::string& raw_name)
  {
    const std::string name = normalize(raw_name);
    if (!has_parameter(name))
      throw std::runtime_error("Required parameter '" + name +
                               "' is not set for node '" + get_name() +
                               "'. Check the YAML/launch configuration.");
    return get_parameter(name);
  }

  double get_double(const std::string& raw_name)
  {
    return as_double(require_param(raw_name));
  }
  double get_double(const std::string& raw_name, double default_value)
  {
    const std::string name = normalize(raw_name);
    if (!has_parameter(name)) declare_parameter(name, default_value);
    return as_double(get_parameter(name));
  }

  int get_int(const std::string& raw_name)
  {
    return static_cast<int>(require_param(raw_name).as_int());
  }
  int get_int(const std::string& raw_name, int default_value)
  {
    const std::string name = normalize(raw_name);
    if (!has_parameter(name)) declare_parameter(name, default_value);
    const auto p = get_parameter(name);
    return p.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE
             ? static_cast<int>(p.as_double())
             : static_cast<int>(p.as_int());
  }

  bool get_bool(const std::string& raw_name)
  {
    return require_param(raw_name).as_bool();
  }
  bool get_bool(const std::string& raw_name, bool default_value)
  {
    const std::string name = normalize(raw_name);
    if (!has_parameter(name)) declare_parameter(name, default_value);
    return get_parameter(name).as_bool();
  }

  std::string get_string(const std::string& raw_name)
  {
    return require_param(raw_name).as_string();
  }
  std::string get_string(const std::string& raw_name,
                         const std::string& default_value)
  {
    const std::string name = normalize(raw_name);
    if (!has_parameter(name)) declare_parameter(name, default_value);
    return get_parameter(name).as_string();
  }

  std::vector<double> get_double_array(const std::string& raw_name)
  {
    return as_double_array(require_param(raw_name));
  }
  std::vector<double> get_double_array(const std::string& raw_name,
                                       const std::vector<double>& default_value)
  {
    const std::string name = normalize(raw_name);
    if (!has_parameter(name)) declare_parameter(name, default_value);
    return as_double_array(get_parameter(name));
  }

private:
  static std::vector<double> as_double_array(const rclcpp::Parameter& p)
  {
    if (p.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY) {
      const auto ints = p.as_integer_array();
      return std::vector<double>(ints.begin(), ints.end());
    }
    return p.as_double_array();
  }

  static double as_double(const rclcpp::Parameter& p)
  {
    return p.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER
             ? static_cast<double>(p.as_int())
             : p.as_double();
  }
};

}  // namespace sonar_slam
