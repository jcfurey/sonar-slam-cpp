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

// Construct, spin and shut down a node, translating the one startup failure
// whose native message does not say what to do about it.
//
// Every node here uses automatically_declare_parameters_from_overrides, so
// rclcpp::Node's CONSTRUCTOR walks the whole override set. An empty YAML
// sequence (`key: []`) reaches rcl as UNSET rather than as an empty array,
// and auto-declare then throws
//   InvalidParameterValueException: parameter_value_from failed for
//   parameter 'key': No parameter value set
// which names the key but not the cause or the fix. It happens before any
// node code runs, so no accessor guard inside SlamNodeBase can catch it —
// this wrapper is the only place it can be intercepted.
template <class NodeT>
int run_node(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  int rc = 0;
  try {
    rclcpp::spin(std::make_shared<NodeT>());
  } catch (const rclcpp::exceptions::InvalidParameterValueException& e) {
    RCLCPP_FATAL(
      rclcpp::get_logger("sonar_slam"),
      "%s\n"
      "  This is almost always an EMPTY YAML LIST in a params file: `key: []` "
      "parses as UNSET in rcl, not as an empty array, and parameter "
      "auto-declaration rejects it at startup.\n"
      "  Fix: give the key a value, or comment it out entirely if you meant "
      "'not configured'.",
      e.what());
    rc = 1;
  } catch (const std::exception& e) {
    // Constructor validation throws std::runtime_error (require_param, the
    // config sanity checks); without this catch those abort via std::terminate
    // instead of taking the clean exit path this wrapper exists to provide.
    RCLCPP_FATAL(rclcpp::get_logger("sonar_slam"),
                 "Terminating after unrecoverable error: %s", e.what());
    rc = 1;
  }
  rclcpp::shutdown();
  return rc;
}

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
    const auto p = get_parameter(name);
    // An empty YAML sequence (`key: []`) reaches rcl as NOT_SET rather than
    // as an empty array, so has_parameter() is true but every as_*() accessor
    // throws rclcpp::exceptions::InvalidParameterTypeException — an opaque
    // startup crash. Name the real cause instead.
    if (p.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET)
      throw std::runtime_error(
        "Parameter '" + name + "' on node '" + get_name() +
        "' is declared but has no value. An empty YAML list (`" + name +
        ": []`) parses as UNSET in rcl — give it a value, or comment the key "
        "out entirely if you meant 'not configured'.");
    return p;
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
    // tolerate a YAML value written with a decimal point (declared DOUBLE),
    // matching the defaulted overload and this header's leniency contract
    const auto p = require_param(raw_name);
    return p.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE
             ? static_cast<int>(p.as_double())
             : static_cast<int>(p.as_int());
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

  // INTEGER-tolerant like get_double/get_int: ROS1-ported YAMLs commonly say
  // `use_gyro: 0`, which auto-declares as INTEGER and would throw from
  // as_bool() at startup
  static bool coerce_bool(const rclcpp::Parameter& p)
  {
    return p.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER
             ? p.as_int() != 0
             : p.as_bool();
  }
  bool get_bool(const std::string& raw_name)
  {
    return coerce_bool(require_param(raw_name));
  }
  bool get_bool(const std::string& raw_name, bool default_value)
  {
    const std::string name = normalize(raw_name);
    if (!has_parameter(name)) declare_parameter(name, default_value);
    return coerce_bool(get_parameter(name));
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
    const auto p = get_parameter(name);
    // `key: []` in YAML reaches rcl as NOT_SET, not as an empty array, so
    // as_double_array() would throw. For a DEFAULTED lookup that is exactly
    // "not configured" — which is what someone writing `datum: []` means —
    // so fall back rather than killing the node.
    if (p.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET)
      return default_value;
    return as_double_array(p);
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
