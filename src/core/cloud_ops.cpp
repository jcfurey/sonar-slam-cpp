#include "sonar_slam_cpp/cloud_ops.hpp"

#include <pointmatcher/PointMatcher.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/point_types.h>

#include <fstream>
#include <iostream>

namespace sonar_slam {

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;
typedef std::shared_ptr<PM::DataPoints> DPPtr;

namespace {

DPPtr from_eigen(const Matrix& mat)
{
  DP::Labels labels;
  labels.push_back(DP::Label("x", 1));
  labels.push_back(DP::Label("y", 1));
  if (mat.cols() == 3) labels.push_back(DP::Label("z", 1));
  labels.push_back(DP::Label("pad", 1));

  PM::Matrix padded_mat = PM::Matrix::Ones(labels.size(), mat.rows());
  padded_mat.block(0, 0, labels.size() - 1, mat.rows()) = mat.transpose();

  return DPPtr(new DP(padded_mat, labels));
}

DPPtr from_eigen(const Matrix& mat, const Matrix& desc)
{
  DP::Labels labels;
  labels.push_back(DP::Label("x", 1));
  labels.push_back(DP::Label("y", 1));
  if (mat.cols() == 3) labels.push_back(DP::Label("z", 1));
  labels.push_back(DP::Label("pad", 1));

  PM::Matrix padded_mat = PM::Matrix::Ones(labels.size(), mat.rows());
  padded_mat.block(0, 0, labels.size() - 1, mat.rows()) = mat.transpose();

  DP::Labels desc_labels;
  for (int col = 0; col < desc.cols(); ++col)
    desc_labels.push_back(DP::Label("desc" + std::to_string(col), 1));

  return DPPtr(new DP(padded_mat, labels, desc.transpose(), desc_labels));
}

}  // namespace

Matrix downsample(const Matrix& mat_in, float resolution)
{
  if (mat_in.rows() == 0) return mat_in;

  PointMatcherSupport::Parametrizable::Parameters params;
  params["maxSizeByNode"] = std::to_string(resolution);
  params["samplingMethod"] = "3";
  std::shared_ptr<PM::DataPointsFilter> filter =
    PM::get().DataPointsFilterRegistrar.create("OctreeGridDataPointsFilter", params);
  DPPtr cloud_in = from_eigen(mat_in);
  filter->inPlaceFilter(*cloud_in);
  return cloud_in->features.topRows(mat_in.cols()).transpose();
}

std::pair<Matrix, Matrix> downsample(const Matrix& mat_in, const Matrix& desc_in,
                                     float resolution)
{
  if (mat_in.rows() == 0) return std::make_pair(mat_in, desc_in);

  PointMatcherSupport::Parametrizable::Parameters params;
  params["maxSizeByNode"] = std::to_string(resolution);
  params["samplingMethod"] = "3";
  std::shared_ptr<PM::DataPointsFilter> filter =
    PM::get().DataPointsFilterRegistrar.create("OctreeGridDataPointsFilter", params);
  DPPtr cloud_in = from_eigen(mat_in, desc_in);
  filter->inPlaceFilter(*cloud_in);

  Matrix mat_out = cloud_in->features.topRows(mat_in.cols()).transpose();
  Matrix desc_out = cloud_in->descriptors.transpose();
  return std::make_pair(mat_out, desc_out);
}

Matrix remove_outlier(const Matrix& mat_in, double radius, int min_points)
{
  if (mat_in.rows() == 0) return mat_in;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(
    new pcl::PointCloud<pcl::PointXYZ>(mat_in.rows(), 1));
  for (int row = 0; row < mat_in.rows(); ++row) {
    cloud_in->at(row).x = mat_in(row, 0);
    cloud_in->at(row).y = mat_in(row, 1);
    cloud_in->at(row).z = mat_in.cols() == 3 ? mat_in(row, 2) : 0.f;
  }

  pcl::RadiusOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud_in);
  sor.setRadiusSearch(radius);
  sor.setMinNeighborsInRadius(min_points);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
  sor.filter(*cloud_out);

  Matrix mat_out(cloud_out->size(), mat_in.cols());
  for (std::size_t row = 0; row < cloud_out->size(); ++row) {
    mat_out(row, 0) = cloud_out->at(row).x;
    mat_out(row, 1) = cloud_out->at(row).y;
    if (mat_in.cols() == 3) mat_out(row, 2) = cloud_out->at(row).z;
  }
  return mat_out;
}

std::pair<Eigen::MatrixXi, Matrix> match(const Matrix& mat_ref,
                                         const Matrix& mat_in, int knn,
                                         float max_dist)
{
  PointMatcherSupport::Parametrizable::Parameters params;
  params["knn"] = std::to_string(knn);
  params["maxDist"] = std::to_string(max_dist);
  std::shared_ptr<PM::Matcher> matcher =
    PM::get().MatcherRegistrar.create("KDTreeMatcher", params);

  DPPtr cloud_ref = from_eigen(mat_ref);
  DPPtr cloud_in = from_eigen(mat_in);

  matcher->init(*cloud_ref);
  PM::Matches matches = matcher->findClosests(*cloud_in);
  return std::make_pair(matches.ids, matches.dists);
}

// ------------------------------------------------------------------------ ICP
struct ICP::Impl
{
  PM::ICP icp;
};

ICP::ICP() : impl_(new Impl) { impl_->icp.setDefault(); }
ICP::~ICP() = default;
ICP::ICP(ICP&&) noexcept = default;
ICP& ICP::operator=(ICP&&) noexcept = default;

void ICP::load_from_yaml(const std::string& filename)
{
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cout << "Failed to load " << filename << ". Use default configuration."
              << std::endl;
    impl_->icp.setDefault();
  } else {
    impl_->icp.loadFromYaml(ifs);
  }
}

std::pair<std::string, Eigen::Matrix3f> ICP::compute(const Matrix& source,
                                                     const Matrix& target,
                                                     const Eigen::Matrix3f& guess)
{
  DPPtr pc_source = from_eigen(source);
  DPPtr pc_target = from_eigen(target);
  PM::TransformationParameters T = guess;
  try {
    T = impl_->icp(*pc_source, *pc_target, T);
  } catch (const PM::ConvergenceError& e) {
    return std::make_pair(std::string(e.what()), Eigen::Matrix3f(guess));
  }
  return std::make_pair(std::string("success"), Eigen::Matrix3f(T));
}

}  // namespace sonar_slam
