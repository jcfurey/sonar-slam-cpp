// Georeferencing without GDAL: WGS84 -> UTM (Karney/Krüger series, mm-level
// over a zone) plus a survey datum tying the SLAM map frame to the world.
// The datum is [latitude deg, longitude deg, bearing deg] where bearing is
// the compass bearing (degrees clockwise from true north) of the map +x
// axis — the vehicle's initial heading in this stack, since dead reckoning
// zeroes yaw on the first attitude sample.
#pragma once

#include <cmath>
#include <string>

namespace sonar_slam {

struct UtmPoint
{
  double easting = 0.0;
  double northing = 0.0;
  int zone = 0;
  bool north = true;
};

inline UtmPoint wgs84_to_utm(double lat_deg, double lon_deg)
{
  constexpr double a = 6378137.0, f = 1.0 / 298.257223563, k0 = 0.9996;
  const double n = f / (2.0 - f);
  const double A =
    a / (1.0 + n) * (1.0 + n * n / 4.0 + n * n * n * n / 64.0);
  const double a1 = n / 2.0 - 2.0 * n * n / 3.0 + 5.0 * n * n * n / 16.0;
  const double a2 = 13.0 * n * n / 48.0 - 3.0 * n * n * n / 5.0;
  const double a3 = 61.0 * n * n * n / 240.0;

  UtmPoint u;
  u.zone = static_cast<int>(std::floor((lon_deg + 180.0) / 6.0)) + 1;
  u.zone = std::min(60, std::max(1, u.zone));
  u.north = lat_deg >= 0.0;
  const double lon0 = (u.zone * 6.0 - 183.0) * M_PI / 180.0;

  const double phi = lat_deg * M_PI / 180.0;
  const double lam = lon_deg * M_PI / 180.0 - lon0;
  const double sn = 2.0 * std::sqrt(n) / (1.0 + n);
  const double t =
    std::sinh(std::atanh(std::sin(phi)) - sn * std::atanh(sn * std::sin(phi)));
  const double xi = std::atan2(t, std::cos(lam));
  const double eta = std::atanh(std::sin(lam) / std::sqrt(1.0 + t * t));

  double E = eta, N = xi;
  const double aj[3] = {a1, a2, a3};
  for (int j = 1; j <= 3; ++j) {
    E += aj[j - 1] * std::cos(2.0 * j * xi) * std::sinh(2.0 * j * eta);
    N += aj[j - 1] * std::sin(2.0 * j * xi) * std::cosh(2.0 * j * eta);
  }
  u.easting = 500000.0 + k0 * A * E;
  u.northing = (u.north ? 0.0 : 10000000.0) + k0 * A * N;
  return u;
}

// EPSG-style WKT for the matching .prj sidecar (QGIS/ArcGIS read PNG+PGW+PRJ
// as a georeferenced raster)
inline std::string utm_wkt(int zone, bool north)
{
  const int central = zone * 6 - 183;
  return "PROJCS[\"WGS 84 / UTM zone " + std::to_string(zone) +
         (north ? "N" : "S") +
         "\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS "
         "84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\""
         ",0.0174532925199433]],PROJECTION[\"Transverse_Mercator\"],"
         "PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\"," +
         std::to_string(central) +
         "],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\","
         "500000],PARAMETER[\"false_northing\"," +
         (north ? std::string("0") : std::string("10000000")) +
         "],UNIT[\"metre\",1]]";
}

// Map-frame -> world transforms for a datum
struct GeoDatum
{
  double lat0 = 0.0, lon0 = 0.0;
  double bearing_deg = 0.0;  // compass bearing of map +x (deg CW from north)
  UtmPoint origin;
  double ex = 1.0, nx = 0.0;  // (E, N) unit vector of map +x
  double ey = 0.0, ny = 1.0;  // (E, N) unit vector of map +y

  GeoDatum() = default;
  GeoDatum(double lat, double lon, double bearing)
  : lat0(lat), lon0(lon), bearing_deg(bearing), origin(wgs84_to_utm(lat, lon))
  {
    // The compass bearing is TRUE-north; UTM axes are GRID-north, which
    // differs by the grid convergence (up to ~3 deg near zone edges) and a
    // point scale factor (~4e-4). Build the exact local true-ENU -> UTM
    // Jacobian by finite-differencing the projection over one meter of true
    // north / true east (geodesic steps via the correct meridional M(phi)
    // and normal N(phi) radii) — convergence and scale come along for free.
    const double phi = lat0 * M_PI / 180.0;
    const double dlat_per_m = (1.0 / meridional_radius(phi)) * 180.0 / M_PI;
    const double dlon_per_m =
      (1.0 / (normal_radius(phi) * std::cos(phi))) * 180.0 / M_PI;
    const UtmPoint north = wgs84_to_utm(lat0 + dlat_per_m, lon0);
    const UtmPoint east = wgs84_to_utm(lat0, lon0 + dlon_per_m);
    const double eE = east.easting - origin.easting;    // dUTM / d(true east)
    const double eN = east.northing - origin.northing;
    const double nE = north.easting - origin.easting;   // dUTM / d(true north)
    const double nN = north.northing - origin.northing;

    // map +x points along the TRUE bearing; +y is +x rotated 90 deg CCW
    const double b = bearing * M_PI / 180.0;
    const double xe = std::sin(b), xn = std::cos(b);   // +x in true ENU
    const double ye = -std::cos(b), yn = std::sin(b);  // +y in true ENU
    ex = xe * eE + xn * nE;
    nx = xe * eN + xn * nN;
    ey = ye * eE + yn * nE;
    ny = ye * eN + yn * nN;
  }

  static double meridional_radius(double phi)
  {
    constexpr double a = 6378137.0, f = 1.0 / 298.257223563;
    const double e2 = f * (2.0 - f);
    const double s2 = std::sin(phi) * std::sin(phi);
    return a * (1.0 - e2) / std::pow(1.0 - e2 * s2, 1.5);
  }

  static double normal_radius(double phi)
  {
    constexpr double a = 6378137.0, f = 1.0 / 298.257223563;
    const double e2 = f * (2.0 - f);
    const double s2 = std::sin(phi) * std::sin(phi);
    return a / std::sqrt(1.0 - e2 * s2);
  }

  void map_to_utm(double x, double y, double& E, double& N) const
  {
    E = origin.easting + x * ex + y * ey;
    N = origin.northing + x * nx + y * ny;
  }

  // small-extent (survey-scale) lon/lat for GeoJSON output, using the
  // correct local radii (the equatorial-sphere shortcut was 2.5-6.7 m/km off)
  void map_to_lonlat(double x, double y, double& lon, double& lat) const
  {
    const double b = bearing_deg * M_PI / 180.0;
    const double dE = x * std::sin(b) - y * std::cos(b);  // true ENU meters
    const double dN = x * std::cos(b) + y * std::sin(b);
    const double phi = lat0 * M_PI / 180.0;
    lat = lat0 + (dN / meridional_radius(phi)) * 180.0 / M_PI;
    lon = lon0 + (dE / (normal_radius(phi) * std::cos(phi))) * 180.0 / M_PI;
  }
};

}  // namespace sonar_slam
