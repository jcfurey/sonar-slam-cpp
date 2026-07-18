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
    const double alpha = (90.0 - bearing) * M_PI / 180.0;  // ENU angle of +x
    ex = std::cos(alpha);
    nx = std::sin(alpha);
    ey = -std::sin(alpha);
    ny = std::cos(alpha);
  }

  void map_to_utm(double x, double y, double& E, double& N) const
  {
    E = origin.easting + x * ex + y * ey;
    N = origin.northing + x * nx + y * ny;
  }

  // small-extent (survey-scale) lon/lat for GeoJSON output
  void map_to_lonlat(double x, double y, double& lon, double& lat) const
  {
    constexpr double R = 6378137.0;
    const double dE = x * ex + y * ey;
    const double dN = x * nx + y * ny;
    lat = lat0 + (dN / R) * 180.0 / M_PI;
    lon = lon0 + (dE / (R * std::cos(lat0 * M_PI / 180.0))) * 180.0 / M_PI;
  }
};

}  // namespace sonar_slam
