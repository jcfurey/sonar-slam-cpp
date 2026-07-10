#!/usr/bin/env python3
"""Function-parity driver vs the ORIGINAL bruce_slam implementations.

mode=gen: writes fixtures and runs the originals — the pybind natives
(bruce_slam.cfar, bruce_slam.pcl incl. ICP) are imported directly; the small
pure-Python pieces that transitively import gtsam-python (unavailable in the
container right now) are copied VERBATIM from CFAR.py / slam.py /
slam_objects.py below, with a minimal numpy Pose2 standing in for gtsam.Pose2.

mode=compare: numerically diffs py_* against cpp_* outputs.
File format: [i32 rows][i32 cols][row-major data].
"""
import math
import sys
import numpy as np

DIR = sys.argv[2] if len(sys.argv) > 2 else "/tmp/slam_parity"


def write_mat(name, arr):
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    with open(f"{DIR}/{name}", "wb") as f:
        f.write(np.array(arr.shape, np.int32).tobytes())
        f.write(arr.tobytes())


def read_mat(name, dtype):
    with open(f"{DIR}/{name}", "rb") as f:
        rows, cols = np.frombuffer(f.read(8), np.int32)
        return np.frombuffer(f.read(), dtype).reshape(rows, cols)


# --------------------------------------------------------------------------
# minimal Pose2 (gtsam.Pose2 semantics for compose/between/inverse/matrix)
# --------------------------------------------------------------------------
class Pose2:
    def __init__(self, x, y, t):
        self.x, self.y, self.t = float(x), float(y), float(t)

    def matrix(self):
        c, s = math.cos(self.t), math.sin(self.t)
        return np.array([[c, -s, self.x], [s, c, self.y], [0, 0, 1.0]])

    def compose(self, o):
        c, s = math.cos(self.t), math.sin(self.t)
        return Pose2(self.x + c * o.x - s * o.y, self.y + s * o.x + c * o.y,
                     math.atan2(math.sin(self.t + o.t), math.cos(self.t + o.t)))

    def inverse(self):
        c, s = math.cos(self.t), math.sin(self.t)
        return Pose2(-(c * self.x + s * self.y), -(-s * self.x + c * self.y), -self.t)

    def between(self, o):
        return self.inverse().compose(o)


# verbatim from slam_objects.py Keyframe.transform_points
def transform_points(points, pose):
    if len(points) == 0:
        return np.empty_like(points, np.float32)
    T = pose.matrix().astype(np.float32)
    return points.dot(T[:2, :2].T) + T[:2, 2]


# --------------------------------------------------------------------------
# verbatim threshold-factor math from CFAR.py (scipy.optimize.root originals)
# --------------------------------------------------------------------------
class PyCFARThresholds:
    def __init__(self, Ntc, Ngc, Pfa, rank):
        from scipy.optimize import root
        self.Ntc, self.Ngc, self.Pfa, self.rank = Ntc, Ngc, Pfa, rank
        self.threshold_factor_CA = self.Ntc * (self.Pfa ** (-1.0 / self.Ntc) - 1)

        def solve(f):
            x0 = self.threshold_factor_CA
            for ratio in np.logspace(-2, 2, 10):
                ret = root(f, x0 * ratio)
                if ret.success:
                    return ret.x[0]
            raise ValueError("not found")

        self.threshold_factor_SOCA = solve(self.calc_WGN_pfa_SOCA)
        self.threshold_factor_GOCA = solve(self.calc_WGN_pfa_GOCA)
        self.threshold_factor_OS = solve(self.calc_WGN_pfa_OS)

    def calc_WGN_pfa_GOSOCA_core(self, x):
        x = float(x)
        temp = 0.0
        for k in range(int(self.Ntc / 2)):
            l1 = math.lgamma(self.Ntc / 2 + k)
            l2 = math.lgamma(k + 1)
            l3 = math.lgamma(self.Ntc / 2)
            temp += math.exp(l1 - l2 - l3) * (2 + x / (self.Ntc / 2)) ** (-k)
        return temp * (2 + x / (self.Ntc / 2)) ** (-self.Ntc / 2)

    def calc_WGN_pfa_SOCA(self, x):
        return self.calc_WGN_pfa_GOSOCA_core(x) - self.Pfa / 2

    def calc_WGN_pfa_GOCA(self, x):
        x = float(x)
        temp = (1.0 + x / (self.Ntc / 2)) ** (-self.Ntc / 2)
        return temp - self.calc_WGN_pfa_GOSOCA_core(x) - self.Pfa / 2

    def calc_WGN_pfa_OS(self, x):
        l1 = math.lgamma(self.Ntc + 1)
        l2 = math.lgamma(self.Ntc - self.rank + 1)
        l4 = math.lgamma(x + self.Ntc - self.rank + 1)
        l6 = math.lgamma(x + self.Ntc + 1)
        return math.exp(l1 - l2 + l4 - l6) - self.Pfa


# --------------------------------------------------------------------------
# verbatim grid-cost subroutine from slam.py get_matching_cost_subroutine1
# --------------------------------------------------------------------------
def matching_cost_subroutine1(source_points, source_pose, target_points,
                              target_pose, point_noise):
    import cv2
    xmin, ymin = np.min(target_points, axis=0) - 2 * point_noise
    xmax, ymax = np.max(target_points, axis=0) + 2 * point_noise
    resolution = point_noise / 10.0
    xs = np.arange(xmin, xmax, resolution)
    ys = np.arange(ymin, ymax, resolution)
    target_grids = np.zeros((len(ys), len(xs)), np.uint8)

    r = np.int32(np.round((target_points[:, 1] - ymin) / resolution))
    c = np.int32(np.round((target_points[:, 0] - xmin) / resolution))
    r = np.clip(r, 0, target_grids.shape[0] - 1)
    c = np.clip(c, 0, target_grids.shape[1] - 1)
    target_grids[r, c] = 255

    dilate_hs = int(np.ceil(point_noise / resolution))
    dilate_size = 2 * dilate_hs + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_size, dilate_size), (dilate_hs, dilate_hs))
    target_grids = cv2.dilate(target_grids, kernel)

    def subroutine(x):
        delta = Pose2(*x)
        sample_source_pose = source_pose.compose(delta)
        sample_transform = target_pose.between(sample_source_pose)

        points = transform_points(source_points, sample_transform)
        r = np.int32(np.round((points[:, 1] - ymin) / resolution))
        c = np.int32(np.round((points[:, 0] - xmin) / resolution))
        inside = ((0 <= r) & (r < target_grids.shape[0]) &
                  (0 <= c) & (c < target_grids.shape[1]))
        return -np.sum(target_grids[r[inside], c[inside]] > 0)

    return subroutine


def gen():
    import os
    import shutil
    from ament_index_python.packages import get_package_share_directory
    from bruce_slam import cfar, pcl
    try:
        from sklearn.covariance import MinCovDet
    except ImportError:
        MinCovDet = None
        print("NOTE: sklearn unavailable in container -> skipping MCD reference")

    os.makedirs(DIR, exist_ok=True)
    rng = np.random.default_rng(7)

    # ------------------------------------------------------------- CFAR
    img = rng.integers(0, 90, size=(716, 512)).astype(np.float32)
    hot = rng.integers(0, 716 * 512, size=300)
    img.ravel()[hot] = 255.0
    write_mat("img.f32", img)

    det = PyCFARThresholds(40, 10, 1e-2, 10)
    taus = np.float32([det.threshold_factor_CA, det.threshold_factor_SOCA,
                       det.threshold_factor_GOCA, det.threshold_factor_OS])
    print("python taus:", taus)
    write_mat("taus.f32", taus)
    # feed the float32-rounded taus to BOTH sides so borderline pixels agree
    write_mat("py_mask_ca.u8", cfar.ca(img, 20, 5, float(taus[0])))
    write_mat("py_mask_soca.u8", cfar.soca(img, 20, 5, float(taus[1])))
    write_mat("py_mask_goca.u8", cfar.goca(img, 20, 5, float(taus[2])))
    write_mat("py_mask_os.u8", cfar.os(img, 20, 5, 10, float(taus[3])))

    # -------------------------------------------------------- cloud ops
    cloud = (rng.random((5000, 2), np.float32) * 60 - 30).astype(np.float32)
    keys = rng.integers(0, 10, size=(5000, 1)).astype(np.float32)
    write_mat("cloud.f32", cloud)
    write_mat("keys.f32", keys)
    write_mat("py_downsample.f32", pcl.downsample(cloud, 0.5))
    pts, desc = pcl.downsample(cloud, keys, 0.5)
    write_mat("py_downsample_pts.f32", pts)
    write_mat("py_downsample_keys.f32", desc)
    write_mat("py_outlier.f32", pcl.remove_outlier(cloud, 1.0, 5))

    match_ref = (rng.random((3000, 2), np.float32) * 40 - 20).astype(np.float32)
    match_in = (rng.random((1000, 2), np.float32) * 40 - 20).astype(np.float32)
    write_mat("match_ref.f32", match_ref)
    write_mat("match_in.f32", match_in)
    ids, dists = pcl.match(match_ref, match_in, 1, 0.5)
    write_mat("py_match_ids.i32", np.asarray(ids, np.int32))
    write_mat("py_match_dists.f32", np.asarray(dists, np.float32))

    # -------------------------------------------------------------- ICP
    t = np.linspace(0, 20, 400, dtype=np.float32)
    world = np.concatenate([
        np.c_[t, np.zeros_like(t)],
        np.c_[np.full_like(t, 20.0), t],
        np.c_[t, 0.3 * np.sin(t) + 10.0],
    ]).astype(np.float32)

    true_T = Pose2(0.4, -0.3, 0.08)
    icp_target = world
    icp_source = transform_points(world, true_T.inverse())
    icp_source = (icp_source + rng.normal(0, 0.02, icp_source.shape)).astype(np.float32)
    write_mat("icp_source.f32", icp_source)
    write_mat("icp_target.f32", icp_target)
    guess = np.eye(3, dtype=np.float32)
    write_mat("icp_guess.f32", guess)
    shutil.copy(get_package_share_directory("bruce_slam") + "/config/icp.yaml",
                f"{DIR}/icp.yaml")
    icp = pcl.ICP()
    icp.loadFromYaml(f"{DIR}/icp.yaml")
    msg, T = icp.compute(icp_source, icp_target, guess)
    print("python ICP:", msg, "T:", np.asarray(T)[0:2, 2], math.atan2(T[1, 0], T[0, 0]))
    write_mat("py_icp_T.f32", np.asarray(T, np.float32))

    # --------------------------------------------------- transform_points
    write_mat("py_transformed.f32", transform_points(cloud, Pose2(1.5, -2.0, 0.7)))

    # --------------------------------------------------- global-init cost
    source_pose = Pose2(1.0, 2.0, 0.3)
    target_pose = Pose2(0.5, 1.0, -0.2)
    gi_target = transform_points(world, target_pose.inverse()).astype(np.float32)
    gi_source = transform_points(world[::2], source_pose.inverse())
    gi_source = (gi_source + rng.normal(0, 0.05, gi_source.shape)).astype(np.float32)
    write_mat("gi_source.f32", gi_source)
    write_mat("gi_target.f32", gi_target)
    write_mat("gi_poses.f32", np.float32([[1.0, 2.0, 0.3], [0.5, 1.0, -0.2]]))
    deltas = (rng.random((200, 3)) * [1.0, 1.0, 0.3] - [0.5, 0.5, 0.15]).astype(np.float32)
    write_mat("gi_deltas.f32", deltas)

    sub = matching_cost_subroutine1(gi_source, source_pose, gi_target,
                                    target_pose, 0.5)
    costs = np.float32([[sub(d.astype(np.float64))] for d in deltas])
    write_mat("py_costs.f32", costs)

    # -------------------------------------------------------------- MCD
    mcd_samples = rng.normal([0.4, -0.3, 0.08], [0.02, 0.02, 0.01], (30, 3))
    mcd_samples[27] += [1.5, -2.0, 0.9]   # outliers MinCovDet must reject
    mcd_samples[13] += [-0.8, 1.1, -0.5]
    write_mat("mcd_samples.f32", mcd_samples.astype(np.float32))
    if MinCovDet is not None:
        fcov = MinCovDet(store_precision=False, support_fraction=0.8).fit(
            mcd_samples.astype(np.float32).astype(np.float64))
        write_mat("py_mcd_loc.f32", fcov.location_.astype(np.float32))
        write_mat("py_mcd_cov.f32", fcov.covariance_.astype(np.float32))
    print("fixtures + python outputs written to", DIR)


def compare():
    failures = 0

    def check(name, py, cpp, tol, required=True, as_set=False):
        nonlocal failures
        if as_set:
            py = np.array(sorted(map(tuple, np.round(py, 4).tolist())))
            cpp = np.array(sorted(map(tuple, np.round(cpp, 4).tolist())))
        if py.shape != cpp.shape:
            print(f"{name:24s} SHAPE MISMATCH py{py.shape} cpp{cpp.shape}")
            failures += required
            return
        diff = np.abs(py.astype(np.float64) - cpp.astype(np.float64)).max() if py.size else 0.0
        ok = diff <= tol
        print(f"{name:24s} max|diff| = {diff:.3e}  {'OK' if ok else 'FAIL'}")
        failures += required and not ok

    for alg in ("ca", "soca", "goca", "os"):
        check(f"cfar_{alg} (cpu)", read_mat(f"py_mask_{alg}.u8", np.uint8),
              read_mat(f"cpp_mask_{alg}.u8", np.uint8), 0)
        try:
            check(f"cfar_{alg} (gpu)", read_mat(f"py_mask_{alg}.u8", np.uint8),
                  read_mat(f"cpp_mask_gpu_{alg}.u8", np.uint8), 0)
        except FileNotFoundError:
            pass

    check("downsample", read_mat("py_downsample.f32", np.float32),
          read_mat("cpp_downsample.f32", np.float32), 0, as_set=True)
    check("downsample_pts", read_mat("py_downsample_pts.f32", np.float32),
          read_mat("cpp_downsample_pts.f32", np.float32), 0, as_set=True)
    check("downsample_keys", read_mat("py_downsample_keys.f32", np.float32),
          read_mat("cpp_downsample_keys.f32", np.float32), 0, as_set=True)
    check("remove_outlier", read_mat("py_outlier.f32", np.float32),
          read_mat("cpp_outlier.f32", np.float32), 0, as_set=True)
    check("match_ids", read_mat("py_match_ids.i32", np.int32),
          read_mat("cpp_match_ids.i32", np.int32), 0)
    # unmatched entries carry +inf distance on both sides; compare the inf
    # pattern exactly and the finite values numerically
    py_d = read_mat("py_match_dists.f32", np.float32)
    cpp_d = read_mat("cpp_match_dists.f32", np.float32)
    inf_pattern_ok = np.array_equal(np.isfinite(py_d), np.isfinite(cpp_d))
    print(f"{'match_dists inf-pattern':24s} {'OK' if inf_pattern_ok else 'FAIL'}")
    failures += not inf_pattern_ok
    finite = np.isfinite(py_d) & np.isfinite(cpp_d)
    check("match_dists (finite)", py_d[finite], cpp_d[finite], 1e-6)
    check("icp_T", read_mat("py_icp_T.f32", np.float32),
          read_mat("cpp_icp_T.f32", np.float32), 1e-3)
    check("transform_points", read_mat("py_transformed.f32", np.float32),
          read_mat("cpp_transformed.f32", np.float32), 1e-4)

    py_costs = read_mat("py_costs.f32", np.float32)
    cpp_costs = read_mat("cpp_costs.f32", np.float32)
    exact = float((py_costs == cpp_costs).mean())
    print(f"{'gi_costs exact-match':24s} {100 * exact:.1f}% of {len(py_costs)} samples")
    check("gi_costs", py_costs, cpp_costs, 2)

    # report-only: different randomized algorithms, closeness expected
    try:
        check("mcd_location", read_mat("py_mcd_loc.f32", np.float32).ravel(),
              read_mat("cpp_mcd_loc.f32", np.float32).ravel(), 0.01, required=False)
        check("mcd_covariance", read_mat("py_mcd_cov.f32", np.float32),
              read_mat("cpp_mcd_cov.f32", np.float32), 5e-4, required=False)
    except FileNotFoundError:
        print("mcd: no sklearn reference in container, sanity-checking cpp "
              "against known inlier distribution instead")
        loc = read_mat("cpp_mcd_loc.f32", np.float32).ravel()
        check("mcd_loc_vs_truth", np.float32([0.4, -0.3, 0.08]), loc, 0.02,
              required=False)

    print("RESULT:", "FAIL" if failures else "ALL PARITY CHECKS PASSED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    {"gen": gen, "compare": compare}[sys.argv[1]]()
