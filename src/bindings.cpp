#include <omp.h>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bemt/airfoil.hpp"
#include "bemt/bspline.hpp"
#include "bemt/config.hpp"
#include "bemt/solver.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace bemt;

namespace {

BSplineCurve parse_bspline_curve_arg(const nb::handle& obj,
                                     int default_degree,
                                     double x_min,
                                     double x_max,
                                     const ConstRefArrayXd& explicit_r_norm,
                                     const char* name) {
    if (nb::isinstance<BSplineCurve>(obj)) {
        return nb::cast<BSplineCurve>(obj);
    }

    try {
        ConstRefMatrixXd mat = nb::cast<ConstRefMatrixXd>(obj);
        if (mat.cols() == 2 && mat.rows() >= 2 && explicit_r_norm.size() == 0) {
            return BSplineCurve(mat, default_degree);
        }
    } catch (...) {
    }

    try {
        ConstRefArrayXd vec = nb::cast<ConstRefArrayXd>(obj);
        if (explicit_r_norm.size() > 0) {
            return BSplineCurve(explicit_r_norm, vec, default_degree);
        }
        return BSplineCurve::from_1d_values(vec, default_degree, x_min, x_max);
    } catch (...) {
        throw std::invalid_argument(
            std::string("RotorGeometry: ") + name +
            " must be a BSplineCurve instance, a 2D numpy array of shape (N_ctrl, 2) with columns [r/R, value], or a 1D numpy array of shape (N_ctrl,).");
    }
}

} // namespace

NB_MODULE(_fast_bemt_impl, m) {
    m.doc() = "High-performance vectorized and differentiable oblique-inflow BEMT solver core";

    nb::enum_<WakeModel>(m, "WakeModel")
        .value("GlauertPittPeters", WakeModel::kGlauertPittPeters,
               "Annulus Glauert momentum balance + Pitt-Peters oblique wake harmonic correction (Scheme A)")
        .value("GlauertAnnulus", WakeModel::kGlauertAnnulus,
               "Pure annulus Glauert oblique momentum balance without harmonic skew gradient")
        .value("LocalElement", WakeModel::kLocalElement,
               "Independent local blade-element-sector momentum balance (Scheme B)")
        .export_values();

    nb::class_<BSplineCurve>(m, "BSplineCurve")
        .def(nb::init<>())
        .def(nb::init<const ConstRefArrayXd&, const ConstRefArrayXd&, int>(),
             "ctrl_r_norm"_a, "ctrl_vals"_a, "degree"_a = 3,
             "Construct 2D parametric B-spline curve (r/R(u), val(u)) from 1D arrays ctrl_r_norm and ctrl_vals")
        .def(nb::init<const ConstRefMatrixXd&, int>(),
             "ctrl_points"_a, "degree"_a = 3,
             "Construct 2D parametric B-spline curve from (N_ctrl x 2) matrix with columns [r/R, val]")
        .def_static("from_1d_values", &BSplineCurve::from_1d_values,
                    "ctrl_vals"_a, "degree"_a = 3, "r_norm_min"_a = 0.0, "r_norm_max"_a = 1.0,
                    "Construct BSplineCurve from 1D control values using Greville abscissae on [r_norm_min, r_norm_max]")
        .def("eval_u", &BSplineCurve::eval_u,
             "u"_a,
             "Evaluate (r_norm(u), val(u)) at scalar curve parameter u in [0, 1]")
        .def("eval_u_batch", &BSplineCurve::eval_u_batch,
             "u_vec"_a,
             "Evaluate (r_norm_array, val_array) at an array of curve parameters u in [0, 1]")
        .def("sample", &BSplineCurve::sample,
             "n_points"_a = 200,
             "Uniformly sample n_points along u in [0, 1], returning (r_norm_array, val_array) for plotting")
        .def("eval_at_r_norm", &BSplineCurve::eval_at_r_norm,
             "target_r_norm"_a,
             "Invert r/R(u*) = target_r_norm and evaluate scalar val(u*)")
        .def("eval_batch", &BSplineCurve::eval_batch,
             "target_r_norm"_a,
             "Invert r/R(u_i*) = target_r_norm[i] and evaluate val_array at radial stations")
        .def("__call__", &BSplineCurve::eval_at_r_norm,
             "target_r_norm"_a,
             "Evaluate scalar val(u*) at a single radial station r/R")
        .def("__call__", &BSplineCurve::eval_batch,
             "target_r_norm"_a,
             "Shorthand for eval_batch(target_r_norm)")
        .def("invert_r_norm",
             [](const BSplineCurve& self, double target_r_norm) {
                 ArrayXd arr(1);
                 arr[0] = target_r_norm;
                 return self.invert_r_norm(arr)[0];
             },
             "target_r_norm"_a,
             "Invert r/R(u*) = target_r_norm and return scalar curve parameter u* in [0, 1]")
        .def("invert_r_norm", &BSplineCurve::invert_r_norm,
             "target_r_norm"_a,
             "Invert r/R(u_i*) = target_r_norm[i] and return curve parameters u* in [0, 1]")
        .def("basis_matrix",
             [](const BSplineCurve& self, const ConstRefArrayXd& target_r_norm) {
                 return self.basis_matrix(target_r_norm, nullptr);
             },
             "target_r_norm"_a,
             "Compute (N_r x N_ctrl) B-spline basis matrix evaluated at target_r_norm")
        .def("slope_dy_dx",
             [](const BSplineCurve& self, double target_r_norm) {
                 ArrayXd arr(1);
                 arr[0] = target_r_norm;
                 return self.slope_dy_dx(arr)[0];
             },
             "target_r_norm"_a,
             "Compute scalar derivative d(val)/d(r/R) at target_r_norm")
        .def("slope_dy_dx", &BSplineCurve::slope_dy_dx,
             "target_r_norm"_a,
             "Compute derivative d(val)/d(r/R) at target_r_norm array")
        .def_prop_ro("degree", &BSplineCurve::degree)
        .def_prop_ro("num_ctrl_points", &BSplineCurve::num_ctrl_points)
        .def_prop_ro("knots", &BSplineCurve::knots)
        .def_prop_ro("ctrl_r_norm", &BSplineCurve::ctrl_r_norm)
        .def_prop_ro("ctrl_vals", &BSplineCurve::ctrl_vals)
        .def_prop_ro("ctrl_points", &BSplineCurve::ctrl_points);

    nb::class_<Airfoil3D>(m, "Airfoil3D")
        .def(nb::init<const ConstRefArrayXd&,
                      const ConstRefArrayXd&,
                      const ConstRefArrayXd&,
                      const ConstRefArrayXd&,
                      const ConstRefArrayXd&,
                      double,
                      bool>(),
             "alpha_grid"_a, "re_grid"_a, "ma_grid"_a,
             "cl_table"_a, "cd_table"_a,
             "aspect_ratio"_a = 10.0,
             "enable_extrapolation"_a = true,
             "Construct 3D monotone tricubic Hermite airfoil table over (alpha [rad], Re, Ma)")
        .def_static("from_1d_polar", &Airfoil3D::from_1d_polar,
                    "alpha_grid"_a, "cl_1d"_a, "cd_1d"_a,
                    "ref_re"_a = 1e6,
                    "aspect_ratio"_a = 10.0,
                    "enable_extrapolation"_a = true,
                    "Construct Airfoil3D from 1D polar arrays (alpha [rad], Cl, Cd)")
        .def_static("interpolate_section", &Airfoil3D::interpolate_section,
                    "af0"_a, "af1"_a, "weight"_a,
                    "Linearly blend two Airfoil3D tables: (1 - weight) * af0 + weight * af1")
        .def_static("interpolate_spanwise_sections", &Airfoil3D::interpolate_spanwise_sections,
                    "section_r_norm"_a, "section_airfoils"_a, "target_r_norm"_a,
                    "Interpolate M master airfoil tables at section_r_norm onto N_r blade stations target_r_norm")
        .def("eval", &Airfoil3D::eval<double>,
             "alpha"_a, "re"_a = 1e6, "ma"_a = 0.0,
             "Evaluate (Cl, Cd) at scalar (alpha [rad], Re, Ma)")
        .def("eval_batch",
             [](const Airfoil3D& self,
                const ConstRefArrayXd& alpha_vec,
                const ConstRefArrayXd& re_vec,
                const ConstRefArrayXd& ma_vec) {
                 const Eigen::Index n = alpha_vec.size();
                 if (re_vec.size() != n || ma_vec.size() != n) {
                     throw std::invalid_argument("eval_batch: alpha_vec, re_vec, and ma_vec must have the same length.");
                 }
                 ArrayXd cl_out(n), cd_out(n);
                 self.eval_batch(alpha_vec, re_vec, ma_vec, cl_out, cd_out);
                 return std::make_pair(cl_out, cd_out);
             },
             "alpha_vec"_a, "re_vec"_a, "ma_vec"_a,
             "Vectorized batch evaluation returning (cl_array, cd_array)")
        .def_prop_ro("zero_lift_alpha", &Airfoil3D::zero_lift_alpha)
        .def_prop_ro("linear_lift_slope", &Airfoil3D::linear_lift_slope)
        .def_prop_ro("alpha_min", &Airfoil3D::alpha_min)
        .def_prop_ro("alpha_max", &Airfoil3D::alpha_max);

    nb::class_<RotorGeometry>(m, "RotorGeometry")
        .def(nb::init<>())
        .def("__init__",
             [](RotorGeometry* t,
                int num_blades,
                double radius,
                double hub_radius,
                int n_radial,
                const BSplineCurve& chord_curve,
                const BSplineCurve& twist_curve,
                std::vector<std::shared_ptr<const Airfoil3D>> airfoils,
                const ConstRefArrayXd& section_r_norm,
                int spin_dir) {
                 new (t) RotorGeometry();
                 t->num_blades       = num_blades;
                 t->radius           = radius;
                 t->hub_radius       = hub_radius;
                 t->n_radial         = n_radial;
                 t->spin_dir         = spin_dir;
                 t->chord_curve      = chord_curve;
                 t->twist_curve      = twist_curve;
                 t->section_r_norm   = section_r_norm;
                 t->section_airfoils = std::move(airfoils);
                 t->validate_and_finalize();
             },
             "num_blades"_a, "radius"_a, "hub_radius"_a,
             "n_radial"_a, "chord_curve"_a, "twist_curve"_a, "airfoils"_a,
             "section_r_norm"_a = ArrayXd(),
             "spin_dir"_a = 1)
        .def("__init__",
             [](RotorGeometry* t,
                int num_blades,
                double radius,
                double hub_radius,
                const ConstRefArrayXd& r,
                const BSplineCurve& chord_curve,
                const BSplineCurve& twist_curve,
                std::vector<std::shared_ptr<const Airfoil3D>> airfoils,
                const ConstRefArrayXd& section_r_norm,
                int spin_dir,
                const ConstRefArrayXd& dr) {
                 new (t) RotorGeometry();
                 t->num_blades       = num_blades;
                 t->radius           = radius;
                 t->hub_radius       = hub_radius;
                 t->spin_dir         = spin_dir;
                 t->r                = r;
                 if (dr.size() > 0) {
                     t->dr = dr;
                 }
                 t->chord_curve      = chord_curve;
                 t->twist_curve      = twist_curve;
                 t->section_r_norm   = section_r_norm;
                 t->section_airfoils = std::move(airfoils);
                 t->validate_and_finalize();
             },
             "num_blades"_a, "radius"_a, "hub_radius"_a,
             "r"_a, "chord_curve"_a, "twist_curve"_a, "airfoils"_a,
             "section_r_norm"_a = ArrayXd(),
             "spin_dir"_a = 1,
             "dr"_a = ArrayXd())
        .def("__init__",
             [](RotorGeometry* t,
                int num_blades,
                double radius,
                double hub_radius,
                int n_radial,
                nb::handle chord_ctrl,
                nb::handle twist_ctrl,
                std::vector<std::shared_ptr<const Airfoil3D>> airfoils,
                const ConstRefArrayXd& section_r_norm,
                int spin_dir,
                int bspline_degree,
                const ConstRefArrayXd& chord_ctrl_r_norm,
                const ConstRefArrayXd& twist_ctrl_r_norm) {
                 new (t) RotorGeometry();
                 t->num_blades       = num_blades;
                 t->radius           = radius;
                 t->hub_radius       = hub_radius;
                 t->n_radial         = n_radial;
                 t->spin_dir         = spin_dir;
                 t->bspline_degree   = bspline_degree;
                 t->section_r_norm   = section_r_norm;
                 t->section_airfoils = std::move(airfoils);
                 const double x_hub  = hub_radius / std::max(radius, 1e-12);
                 t->chord_curve = parse_bspline_curve_arg(chord_ctrl, bspline_degree, x_hub, 1.0, chord_ctrl_r_norm, "chord_curve/chord_ctrl");
                 t->twist_curve = parse_bspline_curve_arg(twist_ctrl, bspline_degree, x_hub, 1.0, twist_ctrl_r_norm, "twist_curve/twist_ctrl");
                 t->validate_and_finalize();
             },
             "num_blades"_a, "radius"_a, "hub_radius"_a,
             "n_radial"_a, "chord_ctrl"_a, "twist_ctrl"_a, "airfoils"_a,
             "section_r_norm"_a = ArrayXd(),
             "spin_dir"_a = 1,
             "bspline_degree"_a = 3,
             "chord_ctrl_r_norm"_a = ArrayXd(),
             "twist_ctrl_r_norm"_a = ArrayXd())
        .def("__init__",
             [](RotorGeometry* t,
                int num_blades,
                double radius,
                double hub_radius,
                const ConstRefArrayXd& r,
                nb::handle chord_ctrl,
                nb::handle twist_ctrl,
                std::vector<std::shared_ptr<const Airfoil3D>> airfoils,
                int spin_dir,
                const ConstRefArrayXd& dr,
                int bspline_degree,
                const ConstRefArrayXd& section_r_norm,
                const ConstRefArrayXd& chord_ctrl_r_norm,
                const ConstRefArrayXd& twist_ctrl_r_norm) {
                 new (t) RotorGeometry();
                 t->num_blades       = num_blades;
                 t->radius           = radius;
                 t->hub_radius       = hub_radius;
                 t->spin_dir         = spin_dir;
                 t->bspline_degree   = bspline_degree;
                 t->r                = r;
                 t->section_r_norm   = section_r_norm;
                 t->section_airfoils = std::move(airfoils);
                 if (dr.size() > 0) {
                     t->dr = dr;
                 }
                 const double x_hub = hub_radius / std::max(radius, 1e-12);
                 t->chord_curve = parse_bspline_curve_arg(chord_ctrl, bspline_degree, x_hub, 1.0, chord_ctrl_r_norm, "chord_curve/chord_ctrl");
                 t->twist_curve = parse_bspline_curve_arg(twist_ctrl, bspline_degree, x_hub, 1.0, twist_ctrl_r_norm, "twist_curve/twist_ctrl");
                 t->validate_and_finalize();
             },
             "num_blades"_a, "radius"_a, "hub_radius"_a,
             "r"_a, "chord_ctrl"_a, "twist_ctrl"_a, "airfoils"_a,
             "spin_dir"_a = 1,
             "dr"_a = ArrayXd(),
             "bspline_degree"_a = 3,
             "section_r_norm"_a = ArrayXd(),
             "chord_ctrl_r_norm"_a = ArrayXd(),
             "twist_ctrl_r_norm"_a = ArrayXd())
        .def_rw("num_blades", &RotorGeometry::num_blades)
        .def_rw("radius", &RotorGeometry::radius)
        .def_rw("hub_radius", &RotorGeometry::hub_radius)
        .def_rw("spin_dir", &RotorGeometry::spin_dir)
        .def_rw("n_radial", &RotorGeometry::n_radial)
        .def_rw("bspline_degree", &RotorGeometry::bspline_degree)
        .def_rw("r", &RotorGeometry::r)
        .def_rw("dr", &RotorGeometry::dr)
        .def_rw("chord_curve", &RotorGeometry::chord_curve)
        .def_rw("twist_curve", &RotorGeometry::twist_curve)
        .def_rw("chord_ctrl_r_norm", &RotorGeometry::chord_ctrl_r_norm)
        .def_rw("chord_ctrl", &RotorGeometry::chord_ctrl)
        .def_rw("twist_ctrl_r_norm", &RotorGeometry::twist_ctrl_r_norm)
        .def_rw("twist_ctrl", &RotorGeometry::twist_ctrl)
        .def_rw("section_r_norm", &RotorGeometry::section_r_norm)
        .def_rw("section_airfoils", &RotorGeometry::section_airfoils)
        .def_ro("chord_knots", &RotorGeometry::chord_knots)
        .def_ro("twist_knots", &RotorGeometry::twist_knots)
        .def_ro("chord_u_params", &RotorGeometry::chord_u_params)
        .def_ro("twist_u_params", &RotorGeometry::twist_u_params)
        .def_ro("chord_basis", &RotorGeometry::chord_basis)
        .def_ro("twist_basis", &RotorGeometry::twist_basis)
        .def_ro("dchord_dr_norm", &RotorGeometry::dchord_dr_norm)
        .def_ro("dtwist_dr_norm", &RotorGeometry::dtwist_dr_norm)
        .def_ro("chord", &RotorGeometry::chord)
        .def_ro("twist", &RotorGeometry::twist)
        .def_ro("airfoils", &RotorGeometry::airfoils)
        .def("validate_and_finalize", &RotorGeometry::validate_and_finalize);

    m.def("build_clamped_uniform_knots", &bspline::build_clamped_uniform_knots,
          "n_ctrl"_a, "degree"_a = 3);
    m.def("build_greville_abscissae", &bspline::build_greville_abscissae,
          "n_ctrl"_a, "degree"_a = 3, "x_min"_a = 0.0, "x_max"_a = 1.0);
    m.def("build_bspline_basis_matrix",
          [](const ConstRefArrayXd& r, double hub_radius, double radius, int n_ctrl, int degree) {
              return bspline::build_basis_matrix(r, hub_radius, radius, n_ctrl, degree, nullptr);
          },
          "r"_a, "hub_radius"_a, "radius"_a, "n_ctrl"_a, "degree"_a = 3);
    m.def("build_parametric_bspline_basis_matrix",
          [](const ConstRefArrayXd& target_r_norm, const ConstRefArrayXd& ctrl_r_norm, int degree) {
              return bspline::build_parametric_basis_matrix(target_r_norm, ctrl_r_norm, degree, nullptr, nullptr);
          },
          "target_r_norm"_a, "ctrl_r_norm"_a, "degree"_a = 3);

    nb::class_<OperatingCondition>(m, "OperatingCondition")
        .def(nb::init<>())
        .def("__init__",
             [](OperatingCondition* t,
                double vx, double vy, double vz, double omega,
                double pitch_0, double pitch_1c, double pitch_1s,
                double rho, double mu, double a_sound) {
                 new (t) OperatingCondition();
                 t->vx       = vx;
                 t->vy       = vy;
                 t->vz       = vz;
                 t->omega    = omega;
                 t->pitch_0  = pitch_0;
                 t->pitch_1c = pitch_1c;
                 t->pitch_1s = pitch_1s;
                 t->rho      = rho;
                 t->mu       = mu;
                 t->a_sound  = a_sound;
             },
             "vx"_a = 0.0, "vy"_a = 0.0, "vz"_a = 0.0, "omega"_a = 100.0,
             "pitch_0"_a = 0.0, "pitch_1c"_a = 0.0, "pitch_1s"_a = 0.0,
             "rho"_a = 1.225, "mu"_a = 1.7894e-5, "a_sound"_a = 340.29)
        .def_static("from_freestream", &OperatingCondition::from_freestream,
                    "v_inf"_a, "alpha_disk_rad"_a, "omega_rad_s"_a,
                    "pitch_0_rad"_a = 0.0,
                    "pitch_1c_rad"_a = 0.0,
                    "pitch_1s_rad"_a = 0.0,
                    "beta_rad"_a = 0.0,
                    "rho"_a = 1.225, "mu"_a = 1.7894e-5, "a_sound"_a = 340.29)
        .def_rw("vx", &OperatingCondition::vx)
        .def_rw("vy", &OperatingCondition::vy)
        .def_rw("vz", &OperatingCondition::vz)
        .def_rw("omega", &OperatingCondition::omega)
        .def_rw("pitch_0", &OperatingCondition::pitch_0)
        .def_rw("pitch_1c", &OperatingCondition::pitch_1c)
        .def_rw("pitch_1s", &OperatingCondition::pitch_1s)
        .def_rw("rho", &OperatingCondition::rho)
        .def_rw("mu", &OperatingCondition::mu)
        .def_rw("a_sound", &OperatingCondition::a_sound);

    nb::class_<SolverConfig>(m, "SolverConfig")
        .def(nb::init<>())
        .def("__init__",
             [](SolverConfig* t,
                int n_azimuth,
                WakeModel wake_model,
                bool enable_tip_loss,
                bool enable_vrs_correction,
                bool enable_stall_delay,
                bool enable_tangential_induction,
                int max_newton_iters,
                int max_brent_iters,
                int max_wake_iters,
                double tol,
                double newton_max_step) {
                 new (t) SolverConfig();
                 t->n_azimuth                   = n_azimuth;
                 t->wake_model                  = wake_model;
                 t->enable_tip_loss             = enable_tip_loss;
                 t->enable_vrs_correction       = enable_vrs_correction;
                 t->enable_stall_delay          = enable_stall_delay;
                 t->enable_tangential_induction = enable_tangential_induction;
                 t->max_newton_iters            = max_newton_iters;
                 t->max_brent_iters             = max_brent_iters;
                 t->max_wake_iters              = max_wake_iters;
                 t->tol                         = tol;
                 t->newton_max_step             = newton_max_step;
             },
             "n_azimuth"_a = 36,
             "wake_model"_a = WakeModel::kGlauertPittPeters,
             "enable_tip_loss"_a = true,
             "enable_vrs_correction"_a = true,
             "enable_stall_delay"_a = true,
             "enable_tangential_induction"_a = true,
             "max_newton_iters"_a = 15,
             "max_brent_iters"_a = 60,
             "max_wake_iters"_a = 4,
             "tol"_a = 1e-10,
             "newton_max_step"_a = 0.25)
        .def_rw("n_azimuth", &SolverConfig::n_azimuth)
        .def_rw("wake_model", &SolverConfig::wake_model)
        .def_rw("enable_tip_loss", &SolverConfig::enable_tip_loss)
        .def_rw("enable_vrs_correction", &SolverConfig::enable_vrs_correction)
        .def_rw("enable_stall_delay", &SolverConfig::enable_stall_delay)
        .def_rw("enable_tangential_induction", &SolverConfig::enable_tangential_induction)
        .def_rw("max_newton_iters", &SolverConfig::max_newton_iters)
        .def_rw("max_brent_iters", &SolverConfig::max_brent_iters)
        .def_rw("max_wake_iters", &SolverConfig::max_wake_iters)
        .def_rw("tol", &SolverConfig::tol)
        .def_rw("newton_max_step", &SolverConfig::newton_max_step);

    nb::class_<BEMTSensitivity>(m, "BEMTSensitivity")
        .def_ro("valid", &BEMTSensitivity::valid)
        .def_ro("dT_dchord_ctrl", &BEMTSensitivity::dT_dchord_ctrl)
        .def_ro("dT_dtwist_ctrl", &BEMTSensitivity::dT_dtwist_ctrl)
        .def_ro("dQ_dchord_ctrl", &BEMTSensitivity::dQ_dchord_ctrl)
        .def_ro("dQ_dtwist_ctrl", &BEMTSensitivity::dQ_dtwist_ctrl)
        .def_ro("dP_dchord_ctrl", &BEMTSensitivity::dP_dchord_ctrl)
        .def_ro("dP_dtwist_ctrl", &BEMTSensitivity::dP_dtwist_ctrl)
        .def_ro("dFx_dchord_ctrl", &BEMTSensitivity::dFx_dchord_ctrl)
        .def_ro("dFx_dtwist_ctrl", &BEMTSensitivity::dFx_dtwist_ctrl)
        .def_ro("dFy_dchord_ctrl", &BEMTSensitivity::dFy_dchord_ctrl)
        .def_ro("dFy_dtwist_ctrl", &BEMTSensitivity::dFy_dtwist_ctrl)
        .def_ro("dMx_dchord_ctrl", &BEMTSensitivity::dMx_dchord_ctrl)
        .def_ro("dMx_dtwist_ctrl", &BEMTSensitivity::dMx_dtwist_ctrl)
        .def_ro("dMy_dchord_ctrl", &BEMTSensitivity::dMy_dchord_ctrl)
        .def_ro("dMy_dtwist_ctrl", &BEMTSensitivity::dMy_dtwist_ctrl)
        .def_ro("dT_dchord_ctrl_r", &BEMTSensitivity::dT_dchord_ctrl_r)
        .def_ro("dT_dtwist_ctrl_r", &BEMTSensitivity::dT_dtwist_ctrl_r)
        .def_ro("dQ_dchord_ctrl_r", &BEMTSensitivity::dQ_dchord_ctrl_r)
        .def_ro("dQ_dtwist_ctrl_r", &BEMTSensitivity::dQ_dtwist_ctrl_r)
        .def_ro("dP_dchord_ctrl_r", &BEMTSensitivity::dP_dchord_ctrl_r)
        .def_ro("dP_dtwist_ctrl_r", &BEMTSensitivity::dP_dtwist_ctrl_r)
        .def_ro("dFx_dchord_ctrl_r", &BEMTSensitivity::dFx_dchord_ctrl_r)
        .def_ro("dFx_dtwist_ctrl_r", &BEMTSensitivity::dFx_dtwist_ctrl_r)
        .def_ro("dFy_dchord_ctrl_r", &BEMTSensitivity::dFy_dchord_ctrl_r)
        .def_ro("dFy_dtwist_ctrl_r", &BEMTSensitivity::dFy_dtwist_ctrl_r)
        .def_ro("dMx_dchord_ctrl_r", &BEMTSensitivity::dMx_dchord_ctrl_r)
        .def_ro("dMx_dtwist_ctrl_r", &BEMTSensitivity::dMx_dtwist_ctrl_r)
        .def_ro("dMy_dchord_ctrl_r", &BEMTSensitivity::dMy_dchord_ctrl_r)
        .def_ro("dMy_dtwist_ctrl_r", &BEMTSensitivity::dMy_dtwist_ctrl_r)
        .def_ro("dT_dchord", &BEMTSensitivity::dT_dchord)
        .def_ro("dT_dtwist", &BEMTSensitivity::dT_dtwist)
        .def_ro("dQ_dchord", &BEMTSensitivity::dQ_dchord)
        .def_ro("dQ_dtwist", &BEMTSensitivity::dQ_dtwist)
        .def_ro("dP_dchord", &BEMTSensitivity::dP_dchord)
        .def_ro("dP_dtwist", &BEMTSensitivity::dP_dtwist)
        .def_ro("dFx_dchord", &BEMTSensitivity::dFx_dchord)
        .def_ro("dFx_dtwist", &BEMTSensitivity::dFx_dtwist)
        .def_ro("dFy_dchord", &BEMTSensitivity::dFy_dchord)
        .def_ro("dFy_dtwist", &BEMTSensitivity::dFy_dtwist)
        .def_ro("dMx_dchord", &BEMTSensitivity::dMx_dchord)
        .def_ro("dMx_dtwist", &BEMTSensitivity::dMx_dtwist)
        .def_ro("dMy_dchord", &BEMTSensitivity::dMy_dchord)
        .def_ro("dMy_dtwist", &BEMTSensitivity::dMy_dtwist)
        .def_ro("dT_dcl", &BEMTSensitivity::dT_dcl)
        .def_ro("dT_dcd", &BEMTSensitivity::dT_dcd)
        .def_ro("dQ_dcl", &BEMTSensitivity::dQ_dcl)
        .def_ro("dQ_dcd", &BEMTSensitivity::dQ_dcd)
        .def_ro("dP_dcl", &BEMTSensitivity::dP_dcl)
        .def_ro("dP_dcd", &BEMTSensitivity::dP_dcd)
        .def_ro("dFx_dcl", &BEMTSensitivity::dFx_dcl)
        .def_ro("dFx_dcd", &BEMTSensitivity::dFx_dcd)
        .def_ro("dFy_dcl", &BEMTSensitivity::dFy_dcl)
        .def_ro("dFy_dcd", &BEMTSensitivity::dFy_dcd)
        .def_ro("dMx_dcl", &BEMTSensitivity::dMx_dcl)
        .def_ro("dMx_dcd", &BEMTSensitivity::dMx_dcd)
        .def_ro("dMy_dcl", &BEMTSensitivity::dMy_dcl)
        .def_ro("dMy_dcd", &BEMTSensitivity::dMy_dcd)
        .def_ro("dForces_dCond", &BEMTSensitivity::dForces_dCond)
        .def_ro("dTr_dchord_diag", &BEMTSensitivity::dTr_dchord_diag)
        .def_ro("dTr_dtwist_diag", &BEMTSensitivity::dTr_dtwist_diag);

    nb::class_<BEMTResult>(m, "BEMTResult")
        .def_ro("converged", &BEMTResult::converged)
        .def_ro("total_fallback_cells", &BEMTResult::total_fallback_cells)
        .def_ro("thrust", &BEMTResult::thrust)
        .def_ro("torque", &BEMTResult::torque)
        .def_ro("power", &BEMTResult::power)
        .def_ro("fx", &BEMTResult::fx)
        .def_ro("fy", &BEMTResult::fy)
        .def_ro("fz", &BEMTResult::fz)
        .def_ro("mx", &BEMTResult::mx)
        .def_ro("my", &BEMTResult::my)
        .def_ro("mz", &BEMTResult::mz)
        .def_ro("ct", &BEMTResult::ct)
        .def_ro("cq", &BEMTResult::cq)
        .def_ro("cp", &BEMTResult::cp)
        .def_ro("fm", &BEMTResult::fm)
        .def_ro("vi_disk_mean", &BEMTResult::vi_disk_mean)
        .def_ro("wake_skew_chi", &BEMTResult::wake_skew_chi)
        .def_ro("pitt_kx", &BEMTResult::pitt_kx)
        .def_ro("pitt_ky", &BEMTResult::pitt_ky)
        .def_ro("psi_grid", &BEMTResult::psi_grid)
        .def_ro("vi_annulus", &BEMTResult::vi_annulus)
        .def_ro("dT_dr_mean", &BEMTResult::dT_dr_mean)
        .def_ro("dQ_dr_mean", &BEMTResult::dQ_dr_mean)
        .def_ro("phi", &BEMTResult::phi)
        .def_ro("alpha", &BEMTResult::alpha)
        .def_ro("vi", &BEMTResult::vi)
        .def_ro("vt", &BEMTResult::vt)
        .def_ro("cl", &BEMTResult::cl)
        .def_ro("cd", &BEMTResult::cd)
        .def_ro("re", &BEMTResult::re)
        .def_ro("ma", &BEMTResult::ma)
        .def_ro("f_tip", &BEMTResult::f_tip)
        .def_ro("df_z", &BEMTResult::df_z)
        .def_ro("df_theta", &BEMTResult::df_theta)
        .def_ro("sensitivity", &BEMTResult::sensitivity);

    nb::class_<BEMTSolver>(m, "BEMTSolver")
        .def(nb::init<RotorGeometry, SolverConfig>(),
             "geom"_a, "cfg"_a = SolverConfig{})
        .def("solve", &BEMTSolver::solve,
             "cond"_a, "compute_gradients"_a = false,
             nb::call_guard<nb::gil_scoped_release>(),
             "Solve single operating condition with optional analytical IFT gradients")
        .def("solve_batch", &BEMTSolver::solve_batch,
             "conds"_a, "compute_gradients"_a = false,
             nb::call_guard<nb::gil_scoped_release>(),
             "Solve multiple operating conditions in parallel via OpenMP")
        .def_prop_ro("geometry", &BEMTSolver::geometry)
        .def_prop_ro("config", &BEMTSolver::config);

    m.def("set_num_threads", &omp_set_num_threads, "num_threads"_a,
        "Set the number of OpenMP threads used by BEMTSolver");
    m.def("get_max_threads", &omp_get_max_threads,
        "Get current maximum number of OpenMP threads");
}
