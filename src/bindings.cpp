#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bemt/airfoil.hpp"
#include "bemt/config.hpp"
#include "bemt/solver.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace bemt;

NB_MODULE(_fast_bemt_impl, m) {
    m.doc() = "High-performance vectorized and differentiable oblique-inflow BEMT solver core";

    // ========================================================================
    // 1. 枚举类型 WakeModel
    // ========================================================================
    nb::enum_<WakeModel>(m, "WakeModel")
        .value("GlauertPittPeters", WakeModel::kGlauertPittPeters,
               "Annulus Glauert momentum balance + Pitt-Peters oblique wake harmonic correction (Scheme A)")
        .value("GlauertAnnulus", WakeModel::kGlauertAnnulus,
               "Pure annulus Glauert oblique momentum balance without harmonic skew gradient")
        .value("LocalElement", WakeModel::kLocalElement,
               "Independent local blade-element-sector momentum balance (Scheme B)")
        .export_values();

    // ========================================================================
    // 2. 三维翼型插值类 Airfoil3D
    // ========================================================================
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

    // ========================================================================
    // 3. 旋翼几何结构体 RotorGeometry
    // ========================================================================
    nb::class_<RotorGeometry>(m, "RotorGeometry")
        .def(nb::init<>())
        .def("__init__",
             [](RotorGeometry* t,
                int num_blades,
                double radius,
                double hub_radius,
                const ConstRefArrayXd& r,
                const ConstRefArrayXd& chord,
                const ConstRefArrayXd& twist,
                std::vector<std::shared_ptr<const Airfoil3D>> airfoils,
                int spin_dir,
                const ConstRefArrayXd& dr) {
                 new (t) RotorGeometry();
                 t->num_blades = num_blades;
                 t->radius     = radius;
                 t->hub_radius = hub_radius;
                 t->spin_dir   = spin_dir;
                 t->r          = r;
                 t->chord      = chord;
                 t->twist      = twist;
                 t->airfoils   = std::move(airfoils);
                 if (dr.size() > 0) {
                     t->dr = dr;
                 }
                 t->validate_and_finalize();
             },
             "num_blades"_a, "radius"_a, "hub_radius"_a,
             "r"_a, "chord"_a, "twist"_a, "airfoils"_a,
             "spin_dir"_a = 1,
             "dr"_a = ArrayXd())
        .def_rw("num_blades", &RotorGeometry::num_blades)
        .def_rw("radius", &RotorGeometry::radius)
        .def_rw("hub_radius", &RotorGeometry::hub_radius)
        .def_rw("spin_dir", &RotorGeometry::spin_dir)
        .def_rw("r", &RotorGeometry::r)
        .def_rw("dr", &RotorGeometry::dr)
        .def_rw("chord", &RotorGeometry::chord)
        .def_rw("twist", &RotorGeometry::twist)
        .def_rw("airfoils", &RotorGeometry::airfoils)
        .def("validate_and_finalize", &RotorGeometry::validate_and_finalize);

    // ========================================================================
    // 4. 运行工况结构体 OperatingCondition
    // ========================================================================
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

    // ========================================================================
    // 5. 求解器配置结构体 SolverConfig
    // ========================================================================
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
                double tol) {
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
             "tol"_a = 1e-10)
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

    // ========================================================================
    // 6. 解析导数结果结构体 BEMTSensitivity
    // ========================================================================
    nb::class_<BEMTSensitivity>(m, "BEMTSensitivity")
        .def_ro("valid", &BEMTSensitivity::valid)
        .def_ro("dT_dchord", &BEMTSensitivity::dT_dchord)
        .def_ro("dT_dtwist", &BEMTSensitivity::dT_dtwist)
        .def_ro("dQ_dchord", &BEMTSensitivity::dQ_dchord)
        .def_ro("dQ_dtwist", &BEMTSensitivity::dQ_dtwist)
        .def_ro("dP_dchord", &BEMTSensitivity::dP_dchord)
        .def_ro("dP_dtwist", &BEMTSensitivity::dP_dtwist)
        .def_ro("dMx_dchord", &BEMTSensitivity::dMx_dchord)
        .def_ro("dMx_dtwist", &BEMTSensitivity::dMx_dtwist)
        .def_ro("dMy_dchord", &BEMTSensitivity::dMy_dchord)
        .def_ro("dMy_dtwist", &BEMTSensitivity::dMy_dtwist)
        .def_ro("dForces_dCond", &BEMTSensitivity::dForces_dCond)
        .def_ro("dTr_dchord_diag", &BEMTSensitivity::dTr_dchord_diag)
        .def_ro("dTr_dtwist_diag", &BEMTSensitivity::dTr_dtwist_diag);

    // ========================================================================
    // 7. 求解结果结构体 BEMTResult
    // ========================================================================
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

    // ========================================================================
    // 8. 核心求解器类 BEMTSolver (求解时自动释放 Python GIL 锁，充分发挥 OpenMP 性能)
    // ========================================================================
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
}