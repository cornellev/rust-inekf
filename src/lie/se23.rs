use nalgebra::{Matrix3, Matrix5, SMatrix, SVector, Vector3};
use std::ops::Mul;
use super::so3;

// easy types for state definition
//TODO: need to change to include biases, right?
pub type Vector9 = SVector<f64, 9>;
pub type Matrix9 = SMatrix<f64, 9, 9>;

#[derive(Debug, Clone, Copy)]
pub struct SE23 {
    pub r: Matrix3<f64>,
    pub v: Vector3<f64>,
    pub p: Vector3<f64>,
}

impl SE23 {
    pub fn new(r: Matrix3<f64>, v: Vector3<f64>, p: Vector3<f64>) -> Self {
        Self { r, v, p }
    }

    pub fn identity() -> Self {
        Self::new(Matrix3::identity(), Vector3::zeros(), Vector3::zeros())
    }

    // Closed form inverse: (R^T, -R^T v, -R^T p).
    pub fn inverse(&self) -> Self {
        let rt: Matrix3<f64> = self.r.transpose();
        Self::new(rt, -rt * self.v, -rt * self.p)
    }

    // Convert SE23 state to Matrix5 object
    pub fn to_matrix(&self) -> Matrix5<f64> {
        let mut mat = Matrix5::identity();
        mat.fixed_view_mut::<3, 3>(0,0).copy_from(&self.r);
        mat.fixed_view_mut::<3, 1>(0,3).copy_from(&self.v);
        mat.fixed_view_mut::<3, 1>(0,4).copy_from(&self.p);
        mat
    }

    pub fn adjoint(&self) -> Matrix9 {
        let mut a = Matrix9::zeros();
        a.fixed_view_mut::<3, 3>(0, 0).copy_from(&self.r);
        a.fixed_view_mut::<3, 3>(3, 3).copy_from(&self.r);
        a.fixed_view_mut::<3, 3>(6, 6).copy_from(&self.r);
        a.fixed_view_mut::<3, 3>(3, 0).copy_from(&(so3::hat(&self.v) * self.r));
        a.fixed_view_mut::<3, 3>(6, 0).copy_from(&(so3::hat(&self.p) * self.r));
        a
    }
}

// Be able to print state
impl std::fmt::Display for SE23 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.to_matrix(), f)
    }
}

impl Mul for SE23 {
    type Output = SE23;

    //standard SE_2(3) group composition
    fn mul(self, rhs: SE23) -> SE23 {
        SE23::new(
            self.r * rhs.r,
            self.r  *rhs.v + self.v,
            self.r * rhs.p + self.p
        )
    }
}

// xi^ = [[phi^, rho_v, rho_p], [0, 0_2x2]]
pub fn hat(xi: &Vector9) -> Matrix5<f64> {
    let mut mat = Matrix5::zeros();
    mat.fixed_view_mut::<3,3>(0,0)
        .copy_from(&so3::hat(&xi.fixed_rows::<3>(0).into_owned())); // rotation block
    mat.fixed_view_mut::<3, 1>(0,3).copy_from(&xi.fixed_rows::<3>(3)); // velocity
    // block
    mat.fixed_view_mut::<3,1>(0,4).copy_from(&xi.fixed_rows::<3>(6));
    mat
}

pub fn vee(mat: &Matrix5<f64>) -> Vector9 {
    let mut xi = Vector9::zeros();
    xi.fixed_rows_mut::<3>(0)
        .copy_from(&so3::vee(&mat.fixed_view::<3,3>(0,0).into_owned()));
    xi.fixed_rows_mut::<3>(3).copy_from(&mat.fixed_view::<3,1>(0,3));
    xi.fixed_rows_mut::<3>(6).copy_from(&mat.fixed_view::<3,1>(0,4));
    xi
}

pub fn exp(xi: &Vector9) -> SE23 {
    let phi: Vector3<f64> = xi.fixed_rows::<3>(0).into_owned();
    let jl: Matrix3<f64> = so3::left_jacobian(&phi);
    SE23::new(
        so3::exp(&phi),
        jl * xi.fixed_rows::<3>(3),
        jl * xi.fixed_rows::<3>(6),
    )
}

pub fn log(x: &SE23) -> Vector9 {
    let phi = so3::log(&x.r);
    let jinv = so3::left_jacobian_inv(&phi);
    let mut xi = Vector9::zeros();
    xi.fixed_rows_mut::<3>(0).copy_from(&phi);
    xi.fixed_rows_mut::<3>(3).copy_from(&(jinv * x.v));
    xi.fixed_rows_mut::<3>(6).copy_from(&(jinv * x.p));
    xi
}

pub fn ad(xi: &Vector9) -> Matrix9 {
    let phi_hat = so3::hat(&xi.fixed_rows::<3>(0).into_owned());
    let mut a = Matrix9::zeros();
    a.fixed_view_mut::<3, 3>(0, 0).copy_from(&phi_hat);
    a.fixed_view_mut::<3, 3>(3, 3).copy_from(&phi_hat);
    a.fixed_view_mut::<3, 3>(6, 6).copy_from(&phi_hat);
    a.fixed_view_mut::<3, 3>(3, 0)
        .copy_from(&so3::hat(&xi.fixed_rows::<3>(3).into_owned()));
    a.fixed_view_mut::<3, 3>(6, 0)
        .copy_from(&so3::hat(&xi.fixed_rows::<3>(6).into_owned()));
    a
}

// ------------------------------------
// test suite
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Matrix2;
    use proptest::prelude::*;
    use std::f64::consts::PI;

    // -- helper functions --
    // check matrix exponential against "brute force"
    pub(crate) fn mexp<const N: usize>(a: &SMatrix<f64, N, N>) -> SMatrix<f64, N, N> {
        let s = (a.norm().log2().ceil().max(0.0) as i32) + 1;
        let scaled = a / 2f64.powi(s);
        let mut term = SMatrix::<f64, N, N>::identity();
        let mut accumulated = SMatrix::<f64, N, N>::identity();
        for k in 1..=24 {
            term = term * scaled / k as f64;
            accumulated += term;
        }
        for _ in 0..s {
            accumulated = accumulated * accumulated;
        }
        accumulated
    }

    // Algebra element: rotation angle drawn from `angles`,
    // translations uniform in [-5,5].
    fn xi_in(angles: impl Strategy<Value = f64>) -> impl Strategy<Value = Vector9> {
        (
            prop::array::uniform3(-1.0f64..1.0),
            angles,
            prop::array::uniform6(-5.0f64..5.0)
        ).prop_map(|(axis, angle, t)| {
            let a = Vector3::from_column_slice(&axis);
            let phi = if a.norm() > 1e-12 {
                a.normalize() * angle
            } else {
                Vector3::zeros()
            };
            let mut xi = Vector9::zeros();
            xi.fixed_rows_mut::<3>(0).copy_from(&phi);
            for i in 0..6 {
                xi[3 + i] = t[i];
            }
            xi
        })
    }

    // ||phi|| < pi; required wherever log is involved
    fn any_xi() -> impl Strategy<Value = Vector9> {
        xi_in(0.0f64..PI - 1e-6)
    }

    // unrestricted angle; only for exp / Ad, which are entire
    fn wide_xi() -> impl Strategy<Value = Vector9> {
        xi_in(0.0f64..6.0)
    }

    // Group element without SE23::exp
    pub(crate) fn any_x() -> impl Strategy<Value = SE23> {
        any_xi().prop_map(|xi| {
            SE23::new(
                so3::exp(&xi.fixed_rows::<3>(0).into_owned()),
                Vector3::new(xi[3],xi[4],xi[5]),
                Vector3::new(xi[6],xi[7],xi[8]),
            )
        })
    }


    // -- actual tests --

    #[test]
    fn adjoint_of_identity() {
        assert_relative_eq!(
            SE23::identity().adjoint(),
            Matrix9::identity(),
            epsilon=1e-15
        );
    }

    proptest! {
        // group tests
        #[test]
        fn test_exact_inverse(x in any_x()) {
            assert_relative_eq!(
                (x * x.inverse()).to_matrix(),
                Matrix5::identity(),
                epsilon=1e-13
            );

            assert_relative_eq!(
                (x.inverse() * x).to_matrix(),
                Matrix5::identity(),
                epsilon=1e-13
            );
        }

        #[test]
        fn compose_matches_expected(a in any_x(), b in any_x()) {
            assert_relative_eq!(
                (a * b).to_matrix(),
                a.to_matrix() * b.to_matrix(),
                epsilon = 1e-13
            );
        }

        //NOTE: why 0.0? Why does that make sense?
        #[test]
        fn to_matrix_correct_shape(x in any_x()) {
            let m = x.to_matrix();
            for i in 3..5 {
                for j in 0..3 {
                    assert_eq!(m[(i, j)], 0.0);
                }
            }
            assert_relative_eq!(m.fixed_view::<2,2>(3,3).into_owned(), Matrix2::identity());
        }

        #[test]
        fn log_inverts_exp(xi in any_xi()) {
            assert_relative_eq!(log(&exp(&xi)), xi, epsilon=1e-8);
        }

        #[test]
        fn exp_inverts_log(x in any_x()) {
            assert_relative_eq!(
                exp(&log(&x)).to_matrix(),
                x.to_matrix(),
                epsilon=1e-11
            );
        }

        // lie algebra tests
        #[test]
        fn hat_vee_roundtrip(xi in wide_xi()) {
            assert_relative_eq!(vee(&hat(&xi)), xi, epsilon=1e-15);
        }

        #[test]
        fn hat_has_zero_bottom_block(xi in wide_xi()) {
            assert_relative_eq!(
                hat(&xi).fixed_view::<2,5>(3,0).into_owned(),
                SMatrix::<f64, 2,5>::zeros()
            );
        }

        #[test]
        fn exp_matches_mexp(xi in wide_xi()) {
            assert_relative_eq!(
                exp(&xi).to_matrix(),
                mexp(&hat(&xi)),
                epsilon=1e-12
            );
        }

        // adjoint tests
        #[test]
        fn ad_matches_commutation(xi in wide_xi(), eta in wide_xi()) {
            let bracket = hat(&xi) * hat(&eta) - hat(&eta) * hat(&xi);
            assert_relative_eq!(
                hat(&(ad(&xi) * eta)),
                bracket,
                epsilon = 1e-12
            );
        }

        #[test]
        fn adjoint_is_homomorphism(a in any_x(), b in any_x()) {
            assert_relative_eq!(
                (a * b).adjoint(),
                a.adjoint() * b.adjoint(),
                epsilon=1e-11
            );

            assert_relative_eq!(
                a.inverse().adjoint(),
                a.adjoint().try_inverse().unwrap(),
                epsilon=1e-11
            );
        }

        #[test]
        fn adjoint_exp_is_exp_ad(xi in wide_xi()) {
            assert_relative_eq!(
                exp(&xi).adjoint(),
                mexp(&ad(&xi)),
                epsilon=1e-11
            );
        }

        #[test]
        fn conjugation_matches_adjoint(x in any_x(), xi in any_xi()) {
            let lhs = x * exp(&xi) * x.inverse();
            let rhs = exp(&(x.adjoint() * xi));
            assert_relative_eq!(
                lhs.to_matrix(),
                rhs.to_matrix(),
                epsilon=1e-10
            );
        }
    }
}
