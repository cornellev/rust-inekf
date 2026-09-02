use nalgebra::{Matrix3, Matrix5, SMatrix, SVector, Vector3};
use std::ops::Mul;
use super::so3;

// easy types for state definition
//TODO: need to change to include biases, right?
pub type Vector9 = SVector<f64, 9>;
pub type Matrix9 = SMatrix<f64, 9, 9>;

#[derive(Debug, Clone, Copy, PartialEq)]
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

// test suite
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Matrix2;
    use proptest::prelude::*;
    use std::f64::consts::PI;

    // -- helper functions --
    // check matrix exponential against "brute force"
    fn mexp<const N: usize>(a: &SMatrix<f64, N, N>) -> SMatrix<f64, N, N> {
        let s = (a.norm().log2().ceil().max(0.0) as i32) * 4;
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

    //TODO: this should be random numbers, not just this one, right?
    fn sample_xi() -> Vector9 {
        Vector9::from_column_slice(&[0.3, -0.7, 1.1, 2.0, -0.5, 0.8, 1.2, 0.4, 3.0])
    }

    fn sample_x() -> SE23 {
        SE23::new(
            so3::exp(&Vector3::new(0.2, -0.5, 0.9)),
            Vector3::new(1.0, -0.4, 0.2),
            Vector3::new(-3.0, 0.5, 12.0),
        )
    }

    fn second_sample_x() -> SE23 {
        SE23::new(
            so3::exp(&Vector3::new(-1.1, 0.3, 0.6)),
            Vector3::new(0.7, 2.0,-1.5),
            Vector3::new(4.0, -2.5, 0.1)
        )
    }


    // -- actual tests --

    // first: group tests
    #[test]
    fn test_exact_inverse() {
        let x = sample_x();
        assert_relative_eq!(
            (x * x.inverse()).to_matrix(),
            Matrix5::identity(), 
            epsilon=1e-14
        );

        assert_relative_eq!(
            (x.inverse() * x).to_matrix(),
            Matrix5::identity(),
            epsilon=1e-14
        );
    }

    #[test]
    fn compose_matches_expected() {
        let (a,b) = (sample_x(), second_sample_x());
        assert_relative_eq!(
            (a * b).to_matrix(),
            a.to_matrix() * b.to_matrix(),
            epsilon = 1e-13
        );
    }

    //NOTE: why 0.0? Why does that make sense?
    #[test]
    fn to_matrix_correct_shape() {
        let m = sample_x().to_matrix();
        for i in 3..5 {
            for j in 0..3 {
                assert_eq!(m[(i, j)], 0.0);
            }
        }
        assert_relative_eq!(m.fixed_view::<2,2>(3,3).into_owned(), Matrix2::identity());
    }

    #[test]
    fn log_inverts_exp() {
        for xi in [
            //TODO: add RNG
            sample_xi(),
            2.0 * sample_xi()
        ] {
            assert_relative_eq!(log(&exp(&xi)), xi, epsilon=1e-9);
        }
    }

    #[test]
    fn exp_inverts_log() {
        for x in [
            sample_x(),
            second_sample_x(),
            SE23::identity()
        ] {
            assert_relative_eq!(
                exp(&log(&x)).to_matrix(),
                x.to_matrix(),
                epsilon=1e-12
            );
        }
    }

    proptest! {
        #[test]
        fn log_exp_roundtrip(c in prop::array::uniform9(-1.5f64..1.5)) {
            let xi = Vector9::from_column_slice(&c);
            prop_assume!(xi.fixed_rows::<3>(0).norm() < PI - 1e-6);
        }
    }

    // lie algebra tests
    #[test]
    fn hat_vee_roundtrip() {
        let xi = sample_xi();
        assert_relative_eq!(vee(&hat(&xi)), xi, epsilon=1e-15);
    }

    #[test]
    fn hat_has_zero_bottom_block() {
        let h = hat(&sample_xi());
        assert_relative_eq!(
            h.fixed_view::<2,5>(3,0).into_owned(),
            SMatrix::<f64, 2,5>::zeros()
        );
    }

    #[test]
    fn exp_matches_mexp() {
        for xi in [
            //TODO: add more Vector9 objects via random numbers
            sample_xi(),
            Vector9::zeros()
        ] {
            assert_relative_eq!(
                exp(&xi).to_matrix(),
                mexp(&hat(&xi)),
                epsilon=1e-12
            );
        }
    }

    // adjoint test
    #[test]
    fn ad_matches_commutation() {
        let xi = sample_xi();
        let eta = Vector9::from_column_slice(&[-0.6, 1.3, 0.2, 0.9, -1.7, 0.4, 2.2, 0.1, -0.8]);

        let bracket = hat(&xi) * hat(&eta) - hat(&eta) * hat(&xi);
        assert_relative_eq!(
            hat(&(ad(&xi) * eta)),
            bracket,
            epsilon = 1e-13
        );
    }

    // check homomorphism identities for adjoint
    #[test]
    fn adjoint_is_homomorphism() {
        let (a,b) = (sample_x(), second_sample_x());

        assert_relative_eq!(
            (a * b).adjoint(),
            a.adjoint() * b.adjoint(),
            epsilon=1e-12
        );

        assert_relative_eq!(
            a.inverse().adjoint(),
            a.adjoint().try_inverse().unwrap(),
            epsilon=1e-12
        );

        assert_relative_eq!(
            SE23::identity().adjoint(),
            Matrix9::identity(),
            epsilon=1e-15
        );
    }
}
