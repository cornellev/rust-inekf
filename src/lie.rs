use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;

const SMALL_ANGLE: f64 = 1e-8;
const TAYLOR_THRESHOLD: f64 = 1e-2;
const GAMMA_THRESHOLD: f64 = 1e-1;

pub fn hat(w: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -w.z, w.y,
        w.z, 0.0, -w.x,
        -w.y, w.x, 0.0,
    )
}

pub fn vee(matrix: &Matrix3<f64>) -> Vector3<f64> {
    // The vee operator goes from lie algebra to standard vector space.
    Vector3::new(matrix[(2,1)], matrix[(0,2)], matrix[(1,0)])
}

pub fn exp(omega: &Vector3<f64>) -> Matrix3<f64> {
    let theta2 = omega.norm_squared();
    let k = hat(omega);

    let (a,b) = if theta2 < SMALL_ANGLE * SMALL_ANGLE {
        (1.0 - theta2 / 6.0, 0.5 - theta2 / 24.0)
    } else {
        let theta = theta2.sqrt();
        let half = theta / 2.0;
        let sinc_half = half.sin() / half;
        (theta.sin() / theta, 0.5 * sinc_half * sinc_half)
    };

    Matrix3::identity() + a * k + b * k * k
}

pub fn log(r: &Matrix3<f64>) -> Vector3<f64> {
    let rt = r.transpose();
    let v = vee(&(r-rt));
    let c = (r.trace() - 1.0) / 2.0;
    let s = v.norm() / 2.0;
    let theta = s.atan2(c);

    if theta < SMALL_ANGLE {
        // theta / (2 sin theta) = 0.5 * (1 + theta^2/6 + ...)
        v * (0.5 * (1.0 + theta * theta / 6.0))
    } else if theta < PI - 1e-3 {
        v * (theta / (2.0 * s))
    } else {
        let aat = ((r + rt) / 2.0 - Matrix3::identity() * c) / (1.0 - c);
        let i = (0..3)
            .max_by(|&x, &y| aat[(x,x)].partial_cmp(&aat[(y,y)]).unwrap())
            .unwrap();
        let mut axis = aat.column(i).into_owned().normalize();
        if axis.dot(&v) < 0.0 {
            axis = -axis;
        }
        axis * theta
    }
}

pub fn left_jacobian(
    omega: &Vector3<f64>
) -> Matrix3<f64> {
    let theta2 = omega.norm_squared();
    let k = hat(omega);

    let (a, b) = if theta2 < TAYLOR_THRESHOLD * TAYLOR_THRESHOLD {
(
        0.5 - theta2 / 24.0,
            1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0,
    )
    } else {
        let theta = theta2.sqrt();
        let half = theta / 2.0;
        let sinc_half = half.sin() / half;
        (
            0.5 * sinc_half * sinc_half,
            (theta - theta.sin()) / (theta * theta2)
        )
    };

    Matrix3::identity() + a * k + b * k * k
}

pub fn left_jacobian_inv(
    omega: &Vector3<f64>
) -> Matrix3<f64> {
    let theta2 = omega.norm_squared();
    let k = hat(omega);

    let c = if theta2 < TAYLOR_THRESHOLD * TAYLOR_THRESHOLD {
        1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0
    } else {
        let theta = theta2.sqrt();
        let half = theta / 2.0;
        (1.0 - half * half.cos() / half.sin()) / theta2
    };

    Matrix3::identity() - 0.5 * k + c * k * k
}

pub fn gamma2(
    phi: &Vector3<f64>
) -> Matrix3<f64> {
    let theta2 =phi.norm_squared();
    let theta4 = theta2 * theta2;
    let k = hat(phi);

    let (a, b)= if theta2 < GAMMA_THRESHOLD * GAMMA_THRESHOLD {
        let theta6 = theta4 * theta2;
        (
            1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0 - theta6 / 362880.0,
            1.0 / 24.0 - theta2 / 720.0 + theta4 / 40320.0 - theta6 / 3268800.0,
        )
    } else {
        let theta = theta2.sqrt();
        (
            (theta - theta.sin()) / (theta * theta2),
            (theta2 + 2.0 * theta.cos() - 2.0) / (2.0 * theta4),
        )
    };

    0.5 * Matrix3::identity() + a * k + b * k * k
}


#[cfg(test)]
mod tests {

use super::*;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    #[test]
    fn hat_matches_cross_product() {
        let w = Vector3::new(0.3, -1.2, 0.7);
        let v = Vector3::new(2.0, 0.5, -1.1);
        assert_relative_eq!(hat(&w) * v, w.cross(&v),epsilon=1e-12);
    }

    #[test]
    fn exp_matches_nalgebra() {
        use nalgebra::Rotation3;
        for w in [
            Vector3::new(0.1, -0.2, 0.3),
            Vector3::new(0.0, 0.0, 3.0),
            Vector3::new(1.5, 1.5, 1.5),
        ] {
            let expected = *Rotation3::from_scaled_axis(w).matrix();
            assert_relative_eq!(exp(&w), expected, epsilon=1e-12);
        }
    }
    
    #[test]
    fn log_inverts_exp() {
        for w in [
            Vector3::new(0.3, -0.4, 0.5),
            Vector3::new(1e-12, 0.0, 0.0),
            Vector3::new(0.0, 0.0, PI - 1e-6),
        ] {
            assert_relative_eq!(log(&exp(&w)), w, epsilon = 1e-9);
        }
    }

    #[test]
    fn small_angle_is_finite() {
        let r = exp(&Vector3::new(1e-14, 0.0, 0.0));
        assert!(r.iter().all(|x| x.is_finite()));
        assert_relative_eq!(r, Matrix3::identity(), epsilon=1e-13);
    }

    proptest! {
            #[test]
            fn roundtrip(x in -2.0f64..2.0, y in -2.0f64..2.0, z in -2.0f64..2.0) {
                let w = Vector3::new(x, y, z);
                prop_assume!(w.norm() < PI - 1e-6);
                prop_assert!((log(&exp(&w)) - w).norm() < 1e-9);
            }
    }

    //Inverse test
    #[test]
    fn jacobian_inverse_test() {
        for w in [
            Vector3::new(0.4, -0.7, 1.1),
            Vector3::new(1e-9, 0.0, 0.0),
            Vector3::new(0.0, 0.0, PI - 1e-6),
        ] {
            assert_relative_eq!(
                left_jacobian(&w) * left_jacobian_inv(&w),
                Matrix3::identity(),
                epsilon = 1e-12
            );
        }
    }

    // Test the identity that J_r(\omega) = J_l(-\omega), and that J_r is J_l^\top.
    #[test]
    fn right_jacobian_is_transpose() {
        let w = Vector3::new(0.3, 1.2, -0.5);
        assert_relative_eq!(left_jacobian(&-w), left_jacobian(&w).transpose(), epsilon=1e-14);
    }


    //Adjoint test for left jacobian and exponential
    #[test]
    fn adjoint_relation() {
        let w = Vector3::new(-0.8, 0.2, 1.4);
        assert_relative_eq!(
            exp(&w) * left_jacobian(&w).transpose(),
            left_jacobian(&w),
            epsilon = 1e-13
        );
    }

    #[test]
    fn matches_fd() {
        let w = Vector3::new(0.5, -0.3, 0.9);
        let eps = 1e-7;

        for i in 0..3 {
            let mut d = Vector3::zeros();
            d[i] = eps;

            // Numerical: log of the left-multiplied difference
            let numeric = log(&(exp(&(w + d)) * exp(&w).transpose())) / eps;
            let analytic = left_jacobian(&w).column(i).into_owned();

            assert_relative_eq!(numeric, analytic, epsilon = 1e-6);
        }
    }

    // Gamma_2 function testing
    #[test]
    fn gamma_recurrence() {
        for w in [
            Vector3::new(0.4, -0.7, 1.1),
            Vector3::new(1e-9, 0.0, 0.0),
            Vector3::new(0.0, 0.0, PI - 1e-6),
            Vector3::new(0.05, -0.06, 0.07) // straddles the GAMMA2_threshold
        ] {
            let k = hat(&w);
            // m = 0; \phi^\wedge \Gamma_1 = \Gamma_0 - I (check existing Jacobian)
            assert_relative_eq!(
                k * left_jacobian(&w),
                exp(&w) - Matrix3::identity(),
                epsilon = 1e-13
            );
            // m= 1; \phi^\wedge Gamma_2 = \Gamma_1 - I
            assert_relative_eq!(
                k * gamma2(&w),
                left_jacobian(&w) - Matrix3::identity(),
                epsilon = 1e-13
            );
        }
    }

    // also quadrature test, as the null space of \phi^\wedge is the rotation axis,
    // and there would be no way to know whether or not it failed. this ensures that we have the
    // right axis.
    #[test]
    fn gamma2_matches_quadrature() {
        let w = Vector3::new(0.4, -0.9, 1.3);
        let n = 20_000;
        let mut acc = Matrix3::zeros();
        for i in 0..n {
            let s = (i as f64 + 0.5) / n as f64; // midpoint rule, O(h^2)
            acc += (1.0 - s) * exp(&(s * w));
        }
        acc /= n as f64;
        assert_relative_eq!(gamma2(&w),acc, epsilon=1e-9)
    }
}
