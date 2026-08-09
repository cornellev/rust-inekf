use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;

const SMALL_ANGLE: f64 = 1e-8;

pub fn hat(w: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -w.z, w.y,
        w.z, 0.0, -w.x,
        -w.y, w.x, 0.0,
    )
}

pub fn vee(m: &Matrix3<f64>) -> Vector3<f64> {
    // The vee operator goes from lie algebra to standard vector space.
    Vector3::new(m[(2,1)], m[(0,2)], m[(1,0)])
}

pub fn exp(w: &Vector3<f64>) -> Matrix3<f64> {
    let theta2 = w.norm_squared();
    let k = hat(w);

    let (a,b) = if theta2 < SMALL_ANGLE + SMALL_ANGLE {
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


#[cfg(test)]
mod tests {

use super::*;
    use approx::assert_relative_eq;

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


}
