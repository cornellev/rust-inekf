use nalgebra::{Matrix3, Vector3};

pub fn hat(w: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -w.z, w.y,
        w.z, 0.0, -w.x,
        -w.y, w.x, 0.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hat_matches_cross_product() {
        let w = Vector3::new(0.3, -1.2, 0.7);
        let v = Vector3::new(0.3, -1.2, 0.7);
        assert_relative_eq!(hat(&w) * v, w.cross(&v), epsilon=1e-12);
    }
}
