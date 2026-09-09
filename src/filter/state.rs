use crate::lie::se23::{SE23, Matrix9, Vector9};
use nalgebra::{Vector3, Cholesky};

pub const GRAVITY_VECTOR: Vector3<f64> = Vector3::new(0.0, 0.0, -9.81); // m/s^2

#[derive(Debug, Clone)]
pub struct CarState {
    pub x: SE23,
    pub cov: Matrix9,
}

impl CarState {
    // pub fn new(x: SE23, cov: Matrix9) 
    pub fn new(x: SE23, cov: Matrix9) -> Self { Self { x, cov }}
    pub fn from_sigmas(x: SE23, sigma_phi: Vector3<f64>, sigma_vel: Vector3<f64>, sigma_pos:Vector3<f64>) -> Self {
        let mut sigmas = Vector9::zeros();
        sigmas.fixed_rows_mut::<3>(0).copy_from(&sigma_phi);
        sigmas.fixed_rows_mut::<3>(3).copy_from(&sigma_vel);
        sigmas.fixed_rows_mut::<3>(6).copy_from(&sigma_pos);
        Self::new(x, Matrix9::from_diagonal(&sigmas.map(|s| s * s)))
    }
    pub fn symmetrize(&mut self) {
        self.cov = self.cov.symmetric_part();
    }
    pub fn is_psd(&self) -> bool {
        Cholesky::new(self.cov).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    // use proptest::prelude::*;

    fn any_pose() -> SE23 { SE23::identity()}

    #[test]
    fn from_sigmas_diagonal_check() {
        let s = CarState::from_sigmas(
            any_pose(),
    Vector3::new(1e-2,2e-2,3e-2),
    Vector3::new(0.1, 0.2, 0.3),
    Vector3::new(1.0,2.0,3.0)
        );
        let expected = [1e-2, 2e-2, 3e-2, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0];
        for (i, sig) in expected.iter().enumerate() {
            assert_eq!(s.cov[(i,i)], sig * sig, "diagonal {i}");
        }

        // also check the matrix is diagonal
        for i in 0..9 {
            for j in 0..9 {
                if i !=j {
                    assert_eq!(s.cov[(i,j)], 0.0, "off-diagonal ({i},{j})");
                }
            }
        }
    }

}
