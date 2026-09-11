use crate::lie::se23::SE23;
use nalgebra::{Vector3, Cholesky, SMatrix, SVector};
use crate::filter::sensors::IMU;

pub const GRAVITY_VECTOR: Vector3<f64> = Vector3::new(0.0, 0.0, -9.81); // m/s^2
pub type Vector15 = SVector<f64, 15>;
pub type Matrix15 = SMatrix<f64, 15,15>;

#[derive(Debug, Clone, Copy, Default)]
pub struct Biases {
    pub gyro: Vector3<f64>,
    pub accel: Vector3<f64>,
}

impl Biases {
    pub fn correct(&self, imu: &IMU) -> IMU {
        IMU { gyro: imu.gyro - self.gyro, accel : imu.accel - self.accel}
    }
}

#[derive(Debug, Clone)]
pub struct CarState {
    pub x: SE23,
    pub b: Biases,
    pub cov: Matrix15,
}

pub struct InitialSigmas {
    pub phi: Vector3<f64>,
    pub vel: Vector3<f64>,
    pub pos: Vector3<f64>,
    pub bias_gyro: Vector3<f64>,
    pub bias_accel: Vector3<f64>,
}

impl CarState {
    // pub fn new(x: SE23, cov: Matrix9) 
    pub fn new(x: SE23, b: Biases, cov: Matrix15) -> Self { Self { x, b, cov }}

    pub fn from_sigmas(x: SE23, b: Biases, s: &InitialSigmas) -> Self {
        let mut sig = Vector15::zeros();
        sig.fixed_rows_mut::<3>(0).copy_from(&s.phi);
        sig.fixed_rows_mut::<3>(3).copy_from(&s.vel);
        sig.fixed_rows_mut::<3>(6).copy_from(&s.pos);
        sig.fixed_rows_mut::<3>(9).copy_from(&s.bias_gyro);
        sig.fixed_rows_mut::<3>(12).copy_from(&s.bias_accel);
        Self{ x, b, cov: Matrix15::from_diagonal(&sig.map(|v| v * v)) }
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
        // Components within a block are deliberately distinct (1x, 2x, 3x) so the
        // test can catch a mis-ordered or transposed copy_from, not just a wrong block.
        let sig_i = InitialSigmas {
            phi:        Vector3::new(1e-4, 2e-4, 3e-4),
            vel:        Vector3::new(1e-2, 2e-2, 3e-2),
            pos:        Vector3::new(1e-1, 2e-1, 3e-1),
            bias_gyro:  Vector3::new(1e-3, 2e-3, 3e-3),
            bias_accel: Vector3::new(1e-2, 2e-2, 3e-2),
        };

        let s = CarState::from_sigmas(any_pose(), Biases::default(), &sig_i);

        let expected = [
            1e-4, 2e-4, 3e-4, // phi
            1e-2, 2e-2, 3e-2, // vel
            1e-1, 2e-1, 3e-1, // pos
            1e-3, 2e-3, 3e-3, // gyro bias
            1e-2, 2e-2, 3e-2, // accel bias
        ];
        for (i, sig) in expected.iter().enumerate() {
            assert_eq!(s.cov[(i, i)], sig * sig, "diagonal {i}");
        }

        // also check the matrix is diagonal
        for i in 0..15 {
            for j in 0..15 {
                if i != j {
                    assert_eq!(s.cov[(i, j)], 0.0, "off-diagonal ({i},{j})");
                }
            }
        }
    }

}
