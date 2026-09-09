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
