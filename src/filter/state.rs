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
    pub fn from_sigmaw
}
