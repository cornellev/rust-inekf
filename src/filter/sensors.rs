use nalgebra::Vector3;

//WARNING: this is a placeholder until it is implemented with the rest of `rust-autonomy-stack.`
#[derive(Debug, Clone, Copy)]
pub struct IMU {
    pub gyro: Vector3<f64>, // rad/s
    pub accel: Vector3<f64>, // m/s^2
}
