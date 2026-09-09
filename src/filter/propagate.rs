use crate::filter::state::{CarState, GRAVITY_VECTOR};
use crate::lie::se23::{SE23, Matrix9};
use crate::lie::so3;
use super::sensors::IMU;
use nalgebra::{Matrix3, Vector3};

pub struct ProcessNoise {
    pub sigma_gyro: f64, // rad/s/sqrt(Hz)
    pub sigma_accel: f64, // (m/s^2)/sqrt(Hz)
}

impl ProcessNoise{
    //Q_c = diag(sigma_g^2 I, sigma_a^2 I, 0).
    fn continuous(&self) -> Matrix9 {
        let mut q_cont = Matrix9::zeros();
        let sigma_gyro = Matrix3::identity() * self.sigma_gyro.powi(2);
        let sigma_accel = Matrix3::identity() * self.sigma_gyro.powi(2);
        q_cont.fixed_view_mut::<3,3>(0,0).copy_from(&sigma_gyro);
        q_cont.fixed_view_mut::<3,3>(3,3).copy_from(&sigma_accel);
        q_cont
    }
}

struct Gammas {
    g0: Matrix3<f64>,
    g1: Matrix3<f64>,
    g2: Matrix3<f64>,
}

fn gammas(phi: &Vector3<f64>) -> Gammas {
    Gammas {
        g0: so3::exp(phi),
        g1: so3::left_jacobian(phi),
        g2: so3::gamma2(phi)
    }
}

pub fn propagate_mean(x: &SE23, imu: &IMU, dt: f64) -> SE23 {
    let phi = imu.gyro * dt;
    let g = gammas(&phi);

    SE23::new(
        x.r * g.g0,
        x.v + GRAVITY_VECTOR * dt + x.r * (g.g1 * imu.accel) * dt,
        x.p + x.v * dt + 0.5 * GRAVITY_VECTOR * dt.powi(2) + x.r * (g.g2 * imu.accel) * dt.powi(2)
    )
}
