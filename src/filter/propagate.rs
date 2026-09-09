use crate::filter::state::{CarState, GRAVITY_VECTOR};
use crate::lie::se23::{SE23, Matrix9};
use crate::lie::so3;
use super::sensors::IMU;
use nalgebra::{Matrix3, Vector3};

//NOTE: gamma computed multiple times, should only be done once via data structure and passed down.

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

pub fn transition_matrix(imu: &IMU, dt:f64) -> Matrix9 {
    let phi = imu.gyro * dt;
    let g = gammas(&phi);
    let g0t = g.g0.transpose();

    // Body frame velocity & accel
    let body_vel = g.g1 * imu.accel * dt;
    let body_pos = g.g2 * imu.accel * dt.powi(2);

    let mut f = Matrix9::zeros();
    f.fixed_view_mut::<3,3>(0,0).copy_from(&g0t);
    f.fixed_view_mut::<3,3>(3,3).copy_from(&g0t);
    f.fixed_view_mut::<3,3>(6,6).copy_from(&g0t);
    f.fixed_view_mut::<3,3>(3,0).copy_from(&(-g0t * so3::hat(&body_vel)));
    f.fixed_view_mut::<3,3>(6,0).copy_from(&(-g0t * so3::hat(&body_pos)));
    f.fixed_view_mut::<3,3>(6,3).copy_from(&(g0t * dt));
    f
}

fn discrete_noise(phi_mat: &Matrix9, noise: &ProcessNoise, dt: f64) -> Matrix9 {
    phi_mat * noise.continuous() * phi_mat.transpose() * dt
}

impl CarState {
    pub fn propagate(&mut self, imu: &IMU, noise: &ProcessNoise, dt:f64) {
        let f = transition_matrix(imu, dt);
        self.x = propagate_mean(&self.x, imu, dt);
        self.cov = f * self.cov * f.transpose() + discrete_noise(&f, noise, dt);
        self.symmetrize();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use::nalgebra::Vector3;
    use approx::assert_relative_eq;
    #[test]
    fn zoh_composes() {
        let x0 = SE23::new(
            so3::exp(&Vector3::new(0.3, -0.7, 1.1)),
            Vector3::new(2.0,-1.0,0.5),
            Vector3::new(-3.0,4.0,1.0),
        );
        let accel = Vector3::new(1.5,-0.3,-9.0);

        for gyro in [
            Vector3::zeros(),
            Vector3::new(1e-12,0.0,0.0),
            Vector3::new(0.0,0.0,2.0),
            Vector3::new(0.0, 0.0, 3.33),
        ] {
            let imu = IMU { gyro, accel };
            let dt = 0.06;
            let one = propagate_mean(&x0, &imu, dt);
            let mut many = x0;
            for _ in 0..64 {
                many = propagate_mean(&many, &imu, dt / 64.0);
            }
            assert_relative_eq!(
                one.to_matrix(), many.to_matrix(), epsilon=1e-10
            );
        }
    }
}
