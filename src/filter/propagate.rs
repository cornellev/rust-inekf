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
        let sigma_accel = Matrix3::identity() * self.sigma_accel.powi(2);
        q_cont.fixed_view_mut::<3,3>(0,0).copy_from(&sigma_gyro);
        q_cont.fixed_view_mut::<3,3>(3,3).copy_from(&sigma_accel);
        q_cont
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Increments {
    pub g0: Matrix3<f64>,
    pub dv: Vector3<f64>,
    pub dp: Vector3<f64>,
    pub dt: f64
}

impl Increments {
    pub fn new(imu: &IMU, dt: f64) -> Self {
        let phi = imu.gyro * dt;
        Self {
            g0: so3::exp(&phi),
            dv: so3::left_jacobian(&phi) * imu.accel * dt,
            dp: so3::gamma2(&phi) * imu.accel * dt.powi(2),
            dt,
        }
    }
}


pub fn propagate_mean(x: &SE23, inc: &Increments) -> SE23 {
    SE23::new(
        x.r * inc.g0,
        x.v + GRAVITY_VECTOR * inc.dt + x.r * inc.dv,
        x.p + x.v * inc.dt + 0.5 * GRAVITY_VECTOR * inc.dt.powi(2) + x.r * inc.dp,
    )
}

pub fn transition_matrix(inc: &Increments) -> Matrix9 {
    let g0t = inc.g0.transpose();

    let mut f = Matrix9::zeros();
    f.fixed_view_mut::<3,3>(0,0).copy_from(&g0t);
    f.fixed_view_mut::<3,3>(3,3).copy_from(&g0t);
    f.fixed_view_mut::<3,3>(6,6).copy_from(&g0t);
    f.fixed_view_mut::<3,3>(3,0).copy_from(&(-g0t * so3::hat(&inc.dv)));
    f.fixed_view_mut::<3,3>(6,0).copy_from(&(-g0t * so3::hat(&inc.dp)));
    f.fixed_view_mut::<3,3>(6,3).copy_from(&(g0t * inc.dt));
    f
}

fn discrete_noise(phi_mat: &Matrix9, noise: &ProcessNoise, dt: f64) -> Matrix9 {
    phi_mat * noise.continuous() * phi_mat.transpose() * dt
}

impl CarState {
    pub fn propagate(&mut self, imu: &IMU, noise: &ProcessNoise, dt:f64) {
        let inc = Increments::new(imu, dt);
        let f = transition_matrix(&inc);
        self.x = propagate_mean(&self.x, &inc);
        self.cov = f * self.cov * f.transpose() + discrete_noise(&f, noise, dt);
        self.symmetrize();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::lie::se23::tests::mexp;
    use approx::assert_relative_eq;
    use nalgebra::{SMatrix,Vector3};
    use proptest::prelude::*;

    fn any_pose() -> SE23 { SE23::identity()}

    fn any_imu() -> impl Strategy<Value = IMU> {
        (
            prop::array::uniform3(-3.0f64..3.0),
            prop::array::uniform3(-20.0f64..20.0),
        )
            .prop_map(|(w,a)| IMU {
                gyro: Vector3::from_column_slice(&w),
                accel: Vector3::from_column_slice(&a),
            })
    }

    // CT left-invariant error
    fn a_matrix(imu: &IMU) -> Matrix9 {
        let mut a = Matrix9::zeros();
        let w = -so3::hat(&imu.gyro);
        a.fixed_view_mut::<3,3>(0,0).copy_from(&w);
        a.fixed_view_mut::<3,3>(3,3).copy_from(&w);
        a.fixed_view_mut::<3,3>(6,6).copy_from(&w);
        a.fixed_view_mut::<3,3>(3,0).copy_from(&(-so3::hat(&imu.accel)));
        a.fixed_view_mut::<3,3>(6,3).copy_from(&Matrix3::identity());
        a
    }

    proptest! {
        #[test]
        fn phi_is_mexp(imu in any_imu(), dt in 1e-3f64..1e-1) {
            let f = transition_matrix(&Increments::new(&imu, dt));
            assert_relative_eq!(f, mexp(&(a_matrix(&imu) * dt)), epsilon=1e-12);
        }

        // test group affinity
        #[test]
        fn phi_composition(imu in any_imu(), dt in 1e-3f64..1e-1) {
            let full = transition_matrix(&Increments::new(&imu, dt));
            let half = transition_matrix(&Increments::new(&imu, dt / 2.0));
            assert_relative_eq!(full, half * half, epsilon=1e-12);
        }

        #[test]
        fn cov_stays_psd(imu in any_imu(), dt in 12e-3f64..2e-2) {
            let mut s = CarState::from_sigmas(
                any_pose(),
                Vector3::new(1e-2,2e-2,3e-2),
                Vector3::new(0.1, 0.2, 0.3),
                Vector3::new(1.0,2.0,3.0)
            );
            let noise = ProcessNoise { sigma_gyro: 1e-3, sigma_accel: 1e-2 };
            for _ in 0..100 {
                s.propagate(&imu, &noise, dt);
                prop_assert!(s.is_psd());
                prop_assert!(s.cov.iter().all(|x| x.is_finite()));
            }
            prop_assert!(s.cov.trace() >= 0.0); // neeeds to be PD
        }
    }

    #[test]
    fn free_fall_and_stationary() {
        let dt = 0.1;
        let x0 = SE23::identity();

        //falling case
        let imu = IMU { gyro: Vector3::zeros(), accel: Vector3::zeros() };
        let f = propagate_mean(&x0, &Increments::new(&imu, dt));
        assert_relative_eq!(f.v ,GRAVITY_VECTOR * dt, epsilon=1e-15);
        assert_relative_eq!(f.p ,0.5 * GRAVITY_VECTOR * dt.powi(2), epsilon=1e-15);

        //stationary case
        let imu = IMU { gyro: Vector3::zeros(), accel: -GRAVITY_VECTOR};
        let s = propagate_mean(&x0, &Increments::new(&imu, dt));
        assert_relative_eq!(s.v, Vector3::zeros(), epsilon=1e-15);
        assert_relative_eq!(s.p, Vector3::zeros(), epsilon=1e-15);
    }

}
