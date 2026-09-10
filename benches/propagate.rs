use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::Vector3;
use rust_inekf::filter::propagate::{propagate_mean, transition_matrix, Increments, ProcessNoise};
use rust_inekf::filter::sensors::IMU;
use rust_inekf::filter::state::CarState;
use rust_inekf::lie::{se23::SE23, so3};
use std::hint::black_box;

fn setup() -> (CarState, IMU, ProcessNoise) {
    let x = SE23::new(
        so3::exp(&Vector3::new(0.3,-0.7,1.1)),
        Vector3::new(2.0,-1.0,0.5),
        Vector3::new(-3.0,4.0,1.0),
    );

    let state = CarState::from_sigmas(
        x,
        Vector3::new(1e-2,1e-2,1e-2),
        Vector3::new(0.1,0.1,0.1),
        Vector3::new(1.0,1.0,1.0),
    );
    let imu = IMU {
        gyro: Vector3::new(0.05,-0.02,0.4),
        accel: Vector3::new(1.5,-0.3,-9.0)
    };
    let noise = ProcessNoise { sigma_gyro: 1e-3, sigma_accel: 1e-2 };
    (state, imu, noise)
}

fn bench(c: &mut Criterion) {
    let dt = 1.0 / 200.0; // 200 Hz

    let (mut state, imu, noise) = setup();
    c.bench_function("propagate/full", |b| {
        b.iter(|| state.propagate(black_box(&imu),black_box(&noise),black_box(dt)))
    });

    c.bench_function("propagate/increments", |b| {
        b.iter(|| black_box(Increments::new(black_box(&imu),black_box(dt))))
    });

    let inc = Increments::new(&imu, dt);
    let (base, _, _) = setup();

    c.bench_function("propagate/mean_only", |b| {
        b.iter(|| black_box(propagate_mean(black_box(&base.x), black_box(&inc))))
    });

    c.bench_function("propagate/transition_matrix", |b| {
        b.iter(|| black_box(transition_matrix(black_box(&inc))))
    });

    let f = transition_matrix(&inc);
    c.bench_function("propagate/cov_only", |b| {
        b.iter(|| black_box(black_box(&f) * black_box(&base.cov) * black_box(&f).transpose()))
    });

}

criterion_group!(benches, bench);
criterion_main!(benches);
