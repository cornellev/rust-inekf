pub(super) const SMALL_ANGLE: f64 = 1e-8;
const F3_THRESHOLD: f64 = 1.2e-1; 
const F4_THRESHOLD: f64 = 2.0e-1; 

pub(super) fn f1(theta2: f64) -> f64 {
    if theta2 < SMALL_ANGLE * SMALL_ANGLE {
        1.0 - theta2 / 6.0
    } else {
        let theta: f64 = theta2.sqrt();
        theta.sin() / theta
    }
}

pub(super) fn f2(theta2: f64) -> f64 {
    if theta2 < SMALL_ANGLE * SMALL_ANGLE {
        0.5 - theta2 / 24.0
    } else {
        let half = 0.5 * theta2.sqrt();
        let sinc_half = half.sin() / half;
        0.5 * sinc_half * sinc_half
    }
}

pub(super) fn f3(theta2: f64) -> f64 {
    if theta2 < F3_THRESHOLD * F3_THRESHOLD { f3_series(theta2) } else { f3_closed(theta2) }
}

fn f3_series(theta2: f64) -> f64 {
    let theta4: f64 = theta2 * theta2;
    1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0 - theta4 * theta2 / 362880.0
}

fn f3_closed(theta2: f64) -> f64 {
    let theta: f64 = theta2.sqrt();
    (theta - theta.sin()) / (theta * theta2)
}

pub(super) fn f4(theta2: f64) -> f64 {
    if theta2 < F4_THRESHOLD * F4_THRESHOLD { f4_series(theta2) } else { f4_closed(theta2) }
}

fn f4_series(theta2: f64) -> f64 {
    let theta4: f64 = theta2 * theta2;
    1.0 / 24.0 - theta2 / 720.0 + theta4 / 40320.0 - theta4 * theta2 / 3628800.0
}

fn f4_closed(theta2: f64) -> f64 {
    let theta: f64 = theta2.sqrt();
    (theta2 + 2.0 * theta.cos() - 2.0) / (2.0 * theta2 * theta2)
}
