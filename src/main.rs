use nalgebra::{Vector3, Matrix3, Rotation3};
use rust_inekf::lie::se23::SE23;
use std::f64::consts::PI;

fn main() {
    let axis = Vector3::x_axis();
    let rot_mat: Matrix3<f64> = Rotation3::from_axis_angle(&axis, PI/2.0).into_inner();

    let vel: Vector3<f64> = Vector3::new(1.0, 2.0, 0.0);
    let pos: Vector3<f64> = Vector3::new(0.0, 0.0, 5.0);

    // Print parameters
    println!("Rotation matrix:{:.2}", rot_mat);
    println!("Velocity: {:.2}", vel);
    println!("Position: {:.2}", pos);

    // Construct SE23 object
    let state: SE23 = SE23::new(
        rot_mat,
        vel,
        pos
    );

    println!("State:{:.2}",state);

}
