use nalgebra::Vector3;
use rust_inekf::lie::so3;

fn main() {
    let w = Vector3::new(1.0, 2.0, 3.0);
    println!("{}", so3::hat(&w));
}
