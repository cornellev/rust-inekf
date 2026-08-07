use nalgebra::Vector3;
use rust_inekf::lie::hat;

fn main() {
    let w = Vector3::new(1.0, 2.0, 3.0);
    println!("{}", hat(&w));
}
