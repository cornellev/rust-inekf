use nalgebra::{Matrix3, Matrix5, SMatrix, SVector, Vector3};
use std::ops::Mul;
use super::so3;

// easy types for state definition
//TODO: need to change to include biases, right?
pub type Vector9 = SVector<f64, 9>;
pub type Matrix9 = SMatrix<f64, 9, 9>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SE23 {
    pub r: Matrix3<f64>,
    pub v: Vector3<f64>,
    pub p: Vector3<f64>,
}

impl SE23 {
    pub fn new(r: Matrix3<f64>, v: Vector3<f64>, p: Vector3<f64>) -> Self {
        Self { r, v, p }
    }

    pub fn identity() -> Self {
        Self::new(Matrix3::identity(), Vector3::zeros(), Vector3::zeros())
    }

    // Closed form inverse: (R^T, -R^T v, -R^T p).
    pub fn inverse(&self) -> Self {
        let rt: Matrix3<f64> = self.r.transpose();
        Self::new(rt, -rt * self.v, -rt * self.p)
    }

    // Transition function from state to lie algebra
    pub fn to_algebra(&self) -> Matrix5<f64> {
        let mut mat = Matrix5::identity();
        mat.fixed_view_mut::<3, 3>(0,0).copy_from(&self.r);
        mat.fixed_view_mut::<3, 1>(0,3).copy_from(&self.v);
        mat.fixed_view_mut::<3, 1>(0,4).copy_from(&self.p);
        mat
    }

    pub fn adjoint(&self) -> Matrix9 {
        let mut a = Matrix9::zeros();
        a.fixed_view_mut::<3, 3>(0, 0).copy_from(&self.r);
        a.fixed_view_mut::<3, 3>(3, 3).copy_from(&self.r);
        a.fixed_view_mut::<3, 3>(6, 6).copy_from(&self.r);
        a.fixed_view_mut::<3, 3>(3, 0).copy_from(&(so3::hat(&self.v) * self.r));
        a.fixed_view_mut::<3, 3>(6, 0).copy_from(&(so3::hat(&self.p) * self.r));
        a
    }
}
impl Mul for SE23 {
    type Output = SE23;

    //standard SE_2(3) group composition
    fn mul(self, rhs: SE23) -> SE23 {
        SE23::new(
            self.r * rhs.r,
            self.r  *rhs.v + self.v,
            self.r * rhs.p + self.p
        )
    }
}

// xi^ = [[phi^, rho_v, rho_p], [0, 0_2x2]]
pub fn hat(xi: &Vector9) -> Matrix5<f64> {
    let mut mat = Matrix5::zeros();
    mat.fixed_view_mut::<3,3>(0,0)
        .copy_from(&so3::hat(&xi.fixed_rows::<3>(0).into_owned())); // rotation block
    mat.fixed_view_mut::<3, 1>(0,3).copy_from(&xi.fixed_rows::<3>(3)); // velocity
    // block
    mat.fixed_view_mut::<3,1>(0,4).copy_from(&xi.fixed_rows::<3>(6));
    mat
}

pub fn vee(mat: &Matrix5<f64>) -> Vector9 {
    let mut xi = Vector9::zeros();
    xi.fixed_rows_mut::<3>(0)
        .copy_from(&so3::vee(&mat.fixed_view::<3,3>(0,0).into_owned()));
    xi.fixed_rows_mut::<3>(3).copy_from(&mat.fixed_view::<3,1>(0,3));
    xi.fixed_rows_mut::<3>(6).copy_from(&mat.fixed_view::<3,1>(0,4));
    xi
}

pub fn exp(xi: &Vector9) -> SE23 {
    let phi: Vector3<f64> = xi.fixed_rows::<3>(0).into_owned();
    let jl: Matrix3<f64> = so3::left_jacobian(&phi);
    SE23::new(
        so3::exp(&phi),
        jl * xi.fixed_rows::<3>(3),
        jl * xi.fixed_rows::<3>(6),
    )
}

pub fn log(x: &SE23) -> Vector9 {
    let phi = so3::log(&x.r);
    let jinv = so3::left_jacobian_inv(&phi);
    let mut xi = Vector9::zeros();
    xi.fixed_rows_mut::<3>(0).copy_from(&phi);
    xi.fixed_rows_mut::<3>(3).copy_from(&(jinv * x.v));
    xi.fixed_rows_mut::<3>(6).copy_from(&(jinv * x.p));
    xi
}

pub fn ad(xi: &Vector9) -> Matrix9 {
    let phi_hat = so3::hat(&xi.fixed_rows::<3>(0).into_owned());
    let mut a = Matrix9::zeros();
    a.fixed_view_mut::<3, 3>(0, 0).copy_from(&phi_hat);
    a.fixed_view_mut::<3, 3>(3, 3).copy_from(&phi_hat);
    a.fixed_view_mut::<3, 3>(6, 6).copy_from(&phi_hat);
    a.fixed_view_mut::<3, 3>(3, 0)
        .copy_from(&so3::hat(&xi.fixed_rows::<3>(3).into_owned()));
    a.fixed_view_mut::<3, 3>(6, 0)
        .copy_from(&so3::hat(&xi.fixed_rows::<3>(6).into_owned()));
    a
}
