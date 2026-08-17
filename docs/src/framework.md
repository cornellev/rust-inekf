# `rust-inekf` Framework

## State and Error

Define the state of the vehicle on $SE_2(3)$, as:

$$
X =
\begin{bmatrix}
  \rotmat{W}{B} & \fv{W}{v}_{\mathcal{B}} & \fv{W}{p}_{\mathcal{B}} \\
\mathbf{0}_{1\times3} & 1 & 0 \\
\mathbf{0}_{1\times3} & 0 & 1 \\
\end{bmatrix} \in SE_2(3),
\quad \boldsymbol{\theta} = \begin{bmatrix} \mathbf{b}_g \\ \mathbf{b}_a \end{bmatrix} \in \mathbb{R}^6
$$

And we order the error vector $\bsxi \in \mathbb{R}^{15}$ such that:

$$
\bsxi = \begin{bmatrix} \bsxi_R & \bsxi_v & \bsxi_p & \boldsymbol{\zeta}_g & \boldsymbol{\zeta}_a \end{bmatrix}
$$

and this is with the state component of $\bsxi = \log(\bseta^r)^\vee)$, and $\boldsymbol{\zeta} = \hat{\boldsymbol{\theta}} - \boldsymbol{\theta}$. 

We also define the Adjoint on the manifold to be:

$$
\mathrm{Ad}_X = 
\begin{bmatrix}
  \rotmat{W}{B} & 0 & 0 \\
  \fv{W}{v}_{\mathcal{B}}^\vee \rotmat{W}{B} & \rotmat{W}{B} & 0 \\
  \fv{W}{p}_{\mathcal{B}}^\vee \rotmat{W}{B} & 0 & \rotmat{W}{B} \\
\end{bmatrix}, \quad \widehat{\mathrm{Ad}}_X =
\begin{bmatrix}
  \mathrm{Ad}_X & 0 \\
  0 & I_6
\end{bmatrix}
$$

## Propagation
The continuous time dynamics of the system are defined as:

$$
\begin{aligned}
  \dot{\rotmat{W}{B}} &= \rotmat{W}{B}\left(\tilde{\bsomega} - \mathbf{b}_g \right)^\vee \\
  \dfv{W}{v}_{\mathcal{B}} &= \rotmat{W}{B}\left(\tbf{a} - \mathbf{b}_a \right) + \fv{W}{g} \\
  \dfv{W}{p}_{\mathcal{B}} &= \fv{W}{v}_{\mathcal{B}}
\end{aligned}
$$
