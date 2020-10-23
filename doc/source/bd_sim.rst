.. _bd_sim:

Brownian Dynamics Simulations
=============================

Langevin equation for translation
---------------------------------

We consider a discrete chain of :math:`n_{b}` beads with positions :math:`\vec{r}^{(n)}`,
where the bead index :math:`n` runs from 0 to :math:`n_{b}-1`.
We consider a general potential energy :math:`E = E( \{ \vec{r}  \} )`, where :math:`\{ \vec{r}  \}` indicates
the full set of positions.
We define the potential forces on the beads to be

.. math::
    \vec{f}_{E}^{(n)} =  - \frac{\partial E}{\partial \vec{r}^{(n)}}

The Langevin equation for the bead positions is given by

.. math::
    \xi_{r} \frac{\partial \vec{r}^{(n)}}{\partial t} = \vec{f}_{E}^{(n)} + \vec{f}_{B}^{(n)}

The Brownian force :math:`\vec{f}_{B}^{(n)}` is governed by the fluctuation dissipation theorem,
which states that :math:`\vec{f}_{B}^{(n)}` is a
Gaussian-distributed random force
that satisfies

.. math::
    \langle \vec{f}_{B}^{(n)}(t) \rangle & = & 0, \\
    \langle \vec{f}_{B}^{(n)}(t) \vec{f}_{B}^{(n')}(t') \rangle
    & = &
    2 k_{B} T \xi \delta (t - t') \delta_{nn'} \mathbf{I}.

This stochastic differential equation is numerically integrated by
discretizing time in discrete time steps :math:`\delta t`.
Thus, the discrete time at the :math:`j`th time step is
given by :math:`t_{j} = \delta t (j-1)`, where :math:`t_{0} = 0`.
Based on this time discretization, we numerically integrate the Langevin
equation as

.. math::
    \vec{r}^{(n)}_{j+1} =
    \vec{r}^{(n)}_{j} +
    \frac{\delta t}{\xi_{r}} \left(
    \vec{f}_{E_{j}}^{(n)} + \vec{f}_{B_{j}}^{(n)}
    \right)

which suggests an explicit integration algorithm.

The fluctuation dissipation for the discrete time

.. math::
    \langle \vec{f}_{B_{j}}^{(n)} \rangle & = & 0, \\
    \langle \vec{f}_{B_{j}}^{(n)} \vec{f}_{B_{j'}}^{(n')} \rangle
    & = &
    \frac{2 k_{B} T \xi}{\delta t}   \delta_{jj'} \delta_{nn'} \mathbf{I}.

Thus, the value of :math:`\langle \vec{f}_{B_{j}}^{(n)}` is selected from a Gaussian distribution
with variance :math:`\frac{2 k_{B} T \xi}{\delta t}`.

Langevin equation for translation and rotation
----------------------------------------------

We consider a discrete chain of :math:`n_{b}` beads with positions :math:`\vec{r}^{(n)}` and orientation triad :math:`\vec{t}_{i}^{(n)}`,
where the bead index :math:`n` runs from 0 to :math:`n_{b}-1`.
We consider a general potential energy :math:`E = E( \{ \vec{r}, \vec{t}_{i}  \} )`, where :math:`\{ \vec{r}, \vec{t}_{i}  \}` indicates
the full set of positions and orientations.
Since the orientation triad forms an orthonormal basis, the dynamics must maintain the following conditions:

.. math::
    \vec{t}_{i}^{(n)} \cdot \vec{t}_{j}^{(n)} = \delta_{ij}

.. \label{eq:constraint}

for all :math:`i,j` pairs.
These conditions are enforced through Lagrange constraints.  Thus, we define the constrained energy

.. math::
    E_{\lambda} ( \{ \vec{r}, \vec{t}_{i}  \} ) =  E( \{ \vec{r}, \vec{t}_{i}  \} ) -
    \sum_{n=0}^{n_{b}-1} \sum_{i,j = 1}^{3}
    \frac{\lambda_{ij}}{2}
    \left(
    \vec{t}_{i}^{(n)} \cdot \vec{t}_{j}^{(n)} - \delta_{ij}
    \right)

We define the potential forces and torques on the beads to be

.. math::
    \vec{f}_{E}^{(n)} & = & - \frac{\partial E}{\partial \vec{r}^{(n)}} \\
    \vec{\tau}_{E,i}^{(n)} & = & - \frac{\partial E}{\partial \vec{t}_{i}^{(n)}}

The Langevin equation for the bead positions is given by

.. math::
    \xi_{r} \frac{\partial \vec{r}^{(n)}}{\partial t} = \vec{f}_{E}^{(n)} + \vec{f}_{B}^{(n)}

where :math:`\vec{f}_{B}^{(n)}` is a Brownian force (discussed below).

The Langevin equation for the orientation triad must resolve the orthonormal constraints.  This leads to the three Langevin equations

.. math::
    \xi_{t,i} \frac{\partial \vec{t}_{i}^{(n)}}{\partial t}  =  \vec{\tau}_{E,i}^{(n)} + \vec{\tau}_{B,i}^{(n)}
    + \sum_{j=1}^{3} \lambda_{ij} \vec{t}_{j}^{(n)}

The Lagrange constraints are satisfied by setting the time derivative of Eq.~\ref{eq:constraint} = 0,
or we have

.. math::
    \vec{t}_{i}^{(n)} \cdot \frac{\partial \vec{t}_{j}^{(n)}}{\partial t} +
    \vec{t}_{j}^{(n)} \cdot \frac{\partial \vec{t}_{i}^{(n)}}{\partial t} = 0

This leads to the solution to the Lagrange multipliers to be

.. math::
    \lambda_{ij} = - \left( \xi_{t,i} + \xi_{t,j} \right)^{-1}
    \left[
    \xi_{t,i} \vec{t}_{i}^{(n)} \cdot \left( \vec{\tau}_{E,j}^{(n)} + \vec{\tau}_{B,j}^{(n)} \right) +
    \xi_{t,j} \vec{t}_{j}^{(n)} \cdot \left( \vec{\tau}_{E,i}^{(n)} + \vec{\tau}_{B,i}^{(n)} \right)
    \right]

With this development, we now write the Langevin equations as

.. math::
    \xi_{t,1} \frac{\partial \vec{t}_{1}^{(n)}}{\partial t} & = &
    \frac{\xi_{t,1}}{\xi_{t,1}+\xi_{t,2}}
    \left[
    \vec{t}_{2}^{(n)} \cdot \left(  \vec{\tau}_{E,1}^{(n)} + \vec{\tau}_{B,1}^{(n)} \right) -
    \vec{t}_{1}^{(n)} \cdot \left(  \vec{\tau}_{E,2}^{(n)} + \vec{\tau}_{B,2}^{(n)} \right)
    \right] \vec{t}_{2}^{(n)} +
    \nonumber \\
    &  &
    \frac{\xi_{t,1}}{\xi_{t,1}+\xi_{t,3}}
    \left[
    \vec{t}_{3}^{(n)} \cdot \left(  \vec{\tau}_{E,1}^{(n)} + \vec{\tau}_{B,1}^{(n)} \right) -
    \vec{t}_{1}^{(n)} \cdot \left(  \vec{\tau}_{E,3}^{(n)} + \vec{\tau}_{B,3}^{(n)} \right)
    \right] \vec{t}_{3}^{(n)} \\
    \xi_{t,2} \frac{\partial \vec{t}_{2}^{(n)}}{\partial t} & = &
    \frac{\xi_{t,2}}{\xi_{t,1}+\xi_{t,2}}
    \left[
    \vec{t}_{1}^{(n)} \cdot \left(  \vec{\tau}_{E,2}^{(n)} + \vec{\tau}_{B,2}^{(n)} \right) -
    \vec{t}_{2}^{(n)} \cdot \left(  \vec{\tau}_{E,1}^{(n)} + \vec{\tau}_{B,1}^{(n)} \right)
    \right] \vec{t}_{1}^{(n)} +
    \nonumber \\
    &  &
    \frac{\xi_{t,2}}{\xi_{t,2}+\xi_{t,3}}
    \left[
    \vec{t}_{3}^{(n)} \cdot \left(  \vec{\tau}_{E,2}^{(n)} + \vec{\tau}_{B,2}^{(n)} \right) -
    \vec{t}_{2}^{(n)} \cdot \left(  \vec{\tau}_{E,3}^{(n)} + \vec{\tau}_{B,3}^{(n)} \right)
    \right] \vec{t}_{3}^{(n)} \\
    \xi_{t,3} \frac{\partial \vec{t}_{3}^{(n)}}{\partial t} & = &
    \frac{\xi_{t,3}}{\xi_{t,1}+\xi_{t,3}}
    \left[
    \vec{t}_{1}^{(n)} \cdot \left(  \vec{\tau}_{E,3}^{(n)} + \vec{\tau}_{B,3}^{(n)} \right) -
    \vec{t}_{3}^{(n)} \cdot \left(  \vec{\tau}_{E,1}^{(n)} + \vec{\tau}_{B,1}^{(n)} \right)
    \right] \vec{t}_{1}^{(n)} +
    \nonumber \\
    &  &
    \frac{\xi_{t,3}}{\xi_{t,2}+\xi_{t,3}}
    \left[
    \vec{t}_{2}^{(n)} \cdot \left(  \vec{\tau}_{E,3}^{(n)} + \vec{\tau}_{B,3}^{(n)} \right) -
    \vec{t}_{3}^{(n)} \cdot \left(  \vec{\tau}_{E,2}^{(n)} + \vec{\tau}_{B,2}^{(n)} \right)
    \right] \vec{t}_{2}^{(n)}


Brownian dynamics simulations with constrained bond lengths
-----------------------------------------------------------

Overview of bond constraints
Ref [Hinch1994]_