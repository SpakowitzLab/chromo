.. _poly_models:

Polymer Models
==============

Polymer chain model: shearable, stretchable wormlike chain
----------------------------------------------------------

We consider a polymer with :math:`n_{b}` number of beads.
We consider the shearable, stretchable wormlike chain potential, given by

.. math::
    \beta E_{\mathrm{elas}} = \sum_{n=0}^{n_{b}-2}
    \left[
    \frac{\epsilon_{\mathrm{b}}}{2 \Delta} \left| \vec{t}_{3}^{(n+1)} - \vec{t}_{3}^{(n)} - \eta \Delta \vec{r}_{\perp}^{(n)} \right|^{2} +
    \frac{\epsilon_{\mathrm{\parallel}}}{2 \Delta} \left( \Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)} - \Delta \gamma \right)^{2} +
    \frac{\epsilon_{\mathrm{\perp}}}{2 \Delta} \left| \Delta \vec{r}_{\perp}^{(n)} \right|^{2}
    \right],

where :math:`\Delta \vec{r}^{(n)} = \vec{r}^{(n+1)} - \vec{r}^{(n)}` is the bond vector,
:math:`\Delta \vec{r}_{\perp}^{(n)} = \Delta \vec{r}^{(n)} - (\Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)}) \vec{t}_{3}^{(n)}`
is the perpendicular component of the bond vector to the tangent vector.




Polymer chain model: shearable, stretchable wormlike chain with twist
--------------------------------------------------------------------

We consider a closed ring polymer, where the :math:`n_{b}` bead is the same as the zeroth bead.
We consider the shearable, stretchable wormlike chain potential with twist, given by

.. math::
    \beta E_{\mathrm{elas}} = \sum_{n=0}^{n_{b}-1}
    \left[
    \frac{\epsilon_{\mathrm{b}}}{2 \Delta} \left| \vec{t}_{3}^{(n+1)} - \vec{t}_{3}^{(n)} - \eta \Delta \vec{r}_{\perp}^{(n)} \right|^{2} +
    \frac{\epsilon_{\mathrm{\parallel}}}{2 \Delta} \left( \Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)} - \Delta \gamma \right)^{2} +
    \frac{\epsilon_{\mathrm{\perp}}}{2 \Delta} \left| \Delta \vec{r}_{\perp}^{(n)} \right|^{2} +
    \frac{\epsilon_{\mathrm{t}}}{2 \Delta} \left( \omega^{(n)} \right)^{2}
    \right],

where :math:`\Delta \vec{r}^{(n)} = \vec{r}^{(n+1)} - \vec{r}^{(n)}` is the bond vector,
:math:`\Delta \vec{r}_{\perp}^{(n)} = \Delta \vec{r}^{(n)} - (\Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)}) \vec{t}_{3}^{(n)}` is the
perpendicular component of the bond vector to the tangent vector.

The twist angle :math:`\omega^{(n)}` gives the
local twist deformation of the chain.
Geometrically, this is defined by the relationship

.. math::
    \left( 1 + \vec{t}_{3}^{(n)} \cdot \vec{t}_{3}^{(n+1)} \right) \cos \Omega^{(n)} & = &
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)}  \\
    \left( 1 + \vec{t}_{3}^{(n)} \cdot \vec{t}_{3}^{(n+1)} \right) \sin \Omega^{(n)} & = &
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{1}^{(n+1)} -
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{2}^{(n+1)}

where :math:`\omega^{(n)} = \Omega^{(n)} + 2 \pi m^{(n)}`, and :math:`m^{(n)}` gives the number of additional integer turns
of twist within the :math:`n`th segment.
We write a differential change in :math:`\omega^{(n)}` as

.. math::
    \delta \omega^{(n)} & = &
    \frac{\vec{t}_{1}^{(n+1)} \cdot \delta \vec{t}_{2}^{(n)}}{
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)}
    } -
    \frac{\vec{t}_{2}^{(n+1)} \cdot \delta \vec{t}_{1}^{(n)}}{
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)}
    }
    \nonumber \\
    &  &
    + \frac{\vec{t}_{2}^{(n)} \cdot \delta \vec{t}_{1}^{(n+1)}}{
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)}
    }  -
    \frac{\vec{t}_{1}^{(n)} \cdot \delta \vec{t}_{2}^{(n+1)}}{
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)}
    }  \nonumber \\
    &  &
    -  \left(
    \frac{\vec{t}_{2}^{(n)} \cdot \vec{t}_{1}^{(n+1)}  - \vec{t}_{1}^{(n)} \cdot \vec{t}_{2}^{(n+1)} }
    {\vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)} }
    \right)
    \left(
    \frac{
    \vec{t}_{3}^{(n)} \cdot \delta \vec{t}_{3}^{(n+1)} +
    \vec{t}_{3}^{(n+1)} \cdot \delta \vec{t}_{3}^{(n)}
    }{1 + \vec{t}_{3}^{(n)} \cdot \vec{t}_{3}^{(n+1)} }
    \right)


With this development, we write the torque vectors as

.. math::
    \vec{\tau}_{1}^{(n)} & = & \frac{\epsilon_{t}}{\Delta} \omega^{(n)}
    \left(
    \frac{\vec{t}_{2}^{(n+1)}}{
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)}}
    \right)
    -
    \frac{\epsilon_{t}}{\Delta} \omega^{(n-1)}
    \left(
    \frac{\vec{t}_{2}^{(n-1)}}{
    \vec{t}_{1}^{(n-1)} \cdot \vec{t}_{1}^{(n)} +
    \vec{t}_{2}^{(n-1)} \cdot \vec{t}_{2}^{(n)}}
    \right)
    \\
    \vec{\tau}_{2}^{(n)} & = & - \frac{\epsilon_{t}}{\Delta} \omega^{(n)}
    \left(
    \frac{\vec{t}_{1}^{(n+1)}}{
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)}}
    \right)
    +
    \frac{\epsilon_{t}}{\Delta} \omega^{(n-1)}
    \left(
    \frac{\vec{t}_{1}^{(n-1)} }{
    \vec{t}_{1}^{(n-1)} \cdot \vec{t}_{1}^{(n)} +
    \vec{t}_{2}^{(n-1)} \cdot \vec{t}_{2}^{(n)}}
    \right) \\
    \vec{\tau}_{3}^{(n)} & = &
    \vec{\tau}_{b}^{(n)} -
    \vec{\tau}_{b}^{(n-1)} - \eta \left[
    (\Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)}) \vec{\tau}_{b}^{(n)}
    + ( \vec{\tau}_{b}^{(n)} \cdot \vec{t}_{3}^{(n)} ) \Delta \vec{r}^{(n)}
    \right]
    \nonumber \\
    &  &
    - \frac{\epsilon_{\parallel}}{\Delta}
    \left( \Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)} - \Delta \gamma \right) \Delta \vec{r}^{(n)}
    + \frac{\epsilon_{\perp}}{\Delta}
    (\Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)} ) \Delta \vec{r}_{\perp}^{(n)}
    \nonumber \\
    &  &
    +\frac{\epsilon_{t}}{\Delta} \omega^{(n)}
    \left(
    \frac{\vec{t}_{2}^{(n)} \cdot \vec{t}_{1}^{(n+1)}  - \vec{t}_{1}^{(n)} \cdot \vec{t}_{2}^{(n+1)} }
    {\vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)} }
    \right)
    \frac{
    \vec{t}_{3}^{(n+1)}}{1 + \vec{t}_{3}^{(n)} \cdot \vec{t}_{3}^{(n+1)} } \nonumber \\
    &  &
    +
    \frac{\epsilon_{t}}{\Delta} \omega^{(n-1)}
    \left(
    \frac{\vec{t}_{2}^{(n-1)} \cdot \vec{t}_{1}^{(n)}  - \vec{t}_{1}^{(n-1)} \cdot \vec{t}_{2}^{(n)} }
    {\vec{t}_{1}^{(n-1)} \cdot \vec{t}_{1}^{(n)} +
    \vec{t}_{2}^{(n-1)} \cdot \vec{t}_{2}^{(n)} }
    \right)
    \frac{
    \vec{t}_{3}^{(n-1)}}{1 + \vec{t}_{3}^{(n-1)} \cdot \vec{t}_{3}^{(n)} }

where


.. math::
    \vec{\tau}_{b}^{(n)} =
    \frac{\epsilon_{b}}{\Delta} \left(
    \vec{t}_{3}^{(n+1)} - \vec{t}_{3}^{(n)} - \eta \Delta \vec{r}_{\perp}^{(n)}
    \right)

The force on the :math:`n`th bead is given by

.. math::
    \vec{f}^{(n)} & = &
    -\eta \vec{\tau}_{b}^{(n)} + \eta ( \vec{\tau}_{b}^{(n)} \cdot \vec{t}_{3}^{(n)} ) \vec{t}_{3}^{(n)}
    +\eta \vec{\tau}_{b}^{(n-1)} - \eta ( \vec{\tau}_{b}^{(n-1)} \cdot \vec{t}_{3}^{(n-1)} ) \vec{t}_{3}^{(n-1)}
    \nonumber \\
    &  &
    + \frac{\epsilon_{\parallel}}{\Delta}
    \left( \Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)} - \Delta \gamma \right) \vec{t}_{3}^{(n)}
    - \frac{\epsilon_{\parallel}}{\Delta}
    \left( \Delta \vec{r}^{(n-1)} \cdot \vec{t}_{3}^{(n-1)} - \Delta \gamma \right) \vec{t}_{3}^{(n-1)}
    \nonumber \\
    &  &
    + \frac{\epsilon_{\perp}}{\Delta}
    \Delta \vec{r}_{\perp}^{(n)}
    - \frac{\epsilon_{\perp}}{\Delta}
    \Delta \vec{r}_{\perp}^{(n-1)}

