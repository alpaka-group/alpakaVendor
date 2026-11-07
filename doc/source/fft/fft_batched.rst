FFT Batched Transforms
=======================

alpakaVendor supports batched FFT transforms where multiple independent transforms are executed in a single plan.

Batch configuration
-------------------

Set the batch count and byte-distance between batch elements:

.. code-block:: cpp

   auto plan = PlanBuilder<std::complex<float>, Extents<uint32_t, 1u>>{}
       .c2c()
       .extents({N})
       .batch(batchSize)
       .distances(inDistanceBytes, outDistanceBytes)
       .build(device);

Distance semantics
------------------

The distance specifies the number of **bytes** between consecutive batch elements, matching the alpaka pitch convention returned by ``getPitches()``:

- **inDistanceBytes**: Bytes between input batch i and batch i+1
- **outDistanceBytes**: Bytes between output batch i and batch i+1

If distance is 0, it defaults to the physical span of the declared layout: ``extent[0] * stride[0]`` for custom
strides, or ``product(extents) * sizeof(value_type)`` for a contiguous layout.

When a backend library requires element counts, the conversion (dividing by ``sizeof(value_type)``) is performed automatically.

Batched C2C example
-------------------

.. literalinclude:: ../../../doc/code/batched_c2c.cpp
   :language: C++
   :start-after: //! [batched-c2c]
   :end-before: //! [batched-c2c]

.. raw:: html

   <details>
   <summary><strong>Full source: batched_c2c.cpp</strong></summary>

.. literalinclude:: ../../../doc/code/batched_c2c.cpp
   :language: C++
   :caption: batched_c2c.cpp

.. raw:: html

   </details>

Batched R2C/C2R example
-----------------------

.. literalinclude:: ../../../doc/code/batched_r2c.cpp
   :language: C++
   :start-after: //! [batched-r2c]
   :end-before: //! [batched-r2c]

.. raw:: html

   <details>
   <summary><strong>Full source: batched_r2c.cpp</strong></summary>

.. literalinclude:: ../../../doc/code/batched_r2c.cpp
   :language: C++
   :caption: batched_r2c.cpp

.. raw:: html

   </details>

Batched N-D transforms from higher-rank alpaka buffers
------------------------------------------------------

You can batch a rank-``D`` FFT over a rank-``D+1`` alpaka buffer by treating the leading dimension as the batch
dimension and using ``getPitches()`` for byte-strides and byte-distances.

In alpaka, the **last** component (``dim - 1``) is the fast-moving dimension, so ``x`` must be stored last:

- batched 2D FFT: buffer layout ``{batch, y, x}``, plan extents ``{y, x}``
- batched 3D FFT: buffer layout ``{batch, z, y, x}``, plan extents ``{z, y, x}``

Example: batched 2D FFT from a 3D buffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../doc/code/batched_nd.cpp
   :language: C++
   :start-after: //! [batched-2d-from-3d]
   :end-before: //! [batched-2d-from-3d]

Example: batched 3D FFT from a 4D buffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../doc/code/batched_nd.cpp
   :language: C++
   :start-after: //! [batched-3d-from-4d]
   :end-before: //! [batched-3d-from-4d]

.. raw:: html

   <details>
   <summary><strong>Full source: batched_nd.cpp</strong></summary>

.. literalinclude:: ../../../doc/code/batched_nd.cpp
   :language: C++
   :caption: batched_nd.cpp

.. raw:: html

   </details>

Performance notes
-----------------

- Batched transforms are more efficient than individual transforms
- Contiguous batches (distance = product of extents * sizeof(value_type)) perform best
- cuFFT and FFTW both optimize batched execution internally

Validation
----------

- Batch must be >= 1
- Distances must not cause buffer overlaps
- Input/output distances must match the transform layout
