FFT Backends
=============

alpakaVendor supports multiple FFT backends through alpaka's accelerator abstraction.

Memory layout convention
------------------------

alpakaVendor follows alpaka's memory layout convention: **the last index is the fast-moving index** (row-major order).

For a 2D M x N transform:

.. code-block:: cpp

   // M rows, N columns
   // Index 0 (M): slow-moving (row)
   // Index 1 (N): fast-moving (column)
   auto extents = alpaka::Vec<uint32_t, 2u>{M, N};

   // Linear index: row * N + col
   data[row * N + col] = value;

This means:

- For ``Extents<uint32_t, 2u>{M, N}``, the last dimension (N) is contiguous in memory
- For ``Extents<uint32_t, 3u>{D, H, W}``, the last dimension (W) is contiguous in memory

When copying your own data into alpaka buffers, ensure the fast-moving index corresponds to the contiguous dimension.

Backend activation
------------------

Backends are enabled in CMake. The C++ user code stays the same.

FFTW host backend
+++++++++++++++++

.. code-block:: bash

   cmake .. -DalpakaV_DEP_FFTW=ON

cuFFT CUDA backend
++++++++++++++++++

.. code-block:: bash

   cmake .. -DalpakaV_DEP_CUFFT=ON -Dalpaka_DEP_CUDA=ON -Dalpaka_CUDA_NvidiaGpu=ON

rocFFT HIP backend
++++++++++++++++++

.. code-block:: bash

   cmake .. -DalpakaV_DEP_ROCFFT=ON -Dalpaka_DEP_HIP=ON -Dalpaka_HIP_AmdGpu=ON

oneMKL oneAPI backend
+++++++++++++++++++++

.. code-block:: bash

   cmake .. -DCMAKE_CXX_COMPILER=icpx -Dalpaka_DEP_ONEAPI=ON -Dalpaka_ONEAPI_Cpu=ON -DalpakaV_DEP_ONEMKL=ON

Backend-agnostic C++ usage
--------------------------

The documentation examples are instantiated for all available backends:

.. literalinclude:: ../../../doc/code/quickstart_c2c.cpp
   :language: C++
   :start-after: //! [quickstart-c2c-backends]
   :end-before: //! [quickstart-c2c-backends]

Once the device is selected, plan creation is backend-agnostic and execution uses any queue on that device:

.. literalinclude:: ../../../doc/code/quickstart_c2c.cpp
   :language: C++
   :start-after: //! [quickstart-c2c-core]
   :end-before: //! [quickstart-c2c-core]

Validation
----------

Backends validate configuration at plan creation time:

- Dimension must be 1, 2, or 3
- Batch must be >= 1
- Extents must be non-zero
- Complex value type only supports C2C
- Real value type only supports R2C/C2R
- Custom strides must describe a row-major pitched layout: the innermost dimension has a uniform element stride
  (it may be interleaved) and each outer byte stride is a multiple of the next inner byte stride. Each physical
  pitch must also be at least as large as the corresponding extent. This includes padded alpaka views whose row
  pitch is larger than ``extent * sizeof(value_type)``. The host FFTW backend implements this with
  ``fftw_plan_many`` and the per-dimension strides from ``getPitches()``.

Error handling
++++++++++++++

Invalid configurations throw ``std::invalid_argument``:

.. code-block:: cpp

   try {
       auto plan = PlanBuilder<float, Extents<uint32_t, 1u>>{}
           .r2c()
           .extents({0u})  // Invalid: zero extent
           .build(device);
   } catch (const std::invalid_argument& e) {
       std::cerr << "Error: " << e.what() << std::endl;
   }
