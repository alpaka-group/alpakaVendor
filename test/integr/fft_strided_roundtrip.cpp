/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpakaTest/deviceHelper.hpp>
#include <utility>

#include "../unit/test.hpp"
#include "alpaka/fft.hpp"

using namespace alpakaVendor::test;

namespace
{
    /** Create a row-major pitched 2D/3D view over a flat contiguous allocation.
     *
     * The innermost dimension is contiguous and each outer dimension is padded, so the resulting pitches differ
     * from the contiguous layout derived from the extents.
     */
    template<typename T_Value, alpaka::concepts::Vector T_Extents>
    auto makePaddedView(auto& device, T_Value* data, T_Extents const& extents, auto const& pads)
    {
        using Index = alpaka::GetValueType_t<T_Extents>;
        T_Extents pitches{};
        Index pitch = static_cast<Index>(sizeof(T_Value));
        for(uint32_t i = T_Extents::dim(); i-- > 0u;)
        {
            pitches[i] = pitch;
            pitch *= static_cast<Index>(extents[i]) + static_cast<Index>(pads[i]);
        }
        return std::make_pair(alpaka::makeView(device, data, extents, pitches), pitches);
    }

    constexpr float tolerance = 1.0e-4f;
} // namespace

TEMPLATE_LIST_TEST_CASE("FFT 2D C2C roundtrip with padded pitches", "[integr][fft][2d][c2c][strided]", TestBackends)
{
    using namespace alpaka::fft;
    using Complex = alpaka::math::Complex<float>;
    using Extents2D = Extents<uint32_t, 2u>;

    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        constexpr uint32_t nx = 4u;
        constexpr uint32_t ny = 8u;
        constexpr uint32_t padY = 3u;
        constexpr uint32_t paddedRow = ny + padY;

        auto const extents = Extents2D{nx, ny};
        auto const pads = Extents2D{0u, padY};

        // One flat allocation per buffer; the FFT operates on the padded view.
        auto inStorage = alpaka::onHost::allocUnified<Complex>(device, nx * paddedRow);
        auto tmpStorage = alpaka::onHost::allocUnified<Complex>(device, nx * paddedRow);
        auto outStorage = alpaka::onHost::allocUnified<Complex>(device, nx * paddedRow);

        auto [in, inPitches] = makePaddedView<Complex>(device, inStorage.data(), extents, pads);
        auto [tmp, tmpPitches] = makePaddedView<Complex>(device, tmpStorage.data(), extents, pads);
        auto [out, outPitches] = makePaddedView<Complex>(device, outStorage.data(), extents, pads);

        for(uint32_t ix = 0; ix < nx; ++ix)
            for(uint32_t iy = 0; iy < ny; ++iy)
            {
                auto const linear = ix * ny + iy;
                in.data()[ix * paddedRow + iy] = Complex{float(linear + 1u), float(2u * linear + 1u)};
            }

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex, Extents2D>(extents)
                        .c2c()
                        .strides(inPitches, outPitches)
                        .build(device);

        alpaka::fft::onHost::executeForward(queue, plan, in, tmp);
        alpaka::fft::onHost::executeBackward(queue, plan, tmp, out);
        alpaka::onHost::wait(queue);

        for(uint32_t ix = 0; ix < nx; ++ix)
            for(uint32_t iy = 0; iy < ny; ++iy)
            {
                auto const linear = ix * ny + iy;
                CHECK(
                    out.data()[ix * paddedRow + iy].real()
                    == Catch::Approx(float(linear + 1u) * float(nx * ny)).epsilon(tolerance));
                CHECK(
                    out.data()[ix * paddedRow + iy].imag()
                    == Catch::Approx(float(2u * linear + 1u) * float(nx * ny)).epsilon(tolerance));
            }
    }
}

TEMPLATE_LIST_TEST_CASE("FFT 3D C2C roundtrip with padded pitches", "[integr][fft][3d][c2c][strided]", TestBackends)
{
    using namespace alpaka::fft;
    using Complex = alpaka::math::Complex<float>;
    using Extents3D = Extents<uint32_t, 3u>;

    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        constexpr uint32_t nx = 2u;
        constexpr uint32_t ny = 3u;
        constexpr uint32_t nz = 4u;
        constexpr uint32_t padZ = 2u;
        constexpr uint32_t paddedZ = nz + padZ;
        constexpr uint32_t elements = nx * ny * paddedZ;

        auto const extents = Extents3D{nx, ny, nz};
        auto const pads = Extents3D{0u, 0u, padZ};

        auto inStorage = alpaka::onHost::allocUnified<Complex>(device, elements);
        auto tmpStorage = alpaka::onHost::allocUnified<Complex>(device, elements);
        auto outStorage = alpaka::onHost::allocUnified<Complex>(device, elements);

        auto [in, inPitches] = makePaddedView<Complex>(device, inStorage.data(), extents, pads);
        auto [tmp, tmpPitches] = makePaddedView<Complex>(device, tmpStorage.data(), extents, pads);
        auto [out, outPitches] = makePaddedView<Complex>(device, outStorage.data(), extents, pads);

        auto const idx = [](uint32_t ix, uint32_t iy, uint32_t iz) { return (ix * ny + iy) * paddedZ + iz; };

        for(uint32_t ix = 0; ix < nx; ++ix)
            for(uint32_t iy = 0; iy < ny; ++iy)
                for(uint32_t iz = 0; iz < nz; ++iz)
                {
                    auto const linear = (ix * ny + iy) * nz + iz;
                    in.data()[idx(ix, iy, iz)] = Complex{float(linear + 1u), float(2u * linear + 1u)};
                }

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex, Extents3D>(extents)
                        .c2c()
                        .strides(inPitches, outPitches)
                        .build(device);

        alpaka::fft::onHost::executeForward(queue, plan, in, tmp);
        alpaka::fft::onHost::executeBackward(queue, plan, tmp, out);
        alpaka::onHost::wait(queue);

        constexpr float fftSize = float(nx * ny * nz);
        for(uint32_t ix = 0; ix < nx; ++ix)
            for(uint32_t iy = 0; iy < ny; ++iy)
                for(uint32_t iz = 0; iz < nz; ++iz)
                {
                    auto const linear = (ix * ny + iy) * nz + iz;
                    CHECK(
                        out.data()[idx(ix, iy, iz)].real()
                        == Catch::Approx(float(linear + 1u) * fftSize).epsilon(tolerance));
                    CHECK(
                        out.data()[idx(ix, iy, iz)].imag()
                        == Catch::Approx(float(2u * linear + 1u) * fftSize).epsilon(tolerance));
                }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "FFT 2D R2C/C2R roundtrip with padded pitches",
    "[integr][fft][2d][r2c][c2r][strided]",
    TestBackends)
{
    using namespace alpaka::fft;
    using Complex = alpaka::math::Complex<float>;
    using Extents2D = Extents<uint32_t, 2u>;

    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        constexpr uint32_t nx = 4u;
        constexpr uint32_t ny = 8u;
        constexpr uint32_t padYReal = 2u;
        constexpr uint32_t padYComplex = 1u;

        auto const realExtents = Extents2D{nx, ny};
        auto const complexExtents = r2cComplexExtent(realExtents);
        auto const realPads = Extents2D{0u, padYReal};
        auto const complexPads = Extents2D{0u, padYComplex};
        constexpr uint32_t paddedRealRow = ny + padYReal;
        uint32_t const paddedComplexRow = complexExtents[1u] + padYComplex;

        auto inStorage = alpaka::onHost::allocUnified<float>(device, nx * paddedRealRow);
        auto spectrumStorage = alpaka::onHost::allocUnified<Complex>(device, nx * paddedComplexRow);
        auto outStorage = alpaka::onHost::allocUnified<float>(device, nx * paddedRealRow);

        auto [in, inPitches] = makePaddedView<float>(device, inStorage.data(), realExtents, realPads);
        auto [spectrum, spectrumPitches]
            = makePaddedView<Complex>(device, spectrumStorage.data(), complexExtents, complexPads);
        auto [out, outPitches] = makePaddedView<float>(device, outStorage.data(), realExtents, realPads);

        for(uint32_t ix = 0; ix < nx; ++ix)
            for(uint32_t iy = 0; iy < ny; ++iy)
                in.data()[ix * paddedRealRow + iy] = float(ix * ny + iy + 1u);

        auto r2cPlan = alpaka::fft::onHost::PlanBuilder<float, Extents2D>(realExtents)
                           .r2c()
                           .strides(inPitches, spectrumPitches)
                           .build(device);
        auto c2rPlan = alpaka::fft::onHost::PlanBuilder<float, Extents2D>(realExtents)
                           .c2r()
                           .strides(spectrumPitches, outPitches)
                           .build(device);

        alpaka::fft::onHost::executeForward(queue, r2cPlan, in, spectrum);
        alpaka::fft::onHost::executeBackward(queue, c2rPlan, spectrum, out);
        alpaka::onHost::wait(queue);

        for(uint32_t ix = 0; ix < nx; ++ix)
            for(uint32_t iy = 0; iy < ny; ++iy)
                CHECK(
                    out.data()[ix * paddedRealRow + iy]
                    == Catch::Approx(float(ix * ny + iy + 1u) * float(nx * ny)).epsilon(tolerance));
    }
}

TEMPLATE_LIST_TEST_CASE(
    "FFT batched 2D C2C with padded pitches and default batch distance",
    "[integr][fft][batched][2d][c2c][strided]",
    TestBackends)
{
    using namespace alpaka::fft;
    using Complex = alpaka::math::Complex<float>;
    using Extents2D = Extents<uint32_t, 2u>;
    using Extents3D = Extents<uint32_t, 3u>;

    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        constexpr uint32_t batch = 3u;
        constexpr uint32_t nx = 3u;
        constexpr uint32_t ny = 4u;
        constexpr uint32_t padY = 2u;
        constexpr uint32_t paddedRow = ny + padY;

        auto const transformExtents = Extents2D{nx, ny};
        auto const bufferExtents = Extents3D{batch, nx, ny};
        auto const bufferPads = Extents3D{0u, 0u, padY};

        auto inStorage = alpaka::onHost::allocUnified<Complex>(device, batch * nx * paddedRow);
        auto tmpStorage = alpaka::onHost::allocUnified<Complex>(device, batch * nx * paddedRow);
        auto outStorage = alpaka::onHost::allocUnified<Complex>(device, batch * nx * paddedRow);

        auto [in, inPitches] = makePaddedView<Complex>(device, inStorage.data(), bufferExtents, bufferPads);
        auto [tmp, tmpPitches] = makePaddedView<Complex>(device, tmpStorage.data(), bufferExtents, bufferPads);
        auto [out, outPitches] = makePaddedView<Complex>(device, outStorage.data(), bufferExtents, bufferPads);

        auto const idx = [](uint32_t b, uint32_t ix, uint32_t iy) { return (b * nx + ix) * paddedRow + iy; };

        for(uint32_t b = 0; b < batch; ++b)
            for(uint32_t ix = 0; ix < nx; ++ix)
                for(uint32_t iy = 0; iy < ny; ++iy)
                {
                    auto const linear = (b * nx + ix) * ny + iy;
                    in.data()[idx(b, ix, iy)] = Complex{float(linear + 1u), float(2u * linear + 1u)};
                }

        // No explicit distances: the backend must derive the batch distance from the pitched stride.
        auto plan = alpaka::fft::onHost::PlanBuilder<Complex, Extents2D>(transformExtents)
                        .c2c()
                        .batch(batch)
                        .strides(
                            alpaka::pCast<std::size_t>(inPitches.template rshrink<2u>()),
                            alpaka::pCast<std::size_t>(outPitches.template rshrink<2u>()))
                        .build(device);

        alpaka::fft::onHost::executeForward(queue, plan, in, tmp);
        alpaka::fft::onHost::executeBackward(queue, plan, tmp, out);
        alpaka::onHost::wait(queue);

        constexpr float fftSize = float(nx * ny);
        for(uint32_t b = 0; b < batch; ++b)
            for(uint32_t ix = 0; ix < nx; ++ix)
                for(uint32_t iy = 0; iy < ny; ++iy)
                {
                    auto const linear = (b * nx + ix) * ny + iy;
                    CHECK(
                        out.data()[idx(b, ix, iy)].real()
                        == Catch::Approx(float(linear + 1u) * fftSize).epsilon(tolerance));
                    CHECK(
                        out.data()[idx(b, ix, iy)].imag()
                        == Catch::Approx(float(2u * linear + 1u) * fftSize).epsilon(tolerance));
                }
    }
}
