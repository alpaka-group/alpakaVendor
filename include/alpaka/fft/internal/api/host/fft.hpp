/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/fft/internal/api/config.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#if ALPAKAV_HAS_FFTW
namespace alpaka::fft::internal
{
    template<typename T>
    struct FftwTraits;

    template<>
    struct FftwTraits<float>
    {
        using plan_type = fftwf_plan;
        using complex_type = fftwf_complex;
        static constexpr unsigned flags = FFTW_ESTIMATE;
    };

    template<>
    struct FftwTraits<double>
    {
        using plan_type = fftw_plan;
        using complex_type = fftw_complex;
        static constexpr unsigned flags = FFTW_ESTIMATE;
    };

    template<typename T_Value, alpaka::concepts::Vector T_Extents>
    struct PlanImpl<alpaka::api::Host, T_Value, T_Extents>
    {
        using value_type = T_Value;
        using real_type = Real_t<T_Value>;
        using traits = FftwTraits<real_type>;
        using plan_type = typename traits::plan_type;

        struct PlanState
        {
            plan_type m_forwardPlan = nullptr;
            plan_type m_inversePlan = nullptr;
            std::mutex m_planMutex{};

            ~PlanState()
            {
                destroyPlan(m_forwardPlan);
                if(m_inversePlan != m_forwardPlan)
                    destroyPlan(m_inversePlan);
            }

        private:
            static void destroyPlan(plan_type& plan) noexcept
            {
                if(plan == nullptr)
                    return;
                if constexpr(std::same_as<real_type, float>)
                    fftwf_destroy_plan(plan);
                else
                    fftw_destroy_plan(plan);
                plan = nullptr;
            }
        };

        Transform m_transform;
        static constexpr uint32_t T_dim = T_Extents::dim();
        using index_type = alpaka::GetValueType_t<T_Extents>;

        Layout<T_Extents> m_layout;
        PlanOptions m_options;
        std::shared_ptr<PlanState> m_state;

        PlanImpl(auto&, Transform transform, Layout<T_Extents> layout, PlanOptions options)
            : m_transform{transform}
            , m_layout{layout}
            , m_options{options}
            , m_state{std::make_shared<PlanState>()}
        {
            validateConfig();
        }

        PlanImpl(PlanImpl&& other) noexcept
            : m_transform{other.m_transform}
            , m_layout{other.m_layout}
            , m_options{other.m_options}
            , m_state{std::move(other.m_state)}
        {
            std::lock_guard lock(other.m_pendingWaitsMutex);
            m_pendingWaits = std::move(other.m_pendingWaits);
        }

        PlanImpl& operator=(PlanImpl&& other) noexcept
        {
            if(this != &other)
            {
                m_transform = other.m_transform;
                m_layout = other.m_layout;
                m_options = other.m_options;
                m_state = std::move(other.m_state);
                {
                    std::scoped_lock lock(m_pendingWaitsMutex, other.m_pendingWaitsMutex);
                    m_pendingWaits = std::move(other.m_pendingWaits);
                }
            }
            return *this;
        }

        ~PlanImpl()
        {
            waitForPending();
        }

        void validateConfig() const
        {
            validate(T_dim >= 1u && T_dim <= 3u, "FFT only supports dimensions 1..3.");
            validate(m_layout.batch >= 1u, "FFT batch must be >= 1.");
            for(uint32_t i = 0u; i < T_dim; ++i)
                validate(m_layout.extents[i] > static_cast<index_type>(0u), "FFT extents must be non-zero.");
            validate(m_options.normalization == Normalization::none, "Only Normalization::none is supported.");
            if(m_options.workspacePolicy == WorkspacePolicy::userProvided)
                throw std::invalid_argument("FFTW backend does not support user-provided workspace in v1.");
            if constexpr(ComplexScalar<T_Value>)
                validate(m_transform == Transform::c2c, "Complex plan value type only supports C2C.");
            else
                validate(m_transform != Transform::c2c, "Real plan value type only supports R2C/C2R.");

            validateStrides(
                m_layout.inStrides,
                inputElementBytes<T_Value>(m_transform),
                expectedInputExtents(m_layout, m_transform, m_options.placement),
                "input");
            validateStrides(
                m_layout.outStrides,
                outputElementBytes<T_Value>(m_transform),
                expectedOutputExtents(m_layout, m_transform, m_options.placement),
                "output");
        }

        /** Validate user-provided byte-strides for the FFTW backend.
         *
         * FFTW's `plan_many` interface describes a transform inside a row-major pitched array: the innermost
         * dimension has a single element stride (it may be interleaved) and every outer stride is an integer
         * multiple of the next inner stride. This matches alpaka's pitch convention, where only the row pitch may
         * include padding.
         */
        static void validateStrides(
            alpaka::Vec<std::size_t, T_dim> const& byteStrides,
            std::size_t elementBytes,
            alpaka::concepts::Vector auto const& logicalExtents,
            char const* what)
        {
            if(areZero(byteStrides))
                return;
            for(uint32_t i = 0u; i < T_dim; ++i)
            {
                validate(byteStrides[i] != 0u, std::string{"FFT "} + what + " stride must be non-zero.");
                validate(
                    byteStrides[i] % elementBytes == 0u,
                    std::string{"FFT "} + what + " stride must be a multiple of the element size.");
            }
            auto const elemStrides = stridesToElements<index_type>(byteStrides, elementBytes);
            // Throws unless the strides describe a row-major pitched layout.
            auto const embeds = embedsFromStrides<index_type>(elemStrides, logicalExtents);
            for(uint32_t i = 1u; i < T_dim; ++i)
                validate(
                    embeds[i] >= static_cast<index_type>(logicalExtents[i]),
                    std::string{"FFT "} + what + " stride is too small for its extent.");
        }

        [[nodiscard]] auto n() const
        {
            std::array<int, T_dim> nVals{};
            for(uint32_t i = 0u; i < T_dim; ++i)
                nVals[i] = static_cast<int>(m_layout.extents[i]);
            return nVals;
        }

        [[nodiscard]] auto inEmbed() const
        {
            auto const ext = expectedInputExtents(m_layout, m_transform, m_options.placement);
            auto const embeds = embedsFromStrides<index_type>(
                stridesToElements<index_type>(
                    resolvedInStrides<T_Value>(m_layout, m_transform, m_options.placement),
                    inputElementBytes<T_Value>(m_transform)),
                ext);
            std::array<int, T_dim> vals{};
            for(uint32_t i = 0u; i < T_dim; ++i)
                vals[i] = static_cast<int>(embeds[i]);
            return vals;
        }

        [[nodiscard]] auto outEmbed() const
        {
            auto const ext = expectedOutputExtents(m_layout, m_transform, m_options.placement);
            auto const embeds = embedsFromStrides<index_type>(
                stridesToElements<index_type>(
                    resolvedOutStrides<T_Value>(m_layout, m_transform, m_options.placement),
                    outputElementBytes<T_Value>(m_transform)),
                ext);
            std::array<int, T_dim> vals{};
            for(uint32_t i = 0u; i < T_dim; ++i)
                vals[i] = static_cast<int>(embeds[i]);
            return vals;
        }

        /** Element stride of the innermost dimension, as required by FFTW's `plan_many` interface. */
        [[nodiscard]] auto inStride() const
        {
            auto const strides = stridesToElements<int>(
                resolvedInStrides<T_Value>(m_layout, m_transform, m_options.placement),
                inputElementBytes<T_Value>(m_transform));
            return strides[T_dim - 1u];
        }

        [[nodiscard]] auto outStride() const
        {
            auto const strides = stridesToElements<int>(
                resolvedOutStrides<T_Value>(m_layout, m_transform, m_options.placement),
                outputElementBytes<T_Value>(m_transform));
            return strides[T_dim - 1u];
        }

        /** Element distance between consecutive batches.
         *
         * Without an explicit distance the physical span of the pitched layout is used. For a contiguous layout
         * this is identical to `product(extents) * elementSize`; for a pitched layout it is `extent[0] * stride[0]`,
         * which accounts for the row padding.
         */
        [[nodiscard]] auto inDistance() const
        {
            auto const ext = expectedInputExtents(m_layout, m_transform, m_options.placement);
            auto const stride0 = resolvedInStrides<T_Value>(m_layout, m_transform, m_options.placement)[0u];
            auto const bytes = m_layout.inDistance != 0u
                                   ? m_layout.inDistance
                                   : static_cast<std::size_t>(ext[0u]) * static_cast<std::size_t>(stride0);
            return distanceToElements<int>(bytes, inputElementBytes<T_Value>(m_transform));
        }

        [[nodiscard]] auto outDistance() const
        {
            auto const ext = expectedOutputExtents(m_layout, m_transform, m_options.placement);
            auto const stride0 = resolvedOutStrides<T_Value>(m_layout, m_transform, m_options.placement)[0u];
            auto const bytes = m_layout.outDistance != 0u
                                   ? m_layout.outDistance
                                   : static_cast<std::size_t>(ext[0u]) * static_cast<std::size_t>(stride0);
            return distanceToElements<int>(bytes, outputElementBytes<T_Value>(m_transform));
        }

        [[nodiscard]] static unsigned planningFlags()
        {
            return traits::flags | FFTW_UNALIGNED;
        }

        void createPlans(void* primary, void* secondary)
        {
            auto nVals = n();
            auto inEmbedVals = inEmbed();
            auto outEmbedVals = outEmbed();
            int const inStrideValue = inStride();
            int const outStrideValue = outStride();
            auto const inDistanceValue = inDistance();
            auto const outDistanceValue = outDistance();
            int batch = static_cast<int>(m_layout.batch);

            if constexpr(ComplexScalar<T_Value>)
            {
                auto* inPtr = reinterpret_cast<typename traits::complex_type*>(primary);
                auto* outPtr = reinterpret_cast<typename traits::complex_type*>(
                    m_options.placement == Placement::inPlace ? primary : secondary);
                if constexpr(std::same_as<real_type, float>)
                {
                    m_state->m_forwardPlan = fftwf_plan_many_dft(
                        static_cast<int>(T_dim),
                        nVals.data(),
                        batch,
                        inPtr,
                        inEmbedVals.data(),
                        inStrideValue,
                        inDistanceValue,
                        outPtr,
                        outEmbedVals.data(),
                        outStrideValue,
                        outDistanceValue,
                        FFTW_FORWARD,
                        planningFlags());
                    m_state->m_inversePlan = fftwf_plan_many_dft(
                        static_cast<int>(T_dim),
                        nVals.data(),
                        batch,
                        inPtr,
                        inEmbedVals.data(),
                        inStrideValue,
                        inDistanceValue,
                        outPtr,
                        outEmbedVals.data(),
                        outStrideValue,
                        outDistanceValue,
                        FFTW_BACKWARD,
                        planningFlags());
                }
                else
                {
                    m_state->m_forwardPlan = fftw_plan_many_dft(
                        static_cast<int>(T_dim),
                        nVals.data(),
                        batch,
                        inPtr,
                        inEmbedVals.data(),
                        inStrideValue,
                        inDistanceValue,
                        outPtr,
                        outEmbedVals.data(),
                        outStrideValue,
                        outDistanceValue,
                        FFTW_FORWARD,
                        planningFlags());
                    m_state->m_inversePlan = fftw_plan_many_dft(
                        static_cast<int>(T_dim),
                        nVals.data(),
                        batch,
                        inPtr,
                        inEmbedVals.data(),
                        inStrideValue,
                        inDistanceValue,
                        outPtr,
                        outEmbedVals.data(),
                        outStrideValue,
                        outDistanceValue,
                        FFTW_BACKWARD,
                        planningFlags());
                }
                validate(m_state->m_forwardPlan != nullptr, "FFTW forward plan creation failed.");
                validate(m_state->m_inversePlan != nullptr, "FFTW backward plan creation failed.");
            }
            else if(m_transform == Transform::r2c)
            {
                auto* inPtr = reinterpret_cast<real_type*>(primary);
                auto* outPtr = reinterpret_cast<typename traits::complex_type*>(
                    m_options.placement == Placement::inPlace ? primary : secondary);
                if constexpr(std::same_as<real_type, float>)
                    m_state->m_forwardPlan = fftwf_plan_many_dft_r2c(
                        static_cast<int>(T_dim),
                        nVals.data(),
                        batch,
                        inPtr,
                        inEmbedVals.data(),
                        inStrideValue,
                        inDistanceValue,
                        outPtr,
                        outEmbedVals.data(),
                        outStrideValue,
                        outDistanceValue,
                        planningFlags());
                else
                    m_state->m_forwardPlan = fftw_plan_many_dft_r2c(
                        static_cast<int>(T_dim),
                        nVals.data(),
                        batch,
                        inPtr,
                        inEmbedVals.data(),
                        inStrideValue,
                        inDistanceValue,
                        outPtr,
                        outEmbedVals.data(),
                        outStrideValue,
                        outDistanceValue,
                        planningFlags());
                validate(m_state->m_forwardPlan != nullptr, "FFTW plan creation failed.");
            }
            else
            {
                auto* inPtr = reinterpret_cast<typename traits::complex_type*>(primary);
                auto* outPtr
                    = reinterpret_cast<real_type*>(m_options.placement == Placement::inPlace ? primary : secondary);
                if constexpr(std::same_as<real_type, float>)
                    m_state->m_forwardPlan = fftwf_plan_many_dft_c2r(
                        static_cast<int>(T_dim),
                        nVals.data(),
                        batch,
                        inPtr,
                        inEmbedVals.data(),
                        inStrideValue,
                        inDistanceValue,
                        outPtr,
                        outEmbedVals.data(),
                        outStrideValue,
                        outDistanceValue,
                        planningFlags());
                else
                    m_state->m_forwardPlan = fftw_plan_many_dft_c2r(
                        static_cast<int>(T_dim),
                        nVals.data(),
                        batch,
                        inPtr,
                        inEmbedVals.data(),
                        inStrideValue,
                        inDistanceValue,
                        outPtr,
                        outEmbedVals.data(),
                        outStrideValue,
                        outDistanceValue,
                        planningFlags());
                validate(m_state->m_forwardPlan != nullptr, "FFTW plan creation failed.");
            }
        }

        void ensurePlansCreated(void* primary, void* secondary)
        {
            std::lock_guard lock(m_state->m_planMutex);
            if(m_state->m_forwardPlan != nullptr)
                return;
            createPlans(primary, secondary);
        }

        [[nodiscard]] std::size_t workspaceBytes() const noexcept
        {
            return 0u;
        }

        void setWorkspace(void*, std::size_t)
        {
            throw std::invalid_argument("FFTW backend does not support user workspace.");
        }

        void trackCompletion(auto& queue)
        {
            auto event = queue.getDevice().makeEvent();
            queue.enqueue(event);
            std::lock_guard lock(m_pendingWaitsMutex);
            m_pendingWaits.emplace_back([event]() mutable { alpaka::onHost::wait(event); });
        }

        static void executeComplex(
            std::shared_ptr<PlanState> const& state,
            auto* inPtr,
            auto* outPtr,
            Direction direction)
        {
            auto const plan = direction == Direction::forward ? state->m_forwardPlan : state->m_inversePlan;
            validate(plan != nullptr, "FFTW plan creation failed.");
            if constexpr(std::same_as<real_type, float>)
                fftwf_execute_dft(plan, inPtr, outPtr);
            else
                fftw_execute_dft(plan, inPtr, outPtr);
        }

        static void executeR2C(
            std::shared_ptr<PlanState> const& state,
            real_type* rawInPtr,
            typename traits::complex_type* outCpx)
        {
            validate(state->m_forwardPlan != nullptr, "FFTW plan creation failed.");
            if constexpr(std::same_as<real_type, float>)
                fftwf_execute_dft_r2c(state->m_forwardPlan, rawInPtr, outCpx);
            else
                fftw_execute_dft_r2c(state->m_forwardPlan, rawInPtr, outCpx);
        }

        static void executeC2R(
            std::shared_ptr<PlanState> const& state,
            typename traits::complex_type* inCpx,
            real_type* rawOutPtr)
        {
            validate(state->m_forwardPlan != nullptr, "FFTW plan creation failed.");
            if constexpr(std::same_as<real_type, float>)
                fftwf_execute_dft_c2r(state->m_forwardPlan, inCpx, rawOutPtr);
            else
                fftw_execute_dft_c2r(state->m_forwardPlan, inCpx, rawOutPtr);
        }

        void execute(auto& queue, auto const& in, auto& out, Direction direction)
        {
            auto* rawInPtr = removeCvPtr(in.data());
            auto* rawOutPtr = removeCvPtr(out.data());
            ensurePlansCreated(rawInPtr, rawOutPtr);

            using QueueNative = decltype(queue.getNativeHandle());
            if constexpr(ComplexScalar<T_Value>)
            {
                validate(direction == Direction::forward || direction == Direction::backward, "Invalid direction.");
                auto* inPtr = reinterpret_cast<typename traits::complex_type*>(rawInPtr);
                auto* outPtr = reinterpret_cast<typename traits::complex_type*>(rawOutPtr);
                queue.enqueueNativeFn(
                    [state = m_state, inPtr, outPtr, direction]([[maybe_unused]] QueueNative nativeQueue)
                    { executeComplex(state, inPtr, outPtr, direction); });
            }
            else
            {
                using InValue = std::remove_cv_t<std::remove_pointer_t<ALPAKA_TYPEOF(rawInPtr)>>;
                if constexpr(std::same_as<InValue, real_type>)
                {
                    validate(direction == Direction::forward, "R2C only supports forward execution.");
                    auto* outCpx = reinterpret_cast<typename traits::complex_type*>(rawOutPtr);
                    queue.enqueueNativeFn(
                        [state = m_state, rawInPtr, outCpx]([[maybe_unused]] QueueNative nativeQueue)
                        {
                            // Calling a helper function instead of the fft function directly is required because the
                            // nvcc frontend is ignoring `if constexpr`.
                            executeR2C(state, rawInPtr, outCpx);
                        });
                }
                else
                {
                    validate(direction == Direction::backward, "C2R only supports backward execution.");
                    auto* inCpx = reinterpret_cast<typename traits::complex_type*>(rawInPtr);
                    queue.enqueueNativeFn(
                        [state = m_state, inCpx, rawOutPtr]([[maybe_unused]] QueueNative nativeQueue)
                        {
                            // Calling a helper function instead of the fft function directly is required because the
                            // nvcc frontend is ignoring `if constexpr`.
                            executeC2R(state, inCpx, rawOutPtr);
                        });
                }
            }
        }

    private:
        void waitForPending() noexcept
        {
            std::vector<std::function<void()>> waits;
            {
                std::lock_guard lock(m_pendingWaitsMutex);
                waits.swap(m_pendingWaits);
            }
            for(auto& waitFn : waits)
            {
                try
                {
                    waitFn();
                }
                catch(...)
                {
                }
            }
        }

        std::mutex m_pendingWaitsMutex;
        std::vector<std::function<void()>> m_pendingWaits;
    };
} // namespace alpaka::fft::internal
#endif
