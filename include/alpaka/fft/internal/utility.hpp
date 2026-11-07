/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/fft/common.hpp"

namespace alpaka::fft::internal
{
    template<alpaka::concepts::Vector T_TargetVec>
    [[nodiscard]] constexpr auto castVec(alpaka::concepts::Vector auto const& vec)
    {
        static_assert(T_TargetVec::dim() == ALPAKA_TYPEOF(vec)::dim(), "Extent dimensions must match.");
        return alpaka::lpCast<alpaka::GetValueType_t<T_TargetVec>>(vec);
    }

    template<alpaka::concepts::Vector T_TargetVec>
    [[nodiscard]] constexpr auto normalizeVectorOrScalar(alpaka::concepts::VectorOrScalar auto const& value)
    {
        if constexpr(alpaka::concepts::Vector<ALPAKA_TYPEOF(value)>)
            return castVec<T_TargetVec>(value);
        else
        {
            using source_index_type = ALPAKA_TYPEOF(value);
            using target_index_type = alpaka::GetValueType_t<T_TargetVec>;
            static_assert(
                alpaka::concepts::LosslesslyConvertible<source_index_type, target_index_type>,
                "Extent scalar type must upcast to the extent vector element type without precision loss.");
            return alpaka::Vec<target_index_type, T_TargetVec::dim()>::fill(static_cast<target_index_type>(value));
        }
    }

    [[nodiscard]] constexpr auto asExtentVec(alpaka::concepts::VectorOrScalar auto const& extents)
    {
        if constexpr(alpaka::concepts::Vector<ALPAKA_TYPEOF(extents)>)
            return ALPAKA_TYPEOF(extents){extents};
        else
            return alpaka::Vec<ALPAKA_TYPEOF(extents), 1u>::fill(extents);
    }

    [[nodiscard]] constexpr bool areZero(alpaka::concepts::Vector auto const& strides)
    {
        using Vec = ALPAKA_TYPEOF(strides);
        for(uint32_t i = 0u; i < Vec::dim(); ++i)
            if(strides[i] != static_cast<alpaka::GetValueType_t<Vec>>(0u))
                return false;
        return true;
    }

    template<alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto expectedInputExtents(
        Layout<T_Extents> const& layout,
        Transform transform,
        Placement placement)
    {
        if(placement == Placement::inPlace && transform == Transform::r2c)
            return r2cPaddedRealExtent(layout.extents);
        if(transform == Transform::c2r)
            return r2cComplexExtent(layout.extents);
        return layout.extents;
    }

    template<alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto expectedOutputExtents(
        Layout<T_Extents> const& layout,
        Transform transform,
        Placement placement)
    {
        if(placement == Placement::inPlace && transform == Transform::c2r)
            return r2cPaddedRealExtent(layout.extents);
        if(placement == Placement::inPlace && transform == Transform::r2c)
            return r2cComplexExtent(layout.extents);
        if(transform == Transform::r2c)
            return r2cComplexExtent(layout.extents);
        return layout.extents;
    }

    template<typename T_Value>
    [[nodiscard]] constexpr std::size_t inputElementBytes(Transform transform)
    {
        using real_type = Real_t<T_Value>;
        using complex_type = Complex_t<real_type>;

        if constexpr(ComplexScalar<T_Value>)
            return sizeof(T_Value);
        else
            return transform == Transform::c2r ? sizeof(complex_type) : sizeof(real_type);
    }

    template<typename T_Value>
    [[nodiscard]] constexpr std::size_t outputElementBytes(Transform transform)
    {
        using real_type = Real_t<T_Value>;
        using complex_type = Complex_t<real_type>;

        if constexpr(ComplexScalar<T_Value>)
            return sizeof(T_Value);
        else
            return transform == Transform::r2c ? sizeof(complex_type) : sizeof(real_type);
    }

    /** Return the expected input byte-distance between consecutive batches.
     *
     * When the user has not set a distance (value is 0), the default is the physical span of the declared layout:
     * `extent[0] * stride[0]` for custom strides, or the contiguous product of the input extents and the element
     * size otherwise.
     */
    template<typename T_Value, alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto expectedInDistance(
        Layout<T_Extents> const& layout,
        Transform transform,
        Placement placement)
    {
        using byte_type = typename Layout<T_Extents>::byte_type;
        if(layout.inDistance != static_cast<byte_type>(0u))
            return layout.inDistance;
        auto const ext = expectedInputExtents(layout, transform, placement);
        if(!areZero(layout.inStrides))
            return static_cast<byte_type>(ext[0u]) * static_cast<byte_type>(layout.inStrides[0u]);
        return static_cast<byte_type>(ext.product() * inputElementBytes<T_Value>(transform));
    }

    /** Return the expected output byte-distance between consecutive batches.
     *
     * When the user has not set a distance (value is 0), the default is the physical span of the declared layout:
     * `extent[0] * stride[0]` for custom strides, or the contiguous product of the output extents and the element
     * size otherwise.
     */
    template<typename T_Value, alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto expectedOutDistance(
        Layout<T_Extents> const& layout,
        Transform transform,
        Placement placement)
    {
        using byte_type = typename Layout<T_Extents>::byte_type;
        if(layout.outDistance != static_cast<byte_type>(0u))
            return layout.outDistance;
        auto const ext = expectedOutputExtents(layout, transform, placement);
        if(!areZero(layout.outStrides))
            return static_cast<byte_type>(ext[0u]) * static_cast<byte_type>(layout.outStrides[0u]);
        return static_cast<byte_type>(ext.product() * outputElementBytes<T_Value>(transform));
    }

    /** Return the expected input byte-strides per dimension.
     *
     * Each stride is the number of bytes to advance to the next element along that dimension.
     * The last dimension is the fast-moving one (stride = sizeof(T_Value)).
     */
    template<typename T_Value, alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto expectedInStrides(
        Layout<T_Extents> const& layout,
        Transform transform,
        Placement placement)
    {
        using byte_type = typename Layout<T_Extents>::byte_type;
        using byte_vec_type = alpaka::Vec<byte_type, T_Extents::dim()>;
        auto elemStrides = contiguousStrides(expectedInputExtents(layout, transform, placement));
        byte_vec_type byteStrides{};
        for(uint32_t i = 0u; i < T_Extents::dim(); ++i)
            byteStrides[i] = static_cast<byte_type>(elemStrides[i]) * inputElementBytes<T_Value>(transform);
        return byteStrides;
    }

    /** Return the expected output byte-strides per dimension.
     *
     * Each stride is the number of bytes to advance to the next element along that dimension.
     * The last dimension is the fast-moving one (stride = sizeof(T_Value)).
     */
    template<typename T_Value, alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto expectedOutStrides(
        Layout<T_Extents> const& layout,
        Transform transform,
        Placement placement)
    {
        using byte_type = typename Layout<T_Extents>::byte_type;
        using byte_vec_type = alpaka::Vec<byte_type, T_Extents::dim()>;
        auto elemStrides = contiguousStrides(expectedOutputExtents(layout, transform, placement));
        byte_vec_type byteStrides{};
        for(uint32_t i = 0u; i < T_Extents::dim(); ++i)
            byteStrides[i] = static_cast<byte_type>(elemStrides[i]) * outputElementBytes<T_Value>(transform);
        return byteStrides;
    }

    template<typename T_Value, alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto resolvedInStrides(
        Layout<T_Extents> const& layout,
        Transform transform,
        Placement placement)
    {
        if(!areZero(layout.inStrides))
            return layout.inStrides;
        return expectedInStrides<T_Value>(layout, transform, placement);
    }

    template<typename T_Value, alpaka::concepts::Vector T_Extents>
    [[nodiscard]] constexpr auto resolvedOutStrides(
        Layout<T_Extents> const& layout,
        Transform transform,
        Placement placement)
    {
        if(!areZero(layout.outStrides))
            return layout.outStrides;
        return expectedOutStrides<T_Value>(layout, transform, placement);
    }

    template<typename T_Index, uint32_t T_dim>
    [[nodiscard]] constexpr auto embedsFromStrides(
        alpaka::Vec<T_Index, T_dim> const& elemStrides,
        alpaka::concepts::Vector auto const& logicalExtents)
    {
        std::array<T_Index, T_dim> embeds{};
        embeds[0] = static_cast<T_Index>(logicalExtents[0]);
        for(uint32_t i = 1u; i < T_dim; ++i)
        {
            if(elemStrides[i] <= static_cast<T_Index>(0))
                throw std::invalid_argument("FFT stride must be non-zero.");
            if((elemStrides[i - 1u] % elemStrides[i]) != static_cast<T_Index>(0))
                throw std::invalid_argument("FFT strides must describe a row-major pitched layout.");
            embeds[i] = static_cast<T_Index>(elemStrides[i - 1u] / elemStrides[i]);
        }
        return embeds;
    }

    // ---------------------------------------------------------------------------
    // Backend conversion helpers: bytes -> elements
    // ---------------------------------------------------------------------------

    /** Convert byte-strides to element-strides by dividing each by the element size.
     *
     * FFT backends (FFTW, oneMKL, rocFFT) work with element indices, so byte-strides
     * must be converted before passing to the backend library.
     */
    template<typename T_Index, uint32_t T_dim>
    [[nodiscard]] constexpr alpaka::Vec<T_Index, T_dim> stridesToElements(
        alpaka::Vec<std::size_t, T_dim> const& byteStrides,
        std::size_t elementSize)
    {
        alpaka::Vec<T_Index, T_dim> result{};
        for(uint32_t i = 0u; i < T_dim; ++i)
            result[i] = static_cast<T_Index>(byteStrides[i] / elementSize);
        return result;
    }

    /** Convert a byte-distance to an element-distance by dividing by the element size. */
    template<typename T_Index>
    [[nodiscard]] constexpr T_Index distanceToElements(std::size_t byteDistance, std::size_t elementSize)
    {
        return static_cast<T_Index>(byteDistance / elementSize);
    }

    [[nodiscard]] constexpr auto removeCvPtr(auto* ptr)
    {
        return const_cast<std::remove_const_t<ALPAKA_TYPEOF(*ptr)>*>(ptr);
    }

    void validateViewExtents(auto const& view, alpaka::concepts::Vector auto const& expected, std::string const& what)
    {
        auto const actual = castVec<ALPAKA_TYPEOF(expected)>(view.getExtents());
        if(actual != expected)
            throw std::invalid_argument(what + " extents do not match plan.");
    }

    void validateBatchedViewExtents(
        auto const& view,
        alpaka::concepts::Vector auto const& perTransformExtents,
        std::integral auto batch,
        std::string const& what)
    {
        using Extents = ALPAKA_TYPEOF(perTransformExtents);
        auto const actual = castVec<Extents>(view.getExtents());
        if constexpr(Extents::dim() > 1u)
        {
            for(uint32_t i = 1u; i < Extents::dim(); ++i)
            {
                if(actual[i] != perTransformExtents[i])
                    throw std::invalid_argument(what + " extents do not match plan.");
            }
        }
        if(actual[0] < batch * perTransformExtents[0])
            throw std::invalid_argument(what + " first dimension too small for batched transform.");
    }

    inline void validate(bool cond, std::string const& msg)
    {
        if(!cond)
            throw std::invalid_argument(msg);
    }
} // namespace alpaka::fft::internal
