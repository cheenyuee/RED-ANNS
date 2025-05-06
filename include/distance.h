#pragma once

#include <x86intrin.h>
#include <iostream>

namespace numaann
{
    // enum Metric
    // {
    //     UNDEFINED = 0,
    //     L2 = 1,
    //     INNER_PRODUCT = 2,
    //     PQ = 3
    // };

    class Distance
    {
    public:
        virtual float compare(const float *a, const float *b, unsigned dim) const = 0;
        virtual ~Distance() {}
    };

    // class DistanceL2 : public Distance
    // {
    // public:
    //     float compare(const float *a, const float *b, unsigned dim) const
    //     {
    //         float sum = 0.0;
    //         for (size_t i = 0; i < (size_t)dim; ++i)
    //         {
    //             sum += (a[i] - b[i]) * (a[i] - b[i]);
    //         }
    //         return sum;
    //     }
    // };

    // class DistanceIP : public Distance
    // {
    // public:
    //     float compare(const float *a, const float *b, unsigned dim) const
    //     {
    //         float sum = 0.0;
    //         for (size_t i = 0; i < (size_t)dim; ++i)
    //         {
    //             sum += a[i] * b[i];
    //         }
    //         return -sum;
    //     }
    // };

    class DistanceCOS : public Distance
    {
    public:
        float compare(const float *a, const float *b, unsigned dim) const
        {
            float dot = 0.0;
            float a2 = 0.0;
            float b2 = 0.0;
            for (size_t i = 0; i < (size_t)dim; i++)
            {
                dot += (a[i] * b[i]);
                a2 += a[i] * a[i];
                b2 += b[i] * b[i];
            }
            return -dot / sqrt(a2 * b2);
        }
    };

    class DistanceL2 : public Distance
    {
        static inline __m128 masked_read(int d, const float *x)
        {
            // assert(0 <= d && d < 4);
            __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
            switch (d)
            {
            case 3:
                buf[2] = x[2];
            case 2:
                buf[1] = x[1];
            case 1:
                buf[0] = x[0];
            }
            return _mm_load_ps(buf);
            // cannot use AVX2 _mm_mask_set1_epi32
        }

    public:
        float compare(const float *x, const float *y, unsigned d) const
        {
            __m512 msum0 = _mm512_setzero_ps();

            while (d >= 16)
            {
                __m512 mx = _mm512_loadu_ps(x);
                x += 16;
                __m512 my = _mm512_loadu_ps(y);
                y += 16;
                const __m512 a_m_b1 = mx - my;
                msum0 += a_m_b1 * a_m_b1;
                d -= 16;
            }

            __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
            msum1 += _mm512_extractf32x8_ps(msum0, 0);

            if (d >= 8)
            {
                __m256 mx = _mm256_loadu_ps(x);
                x += 8;
                __m256 my = _mm256_loadu_ps(y);
                y += 8;
                const __m256 a_m_b1 = mx - my;
                msum1 += a_m_b1 * a_m_b1;
                d -= 8;
            }

            __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
            msum2 += _mm256_extractf128_ps(msum1, 0);

            if (d >= 4)
            {
                __m128 mx = _mm_loadu_ps(x);
                x += 4;
                __m128 my = _mm_loadu_ps(y);
                y += 4;
                const __m128 a_m_b1 = mx - my;
                msum2 += a_m_b1 * a_m_b1;
                d -= 4;
            }

            if (d > 0)
            {
                __m128 mx = masked_read(d, x);
                __m128 my = masked_read(d, y);
                __m128 a_m_b1 = mx - my;
                msum2 += a_m_b1 * a_m_b1;
            }

            msum2 = _mm_hadd_ps(msum2, msum2);
            msum2 = _mm_hadd_ps(msum2, msum2);
            return _mm_cvtss_f32(msum2);
            // return result;
        }
    };

    class DistanceIP : public Distance
    {
        static inline __m128 masked_read(int d, const float *x)
        {
            // assert(0 <= d && d < 4);
            __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
            switch (d)
            {
            case 3:
                buf[2] = x[2];
            case 2:
                buf[1] = x[1];
            case 1:
                buf[0] = x[0];
            }
            return _mm_load_ps(buf);
            // cannot use AVX2 _mm_mask_set1_epi32
        }

    public:
        float compare(const float *a, const float *b, unsigned size) const
        {
            // using avx-512
            __m512 msum0 = _mm512_setzero_ps();

            while (size >= 16)
            {
                __m512 mx = _mm512_loadu_ps(a);
                a += 16;
                __m512 my = _mm512_loadu_ps(b);
                b += 16;
                msum0 = _mm512_add_ps(msum0, _mm512_mul_ps(mx, my));
                size -= 16;
            }

            __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
            msum1 += _mm512_extractf32x8_ps(msum0, 0);

            if (size >= 8)
            {
                __m256 mx = _mm256_loadu_ps(a);
                a += 8;
                __m256 my = _mm256_loadu_ps(b);
                b += 8;
                msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(mx, my));
                size -= 8;
            }

            __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
            msum2 += _mm256_extractf128_ps(msum1, 0);

            if (size >= 4)
            {
                __m128 mx = _mm_loadu_ps(a);
                a += 4;
                __m128 my = _mm_loadu_ps(b);
                b += 4;
                msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
                size -= 4;
            }

            if (size > 0)
            {
                __m128 mx = masked_read(size, a);
                __m128 my = masked_read(size, b);
                msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
            }

            msum2 = _mm_hadd_ps(msum2, msum2);
            msum2 = _mm_hadd_ps(msum2, msum2);
            return -1.0 * _mm_cvtss_f32(msum2);
            // return result;
        }
    };

}
