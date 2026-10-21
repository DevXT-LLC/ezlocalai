// Adapted from PyTorch 2.9.1 aten/src/ATen/native/cuda/Reduce.cuh
// and aten/src/ATen/native/SharedReduceOps.h. No PyTorch/CUDA dependency.
/*
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2024 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
namespace ezlocalai {
// Host emulation of an aligned, contiguous float32 scalar mean in PyTorch 2.9.1.
// Device geometry controls the CUDA grid and therefore the addition order.
inline float pytorch_cuda_mean(const std::vector<float>& data, int multiprocessors, int threads_per_mp) {
    const size_t n = data.size();
    if (!n || n > 2147483647 || multiprocessors <= 0 || threads_per_mp < 512)
        throw std::invalid_argument("Unsupported CUDA mean geometry");
    const bool vectorized = n > 128;
    const size_t dim = vectorized ? n / 4 : n;
    size_t width = 1;
    while (width * 2 <= dim && width < 512) width *= 2;
    auto ceildiv = [](size_t a, size_t b) { return (a + b - 1) / b; };
    const size_t vpt = ceildiv(n, width);
    size_t ctas = 1;
    if (vpt >= 256)
        ctas = std::max(std::min(size_t(multiprocessors) * (threads_per_mp / width), ceildiv(vpt, 16)),
                        ceildiv(vpt, 256));

    auto block_sum = [&](std::vector<float> values) {
        // Shared-memory halves precede the ascending warp shuffle offsets.
        for (size_t offset = width / 2; offset >= 32; offset /= 2)
            for (size_t t = 0; t < offset; ++t) values[t] += values[t + offset];
        const size_t warp = std::min(width, size_t(32));
        for (size_t offset = 1; offset < warp; offset *= 2)
            for (size_t t = 0; t + offset < warp; ++t) values[t] += values[t + offset];
        return values[0];
    };
    std::vector<float> partial(ctas);
    for (size_t b = 0; b < ctas; ++b) {
        std::vector<float> threads(width);
        for (size_t t = 0; t < width; ++t) {
            std::array<float, 4> accum{};
            size_t idx = t + b * width;
            const size_t stride = width * ctas;
            if (vectorized) {
                for (; idx * 4 + 3 < n; idx += stride)
                    for (size_t i = 0; i < 4; ++i) accum[i] += data[idx * 4 + i];
                if (b == 0 && n - n % 4 + t < n) accum[0] += data[n - n % 4 + t];
            } else {
                for (; idx + 3 * stride < n; idx += 4 * stride)
                    for (size_t i = 0; i < 4; ++i) accum[i] += data[idx + i * stride];
                for (size_t i = 0; i < 4 && idx + i * stride < n; ++i) accum[i] += data[idx + i * stride];
            }
            threads[t] = ((accum[0] + accum[1]) + accum[2]) + accum[3];
        }
        partial[b] = block_sum(threads);
    }
    float sum = partial[0];
    if (ctas > 1) {
        std::vector<float> threads(width);
        for (size_t t = 0; t < width; ++t)
            for (size_t i = t; i < ctas; i += width) threads[t] += partial[i];
        sum = block_sum(threads);
    }
    return sum * (1.f / n);
}
} // namespace ezlocalai
