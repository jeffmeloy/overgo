package torchrng

import "math"

/*
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

// This altered Go transcription preserves scalar float32 rounding.
// The source is PyTorch 2.12.0+cu126 ATen/native/cpu/avx_mathfun.h.

// These constants define the pinned AVX2 logarithm and sine/cosine polynomials.
// Their scope is CPU normal generation and they change only with its reference contract.
const (
	cpuExponentBits         = 23
	cpuExponentBias         = 127
	cpuExponentMask         = 0x7f800000
	cpuHalf         float32 = 0.5
	cpuSqrtHalf     float32 = 0.707106781186547524
	cpuLogP0        float32 = 7.0376836292e-2
	cpuLogP1        float32 = -1.1514610310e-1
	cpuLogP2        float32 = 1.1676998740e-1
	cpuLogP3        float32 = -1.2420140846e-1
	cpuLogP4        float32 = 1.4249322787e-1
	cpuLogP5        float32 = -1.6668057665e-1
	cpuLogP6        float32 = 2.0000714765e-1
	cpuLogP7        float32 = -2.4999993993e-1
	cpuLogP8        float32 = 3.3333331174e-1
	cpuLogQ1        float32 = -2.12194440e-4
	cpuLogQ2        float32 = 0.693359375
	cpuQuarterTurn  float32 = 1.27323954473516
	cpuDP1          float32 = -0.78515625
	cpuDP2          float32 = -2.4187564849853515625e-4
	cpuDP3          float32 = -3.77489497744594108e-8
	cpuSinP0        float32 = -1.9515295891e-4
	cpuSinP1        float32 = 8.3321608736e-3
	cpuSinP2        float32 = -1.6666654611e-1
	cpuCosP0        float32 = 2.443315711809948e-005
	cpuCosP1        float32 = -1.388731625493765e-003
	cpuCosP2        float32 = 4.166664568298827e-002
	cpuQuadrantBit          = 2
	cpuHalfTurnBit          = 4
	cpuSignShift            = 29
)

// cpuLog evaluates the reference polynomial for positive normal-generator uniforms.
func cpuLog(x float32) float32 {
	bits := math.Float32bits(x)
	e := float32(int32(bits>>cpuExponentBits)-cpuExponentBias) + 1
	x = math.Float32frombits((bits & ^uint32(cpuExponentMask)) | math.Float32bits(cpuHalf))
	var tmp float32
	if x < cpuSqrtHalf {
		tmp = x
		e -= 1
	}
	x = float32(x-1) + tmp
	z := float32(x * x)
	y := cpuLogP0
	for _, p := range [...]float32{cpuLogP1, cpuLogP2, cpuLogP3, cpuLogP4, cpuLogP5, cpuLogP6, cpuLogP7, cpuLogP8} {
		y = float32(y*x) + p
	}
	y = float32(float32(y*x) * z)
	y += float32(e * cpuLogQ1)
	y -= float32(z * cpuHalf)
	return float32(x+y) + float32(e*cpuLogQ2)
}

// cpuSinCos evaluates the reference polynomials for normal-generator angles.
func cpuSinCos(x float32) (float32, float32) {
	signSin := math.Float32bits(x) & upperBit
	x = math.Float32frombits(math.Float32bits(x) & lowerBits)
	j := (int32(float32(x*cpuQuarterTurn)) + 1) &^ 1
	y := float32(j)
	signSin ^= uint32(j&cpuHalfTurnBit) << cpuSignShift
	signCos := uint32((^(j - cpuQuadrantBit))&cpuHalfTurnBit) << cpuSignShift
	x += float32(y * cpuDP1)
	x += float32(y * cpuDP2)
	x += float32(y * cpuDP3)
	z := float32(x * x)
	c := float32(cpuCosP0*z) + cpuCosP1
	c = float32(c*z) + cpuCosP2
	c = float32(float32(c*z) * z)
	c = float32(c-float32(z*cpuHalf)) + 1
	s := float32(cpuSinP0*z) + cpuSinP1
	s = float32(s*z) + cpuSinP2
	s = float32(float32(float32(s*z)*x) + x)
	if j&cpuQuadrantBit != 0 {
		s, c = c, s
	}
	return math.Float32frombits(math.Float32bits(s) ^ signSin), math.Float32frombits(math.Float32bits(c) ^ signCos)
}
