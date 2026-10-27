package executor

import (
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// Strata's native QSA gate and fused per-head norm and rotation (kernels/cuda/strata_native_qsa.cuh, ported from Strata
// at 296b59d2, MIT: src/kernels/cuda/native_qsa.cu, native_rope.cu): native_qsa_gate_apply and native_qsa_rms_norm_rope,
// their host sides translated.

// The launches' geometry, which each kernel's indexing fixes.
const (
	strataNativeQSAGateThreads   = 256   // gate's block
	strataNativeNormRopeThreads  = 256   // norm_rope_kernel's block, a row each
	strataNativeNormRopeMaxRows  = 65535 // a grid's y and the kernel's row bound
	strataNativeNormRopeRotation = 64    // the rotated values native_norm_rope_usable takes
	strataNativeNormRopeNarrow   = 128   // the head widths native_norm_rope_usable takes
	strataNativeNormRopeWide     = 256
)

// launchStrataNativeQSAGate is native_qsa_gate_apply: each of nHead heads of headDim attention values times the sigmoid
// of its gate half of qFull (a head's query then its gate, headDim each).
func launchStrataNativeQSAGate(state *device.State, functions functionSet, attn, qFull, output driver.DevicePtr, nHead, headDim int32) error {
	if nHead <= 0 || headDim <= 0 {
		return fmt.Errorf("native QSA requires positive bounded dimensions")
	}
	count := uint32(nHead) * uint32(headDim)
	return launchGridABI(state, functions[kernelStrataNativeQsaGate],
		driver.Dim3{X: (count + strataNativeQSAGateThreads - 1) / strataNativeQSAGateThreads, Y: 1, Z: 1},
		driver.Dim3{X: strataNativeQSAGateThreads, Y: 1, Z: 1}, &attn, &qFull, &output, &nHead, &headDim)
}

// launchStrataNativeNormRope is native_qsa_rms_norm_rope: rows rows of headDim values (inStride apart) normed by their
// root mean square and gamma, the first nRot rotated at each row's position (multiAxis, a multi-axis table, or 0), into
// out (headDim apart).
func launchStrataNativeNormRope(state *device.State, functions functionSet, x driver.DevicePtr, inStride int32, gamma, out driver.DevicePtr,
	rows, headDim, nRot int32, epsilon float32, scaling strataRopeScaling, positions, multiAxis driver.DevicePtr) error {
	if x == 0 || gamma == 0 || out == 0 || positions == 0 || rows < 1 || rows > strataNativeNormRopeMaxRows ||
		(headDim != strataNativeNormRopeNarrow && headDim != strataNativeNormRopeWide) || inStride < headDim ||
		nRot != strataNativeNormRopeRotation || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) || epsilon < 0 {
		return fmt.Errorf("native_qsa_rms_norm_rope: invalid arguments")
	}
	thetaScale := float32(math.Pow(scaling.freqBase, -strataPairWidth/float64(nRot)))
	args := scaling.kernelArgs(int(nRot))
	return launchGridABI(state, functions[kernelStrataNativeNormRope], driver.Dim3{X: uint32(rows), Y: 1, Z: 1},
		driver.Dim3{X: strataNativeNormRopeThreads, Y: 1, Z: 1},
		&x, &inStride, &gamma, &out, &headDim, &nRot, &epsilon, &thetaScale, &args.freqScale, &args.corrLow,
		&args.corrHigh, &args.extFactor, &args.attnFactor, &positions, &multiAxis)
}
