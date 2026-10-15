//go:build windows

package optimizer

import (
	"math"
	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/driver"
	"overgo/internal/hostoptimizer"
)

// referenceNewtonSchulz preserves the GPU reduction and host scalar arithmetic.
// Both device schedules share this test-only reference.
func (o *deviceOps) referenceNewtonSchulz(
	b nsBuffers,
	rows, cols int,
	handle cublas.Handle,
	stage1Iterations int,
) (driver.DevicePtr, error) {
	n := rows * cols
	dim := gramDim(rows, cols)
	gramN := dim * dim
	tall := rows >= cols
	dX, dXNew := b.dX, b.dXNew

	// Frobenius normalize: ||X||^2 = X_flat[1,n] . X_flat[n,1] via one GEMM.
	if err := o.blas.RowMajorGEMMF32(o.handle, 1, int32(n), 1, dX, dX, b.dNorm); err != nil {
		return 0, err
	}
	if err := o.lib.StreamSynchronize(o.stream); err != nil {
		return 0, err
	}
	normSquared := make([]float32, 1)
	if err := o.lib.MemcpyDtoH(driver.Bytes(normSquared), b.dNorm); err != nil {
		return 0, err
	}
	norm := math.Sqrt(float64(normSquared[0]))
	if norm < hostoptimizer.NewtonSchulzFrobeniusGuard {
		return dX, nil
	}
	if err := o.scale(dX, dX, float32(1/norm), n); err != nil {
		return 0, err
	}

	iterations := stage1Iterations + hostoptimizer.NewtonSchulzStage2Iterations
	for iteration := range iterations {
		coefficients := hostoptimizer.NewtonSchulzStage1
		if iteration >= stage1Iterations {
			coefficients = hostoptimizer.NewtonSchulzStage2
		}
		c0 := float32(coefficients[0])
		c1 := float32(coefficients[1])
		c2 := float32(coefficients[2])

		if tall {
			// gram = XᵀX  [cols x cols]
			if err := o.blas.RowMajorGEMMExF32(handle, true, false, int32(cols), int32(rows), int32(cols), dX, dX, b.dGram); err != nil {
				return 0, err
			}
		} else {
			// gram = XXᵀ  [rows x rows]
			if err := o.blas.RowMajorGEMMExF32(handle, false, true, int32(rows), int32(cols), int32(rows), dX, dX, b.dGram); err != nil {
				return 0, err
			}
		}
		// sq = gram·gram  [dim x dim]
		if err := o.blas.RowMajorGEMMF32(handle, int32(dim), int32(dim), int32(dim), b.dGram, b.dGram, b.dSq); err != nil {
			return 0, err
		}
		// rest = c1*gram + c2*sq
		if err := o.scale(b.dGram, b.dRest, c1, gramN); err != nil {
			return 0, err
		}
		if err := o.scale(b.dSq, b.dSq, c2, gramN); err != nil {
			return 0, err
		}
		if err := o.add(b.dRest, b.dSq, b.dRest, gramN); err != nil {
			return 0, err
		}
		// Xnew = op·rest-product + c0*X  (c0*I folded into the matmul).
		if tall {
			// Xnew = X·rest  [rows x cols]
			if err := o.blas.RowMajorGEMMF32(handle, int32(rows), int32(cols), int32(cols), dX, b.dRest, dXNew); err != nil {
				return 0, err
			}
		} else {
			// Xnew = rest·X  [rows x cols]
			if err := o.blas.RowMajorGEMMF32(handle, int32(rows), int32(rows), int32(cols), b.dRest, dX, dXNew); err != nil {
				return 0, err
			}
		}
		if err := o.scale(dX, b.dCX, c0, n); err != nil {
			return 0, err
		}
		if err := o.add(dXNew, b.dCX, dXNew, n); err != nil {
			return 0, err
		}
		dX, dXNew = dXNew, dX
	}
	return dX, nil
}
