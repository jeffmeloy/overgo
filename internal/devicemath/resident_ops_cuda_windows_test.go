//go:build windows

package devicemath

import (
	"errors"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// WithResidentOps runs one composed resident program and synchronizes once.
func WithResidentOps(worker *device.Worker, run func(*ResidentOps) error) error {
	if worker == nil || run == nil {
		return errors.New("resident ops: worker or program absent")
	}
	return withCUDABLAS(worker, func(session *cudaBLAS) error {
		ops := &ResidentOps{session: session, functions: map[string]driver.Function{}}
		if err := run(ops); err != nil {
			return err
		}
		return session.finish()
	})
}

func (o *ResidentOps) AttentionCoreBackward(
	q, k, v, probability, dOut, dQ, dK, dV, dScores driver.DevicePtr,
	sequence, headDim int,
	scale float64,
) error {
	if q == 0 || k == 0 || v == 0 || probability == 0 || dOut == 0 || dQ == 0 || dK == 0 || dV == 0 || dScores == 0 || sequence <= 0 || headDim <= 0 || math.IsNaN(scale) || math.IsInf(scale, 0) {
		return errors.New("resident ops: invalid attention VJP")
	}
	dProbability, err := o.AllocF32(sequence * sequence)
	if err != nil {
		return err
	}
	return o.AttentionCoreBackwardWithScratch(q, k, v, probability, dOut, dQ, dK, dV, dScores, dProbability, sequence, headDim, scale)
}
