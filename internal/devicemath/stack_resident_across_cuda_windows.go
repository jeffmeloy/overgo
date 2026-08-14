//go:build windows

package devicemath

import (
	"context"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// LayerTensorOffsets are the element offsets of one pre-norm layer's nine weight
// tensors within a flat device buffer (the same offsets serve the weight buffer and
// the gradient buffer, which share a layout). The caller (densecausal) computes
// these from its sorted optimizer plan; the resident stack resolves them against the
// flat weight/grad base pointers.
type LayerTensorOffsets struct {
	InLN, PostLN, Q, K, V, O, Gate, Up, Down int
}

// LayerNormPair carries a layer's two norm-vector slices (input_layernorm and
// post_attention_layernorm). Used both to feed the current host-owned norm weights
// into the resident buffer before the forward and to return the norm-vector
// gradients to the host after the backward -- the norm vectors take a sign update
// (no device kernel), so they stay host-owned while the matrix weights are resident.
type LayerNormPair struct {
	InLN, PostLN []float32
}

func ptrAt(base driver.DevicePtr, elems int) driver.DevicePtr {
	return base + driver.DevicePtr(uint64(elems)*f32Bytes)
}

// ResidentPtr returns the device pointer elems f32 elements into a flat resident
// buffer -- the caller-facing offset helper for addressing sub-tensors of dW/dG/dM.
func ResidentPtr(base driver.DevicePtr, elems int) driver.DevicePtr {
	return ptrAt(base, elems)
}

// StackForwardBackwardResidentWeights runs the whole pre-norm stack forward then
// backward against PRE-UPLOADED, persistent-across-steps device weight buffers: the
// matrix weights live in dW and their gradients are written into dG, both owned by
// the caller and never round-tripped per step. Only the current norm-vector weights
// (normWeights, host-owned because they take a sign update) are copied into dW at
// the top of the session, and only the norm-vector gradients plus the embedding-
// input gradient come back to the host. The layer math and fwd/bwd composition are
// the shared runStack/forwardDevice/backwardDevice (single owner). Attention bias is
// not supported. Same math as StackForwardBackwardResident, weights resident.
func StackForwardBackwardResidentWeights(
	worker *device.Worker,
	embeds []float32,
	dW, dG driver.DevicePtr,
	offsets []LayerTensorOffsets,
	normWeights []LayerNormPair,
	invFreq []float32,
	seq, hidden, heads, kvHeads, hd, inter int,
	rmsEps float64,
	tail func(final []float32) ([]float32, error),
) ([]float32, []LayerNormPair, error) {
	d := newLayerDims(seq, hidden, heads, kvHeads, hd, inter, rmsEps)
	nL := len(offsets)
	if nL == 0 || heads%kvHeads != 0 || len(embeds) != seq*hidden || len(invFreq) != hd/2 {
		return nil, nil, fmt.Errorf("StackForwardBackwardResidentWeights: shape mismatch (layers=%d seq=%d hidden=%d heads=%d kv=%d hd=%d)", nL, seq, hidden, heads, kvHeads, hd)
	}
	if len(normWeights) != nL {
		return nil, nil, fmt.Errorf("StackForwardBackwardResidentWeights: normWeights %d != layers %d", len(normWeights), nL)
	}
	for i, nw := range normWeights {
		if len(nw.InLN) != hidden || len(nw.PostLN) != hidden {
			return nil, nil, fmt.Errorf("StackForwardBackwardResidentWeights: layer %d norm vector length mismatch", i)
		}
	}

	dxEmbed := make([]float32, seq*hidden)
	normGrads := make([]LayerNormPair, nL)
	for i := range normGrads {
		normGrads[i] = LayerNormPair{InLN: make([]float32, hidden), PostLN: make([]float32, hidden)}
	}

	err := withCUDABLAS(worker, func(s *cudaBLAS) error {
		ops, err := newLayerOps(s, stackFnNames, d, invFreq)
		if err != nil {
			return err
		}
		wp, gp, err := residentLayerPointers(s, dW, dG, offsets, normWeights, d.hidden)
		if err != nil {
			return err
		}
		dOutP, err := ops.runStack(embeds, wp, gp, tail)
		if err != nil {
			return err
		}
		// Matrix grads stay resident in dG; only norm-vector grads + dxEmbed come back.
		downloads := make([]cudaDownload, 0, nL*2+1)
		for i := 0; i < nL; i++ {
			downloads = append(downloads,
				cudaDownload{normGrads[i].InLN, gp[i].dInLN}, cudaDownload{normGrads[i].PostLN, gp[i].dPostLN})
		}
		downloads = append(downloads, cudaDownload{dxEmbed, dOutP})
		return s.finish(downloads...)
	})
	if err != nil {
		return nil, nil, err
	}
	return dxEmbed, normGrads, nil
}

// ResidentSlice names a host buffer and the element offset of its device home
// within a flat resident buffer, for batched partial transfers.
type ResidentSlice struct {
	ElemOffset int
	Data       []float32
}

// AllocResidentF32 allocates a persistent device f32 buffer of count elements and
// either uploads init (len must equal count) or zeroes it (init nil). The pointer
// survives across worker.Do calls until FreeResident releases it -- the backbone of
// across-step weight/momentum residency.
func AllocResidentF32(worker *device.Worker, count int, init []float32) (driver.DevicePtr, error) {
	if count <= 0 {
		return 0, fmt.Errorf("AllocResidentF32: count %d must be positive", count)
	}
	if init != nil && len(init) != count {
		return 0, fmt.Errorf("AllocResidentF32: init len %d != count %d", len(init), count)
	}
	bytes, ok := checked.Bytes(uint64(count), f32Bytes)
	if !ok {
		return 0, fmt.Errorf("AllocResidentF32: size overflow for %d elements", count)
	}
	var ptr driver.DevicePtr
	err := worker.Do(context.Background(), func(state *device.State) error {
		p, err := state.Driver.MemAlloc(bytes)
		if err != nil {
			return err
		}
		if init != nil {
			if err := state.Driver.MemcpyHtoD(p, driver.Bytes(init)); err != nil {
				_ = state.Driver.MemFree(p)
				return err
			}
		} else if err := state.Driver.MemsetD32Async(p, 0, uint64(count), state.Stream); err != nil {
			_ = state.Driver.MemFree(p)
			return err
		}
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			_ = state.Driver.MemFree(p)
			return err
		}
		ptr = p
		return nil
	})
	if err != nil {
		return 0, err
	}
	return ptr, nil
}

// FreeResident releases persistent device buffers allocated by AllocResidentF32.
func FreeResident(worker *device.Worker, ptrs ...driver.DevicePtr) error {
	return worker.Do(context.Background(), func(state *device.State) error {
		var err error
		for _, p := range ptrs {
			if p != 0 {
				if e := state.Driver.MemFree(p); e != nil && err == nil {
					err = e
				}
			}
		}
		return err
	})
}

// WriteResident copies host slices into the named element ranges of a flat
// resident buffer in one session (host-to-device) -- the upload counterpart of
// ReadResident. The resident training loop uses it to refresh the resident
// gradient buffer with the host-computed per-step gradients without reallocating.
func WriteResident(worker *device.Worker, base driver.DevicePtr, slices ...ResidentSlice) error {
	return worker.Do(context.Background(), func(state *device.State) error {
		for _, sl := range slices {
			if err := state.Driver.MemcpyHtoD(ptrAt(base, sl.ElemOffset), driver.Bytes(sl.Data)); err != nil {
				return err
			}
		}
		return state.Driver.StreamSynchronize(state.Stream)
	})
}

// ReadResident copies the named element ranges of a flat resident buffer back to
// their host slices in one session (device-to-host).
func ReadResident(worker *device.Worker, base driver.DevicePtr, slices ...ResidentSlice) error {
	return worker.Do(context.Background(), func(state *device.State) error {
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		for _, sl := range slices {
			if err := state.Driver.MemcpyDtoH(driver.Bytes(sl.Data), ptrAt(base, sl.ElemOffset)); err != nil {
				return err
			}
		}
		return nil
	})
}
