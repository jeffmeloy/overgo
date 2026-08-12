//go:build windows

package devicemath

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// stackFnNames is the deduped union of the forward and backward kernel sets, so
// one session loads every function the whole-stack driver launches.
var stackFnNames = func() []string {
	seen := map[string]bool{}
	var out []string
	for _, name := range append(append([]string{}, layerForwardFnNames...), layerBackwardFnNames...) {
		if !seen[name] {
			seen[name] = true
			out = append(out, name)
		}
	}
	return out
}()

func (w LayerForwardWeights) shapeOK(d layerDims) bool {
	return len(w.InLN) == d.hidden && len(w.PostLN) == d.hidden &&
		len(w.Q) == d.width*d.hidden && len(w.K) == d.kvWidth*d.hidden && len(w.V) == d.kvWidth*d.hidden &&
		len(w.O) == d.hidden*d.width && len(w.Gate) == d.inter*d.hidden && len(w.Up) == d.inter*d.hidden &&
		len(w.Down) == d.hidden*d.inter
}

// StackForwardBackwardResident runs the WHOLE pre-norm layer stack forward then
// backward inside ONE cudaBLAS session. Every layer's weights upload once and
// serve both its forward and its backward; every layer's forward activation cache
// stays device-resident and is consumed by that layer's backward with NO host
// download/reupload in between. The only host round-trips are: the final pre-norm
// stream out (for the host head/norm/softmax-CE tail) and the tail's returned
// output-gradient back in. `tail(final)` computes loss/logits/head+norm backward
// on host and returns dOut for the last layer. Returns the embedding-input
// gradient (dX after layer 0) and each layer's weight grads. Attention bias is
// not supported (reject upstream). Same math as the per-layer resident path.
func StackForwardBackwardResident(
	worker *device.Worker,
	embeds []float32,
	layers []LayerForwardWeights,
	invFreq []float32,
	seq, hidden, heads, kvHeads, hd, inter int,
	rmsEps float64,
	tail func(final []float32) ([]float32, error),
) ([]float32, []LayerBackwardResult, error) {
	d := newLayerDims(seq, hidden, heads, kvHeads, hd, inter, rmsEps)
	nL := len(layers)
	if nL == 0 || heads%kvHeads != 0 || len(embeds) != seq*hidden || len(invFreq) != hd/2 {
		return nil, nil, fmt.Errorf("StackForwardBackwardResident: shape mismatch (layers=%d seq=%d hidden=%d heads=%d kv=%d hd=%d)", nL, seq, hidden, heads, kvHeads, hd)
	}
	for i, w := range layers {
		if !w.shapeOK(d) {
			return nil, nil, fmt.Errorf("StackForwardBackwardResident: layer %d weight shape mismatch", i)
		}
	}

	grads := make([]LayerBackwardResult, nL)
	for i := range grads {
		grads[i].DWInLN = make([]float32, hidden)
		grads[i].DWPostLN = make([]float32, hidden)
		grads[i].DWQ = make([]float32, d.width*hidden)
		grads[i].DWK = make([]float32, d.kvWidth*hidden)
		grads[i].DWV = make([]float32, d.kvWidth*hidden)
		grads[i].DWO = make([]float32, hidden*d.width)
		grads[i].DWGate = make([]float32, inter*hidden)
		grads[i].DWUp = make([]float32, inter*hidden)
		grads[i].DWDown = make([]float32, hidden*inter)
	}
	dxEmbed := make([]float32, seq*hidden)
	final := make([]float32, seq*hidden)

	err := withCUDABLAS(worker, func(s *cudaBLAS) error {
		ops, err := newLayerOps(s, stackFnNames, d, invFreq)
		if err != nil {
			return err
		}

		// --- Forward: all layers, caches + weights + input residuals resident. ---
		xIn := make([]driver.DevicePtr, nL+1)
		xIn[0], err = s.upload(embeds)
		if err != nil {
			return err
		}
		wp := make([]layerWeightPtrs, nL)
		cp := make([]layerCachePtrs, nL)
		for i := 0; i < nL; i++ {
			if wp[i], err = uploadLayerWeights(s.cudaScope, layers[i]); err != nil {
				return err
			}
			if cp[i], err = allocLayerCache(s.cudaScope, d); err != nil {
				return err
			}
			if xIn[i+1], err = s.alloc(seq * hidden); err != nil {
				return err
			}
			if err := ops.forwardDevice(xIn[i], wp[i], cp[i], xIn[i+1]); err != nil {
				return err
			}
		}

		// Only host round-trip out: the final pre-norm stream for the head/norm/CE tail.
		if err := s.finish(cudaDownload{final, xIn[nL]}); err != nil {
			return err
		}
		dOutHost, err := tail(final)
		if err != nil {
			return err
		}
		if len(dOutHost) != seq*hidden {
			return fmt.Errorf("StackForwardBackwardResident: tail dOut len %d != %d", len(dOutHost), seq*hidden)
		}

		// --- Backward: caches consumed resident; dOut flows device-to-device. ---
		dOutP, err := s.upload(dOutHost)
		if err != nil {
			return err
		}
		gp := make([]layerGradPtrs, nL)
		for i := range gp {
			if gp[i], err = allocLayerGrads(s.cudaScope, d); err != nil {
				return err
			}
		}
		for i := nL - 1; i >= 0; i-- {
			dXP, err := s.alloc(seq * hidden)
			if err != nil {
				return err
			}
			if err := ops.backwardDevice(xIn[i], dOutP, cp[i], wp[i], gp[i], dXP); err != nil {
				return err
			}
			dOutP = dXP // becomes the previous layer's output gradient
		}

		downloads := make([]cudaDownload, 0, nL*9+1)
		for i := 0; i < nL; i++ {
			downloads = append(downloads,
				cudaDownload{grads[i].DWInLN, gp[i].dInLN}, cudaDownload{grads[i].DWPostLN, gp[i].dPostLN},
				cudaDownload{grads[i].DWQ, gp[i].dQ}, cudaDownload{grads[i].DWK, gp[i].dK}, cudaDownload{grads[i].DWV, gp[i].dV},
				cudaDownload{grads[i].DWO, gp[i].dO}, cudaDownload{grads[i].DWGate, gp[i].dGate},
				cudaDownload{grads[i].DWUp, gp[i].dUp}, cudaDownload{grads[i].DWDown, gp[i].dDown},
			)
		}
		downloads = append(downloads, cudaDownload{dxEmbed, dOutP})
		return s.finish(downloads...)
	})
	if err != nil {
		return nil, nil, err
	}
	return dxEmbed, grads, nil
}
