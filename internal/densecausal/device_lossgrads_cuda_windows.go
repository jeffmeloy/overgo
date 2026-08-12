//go:build windows

package densecausal

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/hostmath"
)

// deviceLossAndGrads is the device counterpart to LossAndGrads: it runs both the
// per-layer forward (deviceForwardStatesCached) and the per-layer backward
// (deviceLayerBackward) on the GPU, while the head projection, final RMSNorm,
// softmax-CE and embedding scatter stay on host. Loss, logits and every parameter
// gradient match LossAndGrads within fp32 tolerance. Attention bias is not yet
// supported (AttnBias must be false).
func (m *Model) deviceLossAndGrads(worker *device.Worker, tokens []int) (float64, []float32, Grads, error) {
	if len(tokens) < 2 {
		return 0, nil, nil, fmt.Errorf("densecausal: need at least 2 tokens, got %d", len(tokens))
	}
	states, caches, err := m.deviceForwardStatesCached(worker, tokens)
	if err != nil {
		return 0, nil, nil, err
	}
	invFreq := hostmath.RopeInvFreq(m.Dims.RopeTheta, m.Dims.HeadDim)
	return m.lossAndGradsFromStates(tokens, states, func(
		index int, input, outputGradient []float32, sequence int, gradients Grads,
	) ([]float32, error) {
		return m.deviceLayerBackward(
			worker, index, input, outputGradient, caches[index], invFreq, sequence, gradients,
		)
	})
}
