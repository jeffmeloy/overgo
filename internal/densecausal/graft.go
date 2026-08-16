package densecausal

import (
	"fmt"
	"math"
	"math/rand"

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
)

// Graft composes a frozen donor MLP into a frozen target model behind a
// trainable bridge pair. The branch reads the residual stream at the output of
// target layer Layer and adds
//
//	Up · donorMLP(Down · x)
//
// where Down maps target hidden width to donor hidden width and Up maps back.
// Up is zero-initialized, so an untrained graft is exactly the unmodified
// target: the composition-viability baseline is the graft at step zero, and
// any held-out delta is attributable to bridge training alone. Only Down and
// Up receive gradients; donor and target weights stay frozen by construction.
type Graft struct {
	Layer             int
	DonorHidden       int
	DonorIntermediate int
	Down              []float32 // [DonorHidden x Hidden]
	Up                []float32 // [Hidden x DonorHidden]
	DonorGate         []float32 // [DonorIntermediate x DonorHidden]
	DonorUp           []float32 // [DonorIntermediate x DonorHidden]
	DonorDown         []float32 // [DonorHidden x DonorIntermediate]
}

// BridgeGradNames key the two trainable bridge matrices in Grads.
const (
	BridgeDownName = "bridge.down"
	BridgeUpName   = "bridge.up"
)

// NewGraft validates donor component geometry against the target and
// initializes the bridge: Down with a small seeded uniform, Up with zeros.
func NewGraft(target *Model, layer int, donorGate, donorUp, donorDown []float32, donorHidden, donorIntermediate int, seed int64) (*Graft, error) {
	hidden := target.Dims.Hidden
	if layer < 0 || layer >= target.Dims.Layers {
		return nil, fmt.Errorf("densecausal: graft layer %d outside target depth %d", layer, target.Dims.Layers)
	}
	if donorHidden <= 0 || donorIntermediate <= 0 {
		return nil, fmt.Errorf("densecausal: invalid donor geometry %dx%d", donorHidden, donorIntermediate)
	}
	if len(donorGate) != donorIntermediate*donorHidden || len(donorUp) != donorIntermediate*donorHidden ||
		len(donorDown) != donorHidden*donorIntermediate {
		return nil, fmt.Errorf("densecausal: donor MLP tensors do not match geometry %dx%d", donorHidden, donorIntermediate)
	}
	graft := &Graft{
		Layer: layer, DonorHidden: donorHidden, DonorIntermediate: donorIntermediate,
		Down:      make([]float32, donorHidden*hidden),
		Up:        make([]float32, hidden*donorHidden),
		DonorGate: donorGate, DonorUp: donorUp, DonorDown: donorDown,
	}
	random := rand.New(rand.NewSource(seed))
	scale := float32(1 / math.Sqrt(float64(hidden)))
	for i := range graft.Down {
		graft.Down[i] = (random.Float32()*2 - 1) * scale
	}
	return graft, nil
}

// branchForward computes the graft branch output and retains the trace needed
// for its VJP. x is the post-layer residual stream [seq*hidden].
func (gr *Graft) branchForward(m *Model, x []float32, seq int) (out, z, gate, up, silu, wOut []float32) {
	hidden := m.Dims.Hidden
	z = make([]float32, seq*gr.DonorHidden)
	hostmath.Linear(z, x, gr.Down, seq, hidden, gr.DonorHidden)
	gate = make([]float32, seq*gr.DonorIntermediate)
	up = make([]float32, seq*gr.DonorIntermediate)
	hostmath.Linear(gate, z, gr.DonorGate, seq, gr.DonorHidden, gr.DonorIntermediate)
	hostmath.Linear(up, z, gr.DonorUp, seq, gr.DonorHidden, gr.DonorIntermediate)
	silu = make([]float32, seq*gr.DonorIntermediate)
	hostmath.SiLUGate(silu, gate, up)
	wOut = make([]float32, seq*gr.DonorHidden)
	hostmath.Linear(wOut, silu, gr.DonorDown, seq, gr.DonorIntermediate, gr.DonorHidden)
	out = make([]float32, seq*hidden)
	hostmath.Linear(out, wOut, gr.Up, seq, gr.DonorHidden, hidden)
	return out, z, gate, up, silu, wOut
}

// GraftLoss runs the grafted forward and returns mean causal CE.
func (m *Model) GraftLoss(gr *Graft, tokens []int) (float64, error) {
	states, _, err := m.graftForwardStates(gr, tokens)
	if err != nil {
		return 0, err
	}
	seq := len(tokens)
	d := m.Dims
	final := states[d.Layers]
	normed := make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(normed, final, m.Weights["model.norm.weight"], seq, d.Hidden, d.RMSEps)
	logits := make([]float32, seq*d.Vocab)
	hostmath.Linear(logits, normed, m.head(), seq, d.Hidden, d.Vocab)
	dLogits := make([]float32, (seq-1)*d.Vocab)
	loss := hostmath.SoftmaxCrossEntropy(dLogits, logits[:(seq-1)*d.Vocab], tokens[1:], seq-1, d.Vocab)
	return loss, nil
}

// graftForwardStates mirrors forwardStates with the branch applied at the
// graft layer's output. The returned preBranch buffer retains the pre-add
// residual stream (checkpoint posture: the branch trace is recomputed from it
// in backward).
func (m *Model) graftForwardStates(gr *Graft, tokens []int) ([][]float32, []float32, error) {
	invFreq := hostmath.RopeInvFreq(m.Dims.RopeTheta, m.Dims.HeadDim)
	var preBranch []float32
	states, err := m.retainedForwardStates(tokens, func(x []float32, index, seq int) error {
		l, err := m.layerWeights(index)
		if err != nil {
			return err
		}
		m.layerForward(x, l, invFreq, seq)
		if index == gr.Layer {
			preBranch = append([]float32(nil), x...)
			out, _, _, _, _, _ := gr.branchForward(m, preBranch, seq)
			addInPlace(x, out)
		}
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	return states, preBranch, nil
}

// GraftLossAndBridgeGrads runs grafted forward + full backward and returns the
// loss with gradients for ONLY the two bridge matrices. Donor and target
// gradients are computed into scratch (the chain needs their VJPs) and
// discarded: freezing is structural, not a flag the optimizer must honor.
func (m *Model) GraftLossAndBridgeGrads(gr *Graft, tokens []int) (float64, Grads, error) {
	if len(tokens) < 2 {
		return 0, nil, fmt.Errorf("densecausal: need at least 2 tokens, got %d", len(tokens))
	}
	states, preBranch, err := m.graftForwardStates(gr, tokens)
	if err != nil {
		return 0, nil, err
	}
	invFreq := hostmath.RopeInvFreq(m.Dims.RopeTheta, m.Dims.HeadDim)
	hidden := m.Dims.Hidden
	loss, _, grads, err := m.lossAndGradsFromStates(tokens, states, func(
		index int, input, outputGradient []float32, seq int, g Grads,
	) ([]float32, error) {
		if index == gr.Layer {
			// outputGradient is dL/dy for y = xOut + branch(xOut); fold the
			// branch VJP in before the layer's own backward.
			dxBranch := gr.branchBackward(m, preBranch, outputGradient, seq, g)
			dxOut := make([]float32, len(outputGradient))
			for i := range dxOut {
				dxOut[i] = outputGradient[i] + dxBranch[i]
			}
			outputGradient = dxOut
		}
		return m.layerBackward(index, input, outputGradient, invFreq, seq, g)
	})
	if err != nil {
		return 0, nil, err
	}
	bridge := Grads{
		BridgeDownName: grads[BridgeDownName],
		BridgeUpName:   grads[BridgeUpName],
	}
	if len(bridge[BridgeDownName]) != gr.DonorHidden*hidden || len(bridge[BridgeUpName]) != hidden*gr.DonorHidden {
		return 0, nil, fmt.Errorf("densecausal: bridge gradients missing from backward")
	}
	return loss, bridge, nil
}

// branchBackward recomputes the branch trace from the retained pre-add
// residual and accumulates bridge gradients; returns dL/d(preBranch) through
// the branch path only.
func (gr *Graft) branchBackward(m *Model, preBranch, dy []float32, seq int, g Grads) []float32 {
	hidden := m.Dims.Hidden
	_, z, gate, up, silu, wOut := gr.branchForward(m, preBranch, seq)

	dW := make([]float32, seq*gr.DonorHidden)
	hostmath.LinearBackward(dW, hostmath.GradientSlot(g, BridgeUpName, hidden*gr.DonorHidden), nil,
		wOut, gr.Up, dy, seq, gr.DonorHidden, hidden, false)
	dSilu := make([]float32, seq*gr.DonorIntermediate)
	donorDownScratch := make([]float32, len(gr.DonorDown))
	hostmath.LinearBackward(dSilu, donorDownScratch, nil,
		silu, gr.DonorDown, dW, seq, gr.DonorIntermediate, gr.DonorHidden, false)
	dGate := make([]float32, seq*gr.DonorIntermediate)
	dUp := make([]float32, seq*gr.DonorIntermediate)
	hostmath.SiLUGateBackward(dGate, dUp, gate, up, dSilu)
	dz := make([]float32, seq*gr.DonorHidden)
	donorGateScratch := make([]float32, len(gr.DonorGate))
	donorUpScratch := make([]float32, len(gr.DonorUp))
	hostmath.LinearBackward(dz, donorGateScratch, nil,
		z, gr.DonorGate, dGate, seq, gr.DonorHidden, gr.DonorIntermediate, false)
	hostmath.LinearBackward(dz, donorUpScratch, nil,
		z, gr.DonorUp, dUp, seq, gr.DonorHidden, gr.DonorIntermediate, true)
	dxBranch := make([]float32, seq*hidden)
	hostmath.LinearBackward(dxBranch, hostmath.GradientSlot(g, BridgeDownName, gr.DonorHidden*hidden), nil,
		preBranch, gr.Down, dz, seq, hidden, gr.DonorHidden, false)
	return dxBranch
}

// TrainBridge runs bridge-only Muon: donor and target stay frozen, the two
// bridge matrices advance through the shared optimizer with the derived or
// supplied learning rate. Returns per-step training losses.
func (m *Model) TrainBridge(gr *Graft, batches [][]int, baseLR, mu float64, steps int) ([]float64, error) {
	if steps <= 0 || len(batches) == 0 {
		return nil, fmt.Errorf("densecausal: bridge training needs steps and batches")
	}
	hidden := m.Dims.Hidden
	downSize := gr.DonorHidden * hidden
	upSize := hidden * gr.DonorHidden
	weights := make([]float32, downSize+upSize)
	copy(weights[:downSize], gr.Down)
	copy(weights[downSize:], gr.Up)
	gradients := make([]float32, len(weights))
	plan, err := optimizer.CompilePlan(len(weights), []optimizer.GroupSpec{
		{Name: BridgeDownName, Start: 0, End: downSize, Rows: gr.DonorHidden, Cols: hidden},
		{Name: BridgeUpName, Start: downSize, End: downSize + upSize, Rows: hidden, Cols: gr.DonorHidden},
	})
	if err != nil {
		return nil, err
	}
	if baseLR <= 0 {
		baseLR = optimizer.DeriveBaseLR(len(weights))
	}
	opt, err := optimizer.New(weights, gradients, plan, optimizer.Config{
		BaseLearningRate: baseLR, Momentum: mu, Schedule: optimizer.ScheduleConstant,
	})
	if err != nil {
		return nil, err
	}
	losses := make([]float64, 0, steps)
	for step := 0; step < steps; step++ {
		batch := batches[step%len(batches)]
		copy(gr.Down, weights[:downSize])
		copy(gr.Up, weights[downSize:])
		loss, bridge, err := m.GraftLossAndBridgeGrads(gr, batch)
		if err != nil {
			return nil, err
		}
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			return nil, fmt.Errorf("densecausal: bridge training diverged at step %d", step)
		}
		copy(gradients[:downSize], bridge[BridgeDownName])
		copy(gradients[downSize:], bridge[BridgeUpName])
		opt.Step()
		losses = append(losses, loss)
	}
	copy(gr.Down, weights[:downSize])
	copy(gr.Up, weights[downSize:])
	return losses, nil
}
