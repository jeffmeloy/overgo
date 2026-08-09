package diffusionimage

import "fmt"

// OTLinearFlowPathInto: linear optimal-transport flow state and velocity
// target (Lipman 2022 as executed by the artifact's train.py):
// x_t = (1-(1-sigma_min)t)*x0 + t*x1, target = x1 - (1-sigma_min)*x0.
func OTLinearFlowPathInto(xT, target, x1, x0, t []float32, batch, elemsPerSample int, sigmaMin float64) error {
	if batch <= 0 || elemsPerSample <= 0 {
		return fmt.Errorf("ot flow: shape batch=%d elems=%d", batch, elemsPerSample)
	}
	n := batch * elemsPerSample
	if len(xT) != n || len(target) != n || len(x1) != n || len(x0) != n {
		return fmt.Errorf("ot flow: lens xT=%d target=%d x1=%d x0=%d want=%d", len(xT), len(target), len(x1), len(x0), n)
	}
	if len(t) != batch {
		return fmt.Errorf("ot flow: len t=%d want=%d", len(t), batch)
	}
	oneMinusSigmaMin := 1 - sigmaMin
	for bi := 0; bi < batch; bi++ {
		tv := float64(t[bi])
		sigmaT := 1 - oneMinusSigmaMin*tv
		base := bi * elemsPerSample
		for i := 0; i < elemsPerSample; i++ {
			j := base + i
			x0v, x1v := float64(x0[j]), float64(x1[j])
			xT[j] = float32(sigmaT*x0v + tv*x1v)
			target[j] = float32(x1v - oneMinusSigmaMin*x0v)
		}
	}
	return nil
}

// ScaledMSELossGradInto: loss = scale*mean((pred-target)^2), dst = dLoss/dpred.
func ScaledMSELossGradInto(dst, pred, target []float32, scale float64) (float64, error) {
	if len(pred) == 0 {
		return 0, fmt.Errorf("scaled mse: empty prediction")
	}
	if len(dst) != len(pred) || len(target) != len(pred) {
		return 0, fmt.Errorf("scaled mse: len dst=%d pred=%d target=%d", len(dst), len(pred), len(target))
	}
	invN := 1 / float64(len(pred))
	gradScale := 2 * scale * invN
	var loss float64
	for i, p := range pred {
		diff := float64(p) - float64(target[i])
		loss += diff * diff
		dst[i] = float32(gradScale * diff)
	}
	return scale * loss * invN, nil
}

// TrainStepSGD: one OT-flow training step on a prepared (x, target) pair —
// forward trace, scaled-MSE gradient (scale 1: a global loss scale is an
// optimizer-rate alias), backward, SGD update in place on the model's bound
// tensors. SGD, not the ported Muon: Newton-Schulz over this artifact's
// ~100 matrix tensors (up to 2048x512) is host-prohibitive per step;
// descent evidence is the training-leg gate, optimizer choice recorded.
func (m *Model) TrainStepSGD(x, target []float32, b, height, width int, lr float64) (float64, error) {
	if lr <= 0 {
		return 0, fmt.Errorf("diffusionimage train: lr %g must be positive", lr)
	}
	tile := m.Cfg.PatchSize << uint(m.Cfg.NumLevels-1)
	if height%tile != 0 || width%tile != 0 {
		return 0, fmt.Errorf("diffusionimage train: size %dx%d must be multiples of tile %d", width, height, tile)
	}
	pred, trace, err := m.forward(x, b, height, width, true)
	if err != nil {
		return 0, err
	}
	dPred := make([]float32, len(pred))
	loss, err := ScaledMSELossGradInto(dPred, pred, target, 1)
	if err != nil {
		return 0, err
	}
	grads := Grads{}
	if _, err := m.backwardFromTrace(trace, dPred, grads); err != nil {
		return 0, err
	}
	for name, weight := range m.parameters() {
		g, ok := grads[name]
		if !ok {
			continue
		}
		for i := range weight {
			weight[i] -= float32(lr * float64(g[i]))
		}
	}
	m.refreshScalars()
	return loss, nil
}

// parameters: the checkpoint tensor map — the compiled bindings alias these
// slices, so in-place updates are live (scalar COPIES refresh separately).
func (m *Model) parameters() map[string][]float32 {
	return m.raw
}

// refreshScalars: re-read the compiled scalar copies (residual scales,
// xATGLU alphas) from the updated tensor map after an optimizer step.
func (m *Model) refreshScalars() {
	at := func(name string) float32 { return m.raw[name][0] }
	m.middleResidual = at("learned_middle_residual_scale")
	refreshBlock := func(blk block) {
		switch b := blk.(type) {
		case *resBlock:
			b.residualScale = at(b.name + ".learned_residual_scale")
		case *attnBlock:
			b.attention.AlphaO = float64(at(b.name + ".attn.o_proj.alpha"))
			b.mlpAlpha = at(b.name + ".mlp.0.alpha")
			b.attentionScale = at(b.name + ".learned_residual_scale_attn")
			b.mlpScale = at(b.name + ".learned_residual_scale_mlp")
		}
	}
	for _, levels := range [][]level{m.encoders, m.decoders} {
		for i := range levels {
			for _, blk := range levels[i].blocks {
				refreshBlock(blk)
			}
		}
	}
	for _, blk := range m.middle {
		refreshBlock(blk)
	}
}
