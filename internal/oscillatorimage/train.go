package oscillatorimage

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"overgo/internal/optimizer"
)

// Grads: parameter gradients keyed by artifact tensor name.
type Grads map[string][]float32

// trainingTrace: Euler state history plus the decoder tape.
type trainingTrace struct {
	states  []float32
	b       int
	decoder decoderTape
}

// trainingForwardTrace: forward keeping every integration state for BPTT.
func (m *Model) trainingForwardTrace(init, drive []float32, b int) ([]float32, trainingTrace) {
	cfg := m.Cfg
	tot := cfg.N + cfg.NCond
	stateSize := b * tot
	states := make([]float32, (cfg.NumSteps+1)*stateSize)
	copy(states[:stateSize], init)
	vel := make([]float32, stateSize)
	trig := make([]float64, 2*tot)
	for step := 0; step < cfg.NumSteps; step++ {
		state := states[step*stateSize : (step+1)*stateSize]
		next := states[(step+1)*stateSize : (step+2)*stateSize]
		conditionalKuramotoForwardInto(vel, state, m.Omega, m.OmegaCond, m.K, m.KCond, drive, b, cfg.N, cfg.NCond, cfg.KScale, cfg.KCondScale, cfg.KDriveScale, trig[:tot], trig[tot:])
		for i := range next {
			next[i] = float32(float64(state[i]) + cfg.Dt*float64(vel[i]))
		}
	}
	finalState := states[cfg.NumSteps*stateSize : (cfg.NumSteps+1)*stateSize]
	features := readoutTransform(finalState, b, cfg.N, tot, 0, cfg.Relativization, cfg.Encoding)
	output, tape := decoderForwardTrace(features, m.Blocks, m.ToOutW, m.ToOutB, b, cfg.InChannels, cfg.InH, cfg.InW, cfg.OutChannels, m.Slope, cfg.TanhOut)
	return output, trainingTrace{states: states, b: b, decoder: tape}
}

func (g Grads) add(name string, values []float32) {
	dst, ok := g[name]
	if !ok {
		dst = make([]float32, len(values))
		g[name] = dst
	}
	for i, v := range values {
		dst[i] += v
	}
}

func (g Grads) ensure(name string, n int) []float32 {
	dst, ok := g[name]
	if !ok || len(dst) != n {
		dst = make([]float32, n)
		g[name] = dst
	}
	return dst
}

// backwardInto: full generator VJP — tanh/decoder, readout, then Euler BPTT.
func (m *Model) backwardInto(trace trainingTrace, drive, dImage []float32, g Grads) {
	cfg := m.Cfg
	b := trace.b
	tot := cfg.N + cfg.NCond
	stateSize := b * tot
	numSteps := len(trace.states)/stateSize - 1

	// Decoder backward over the tape.
	dPre := dImage
	if cfg.TanhOut {
		dPre = make([]float32, len(dImage))
		for i := range dImage {
			value := float64(trace.decoder.output[i])
			dPre[i] = dImage[i] * float32(1-value*value)
		}
	}
	cin, h, w := cfg.InChannels, cfg.InH, cfg.InW
	if len(m.Blocks) > 0 {
		last := trace.decoder.dims[len(trace.decoder.dims)-1]
		cin, h, w = m.Blocks[len(m.Blocks)-1].Cout, upsample*last[1], upsample*last[2]
	}
	dx, dW, dB := conv2dSame3x3Backward(trace.decoder.rows[len(m.Blocks)], m.ToOutW, dPre, b, cin, cfg.OutChannels, h, w)
	g.add(m.tensorName("to_out.weight"), dW)
	g.add(m.tensorName("to_out.bias"), dB)
	for i := len(m.Blocks) - 1; i >= 0; i-- {
		bc, bh, bw := trace.decoder.dims[i][0], trace.decoder.dims[i][1], trace.decoder.dims[i][2]
		var dW1, dB1, dW2, dB2 []float32
		dx, dW1, dB1, dW2, dB2 = resizeConvBlockBackward(trace.decoder.rows[i], m.Blocks[i].W1, m.Blocks[i].B1, m.Blocks[i].W2, m.Blocks[i].B2, dx, b, bc, m.Blocks[i].Cout, bh, bw, m.Slope)
		g.add(m.blockTensorName(i, "w1"), dW1)
		g.add(m.blockTensorName(i, "b1"), dB1)
		g.add(m.blockTensorName(i, "w2"), dW2)
		g.add(m.blockTensorName(i, "b2"), dB2)
	}

	// Readout backward into the final state gradient.
	finalState := trace.states[numSteps*stateSize : (numSteps+1)*stateSize]
	dState := make([]float32, stateSize)
	readoutTransformBackwardInto(dState, tot, 0, dx, finalState, b, cfg.N, tot, 0, cfg.Relativization, cfg.Encoding)

	// Euler BPTT through the dynamics.
	dOmega := g.ensure(m.tensorName("omega"), cfg.N)
	dOmegaCond := g.ensure(m.tensorName("omega_cond"), cfg.NCond)
	dK := g.ensure(m.tensorName("k"), cfg.N*cfg.N)
	dKCond := g.ensure(m.tensorName("k_cond"), cfg.NCond*cfg.NCond)
	dDrive := g.ensure(m.tensorName("k_drive"), len(drive))
	dVel := make([]float32, stateSize)
	stepDState := make([]float32, stateSize)
	for s := numSteps - 1; s >= 0; s-- {
		for i := range dVel {
			dVel[i] = float32(cfg.Dt * float64(dState[i]))
		}
		state := trace.states[s*stateSize : (s+1)*stateSize]
		conditionalKuramotoBackwardAccumulate(stepDState, dOmega, dK, dOmegaCond, dKCond, dDrive, state, m.K, m.KCond, drive, dVel, b, cfg.N, cfg.NCond, cfg.KScale, cfg.KCondScale, cfg.KDriveScale)
		for i := range dState {
			dState[i] += stepDState[i]
		}
	}
}

// driftLossGrad: the reference bootstrap objective — per-class constant
// target 0.1 + 0.2*class, squared error summed, dImage = 2*delta.
func driftLossGrad(image []float32, classes, dim int) (float64, []float32) {
	var loss float64
	dImage := make([]float32, len(image))
	for c := 0; c < classes; c++ {
		target := 0.1 + 0.2*float64(c)
		for i := 0; i < dim; i++ {
			index := c*dim + i
			delta := float64(image[index]) - target
			loss += delta * delta
			dImage[index] = float32(2 * delta)
		}
	}
	return loss, dImage
}

func (m *Model) tensorName(suffix string) string { return m.Namespace + "." + suffix }
func (m *Model) blockTensorName(i int, part string) string {
	return fmt.Sprintf("%s.blocks.%d.%s", m.Namespace, i, part)
}

// trainableTensors: name -> (slice, rows, cols). Matrix shapes follow the
// reference layout: couplings/drive/conv weights are Muon matrices, vectors
// and biases are elementwise.
func (m *Model) trainableTensors() (map[string][]float32, map[string][2]int) {
	cfg := m.Cfg
	tensors := map[string][]float32{}
	shapes := map[string][2]int{}
	put := func(name string, values []float32, rows, cols int) {
		tensors[name] = values
		shapes[name] = [2]int{rows, cols}
	}
	put(m.tensorName("omega"), m.Omega, cfg.N, 1)
	put(m.tensorName("omega_cond"), m.OmegaCond, cfg.NCond, 1)
	put(m.tensorName("k"), m.K, cfg.N, cfg.N)
	put(m.tensorName("k_cond"), m.KCond, cfg.NCond, cfg.NCond)
	put(m.tensorName("k_drive"), m.Drive, len(m.Drive)/(cfg.N*cfg.NCond), cfg.N*cfg.NCond)
	for i := range m.Blocks {
		block := &m.Blocks[i]
		put(m.blockTensorName(i, "w1"), block.W1, block.Cout, len(block.W1)/block.Cout)
		put(m.blockTensorName(i, "b1"), block.B1, block.Cout, 1)
		put(m.blockTensorName(i, "w2"), block.W2, block.Cout, len(block.W2)/block.Cout)
		put(m.blockTensorName(i, "b2"), block.B2, block.Cout, 1)
	}
	put(m.tensorName("to_out.weight"), m.ToOutW, cfg.OutChannels, len(m.ToOutW)/cfg.OutChannels)
	put(m.tensorName("to_out.bias"), m.ToOutB, cfg.OutChannels, 1)
	return tensors, shapes
}

// TrainDrift: the reference bootstrap loop — batch = one row per class, phase
// init resampled per step, drift objective, Muon steps at (baseLR, mu).
// Returns the loss trajectory.
func (m *Model) TrainDrift(steps int, baseLR, mu float64, seed int64) ([]float64, error) {
	cfg := m.Cfg
	tensors, shapes := m.trainableTensors()
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	total := 0
	for _, name := range names {
		total += len(tensors[name])
	}
	weights := make([]float32, total)
	gradients := make([]float32, total)
	specs := make([]optimizer.GroupSpec, 0, len(names))
	offset := 0
	for _, name := range names {
		values := tensors[name]
		shape := shapes[name]
		copy(weights[offset:], values)
		specs = append(specs, optimizer.GroupSpec{
			Name: name, Start: offset, End: offset + len(values),
			Rows: shape[0], Cols: shape[1],
		})
		offset += len(values)
	}
	plan, err := optimizer.CompilePlan(total, specs)
	if err != nil {
		return nil, err
	}
	opt, err := optimizer.New(weights, gradients, plan, optimizer.Config{
		BaseLearningRate: baseLR, Momentum: mu, Schedule: optimizer.ScheduleConstant,
	})
	if err != nil {
		return nil, err
	}

	classes := cfg.NClasses
	tot := cfg.N + cfg.NCond
	dim := cfg.OutChannels * cfg.OutH() * cfg.OutW()
	rng := rand.New(rand.NewSource(seed))
	init := make([]float32, classes*tot)
	trajectory := make([]float64, 0, steps)
	for step := 0; step < steps; step++ {
		// Scatter optimizer weights back into the model tensors.
		offset = 0
		for _, name := range names {
			copy(tensors[name], weights[offset:offset+len(tensors[name])])
			offset += len(tensors[name])
		}
		for i := range init {
			init[i] = float32((rng.Float64()*2 - 1) * math.Pi)
		}
		image, trace := m.trainingForwardTrace(init, m.Drive, classes)
		loss, dImage := driftLossGrad(image, classes, dim)
		grads := Grads{}
		m.backwardInto(trace, m.Drive, dImage, grads)
		trajectory = append(trajectory, loss)
		offset = 0
		for _, name := range names {
			copy(gradients[offset:offset+len(tensors[name])], grads[name])
			offset += len(tensors[name])
		}
		opt.Step()
	}
	// Final scatter so the model reflects the last step.
	offset = 0
	for _, name := range names {
		copy(tensors[name], weights[offset:offset+len(tensors[name])])
		offset += len(tensors[name])
	}
	return trajectory, nil
}
