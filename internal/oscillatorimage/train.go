package oscillatorimage

import (
	"fmt"

	"overgo/internal/hostmath"
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
	dst := hostmath.GradientSlot(g, name, len(values))
	for i, v := range values {
		dst[i] += v
	}
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
	dOmega := hostmath.GradientSlot(g, m.tensorName("omega"), cfg.N)
	dOmegaCond := hostmath.GradientSlot(g, m.tensorName("omega_cond"), cfg.NCond)
	dK := hostmath.GradientSlot(g, m.tensorName("k"), cfg.N*cfg.N)
	dKCond := hostmath.GradientSlot(g, m.tensorName("k_cond"), cfg.NCond*cfg.NCond)
	dDrive := hostmath.GradientSlot(g, m.tensorName("k_drive"), len(drive))
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

func (m *Model) tensorName(suffix string) string { return m.Namespace + "." + suffix }
func (m *Model) blockTensorName(i int, part string) string {
	return fmt.Sprintf("%s.blocks.%d.%s", m.Namespace, i, part)
}
