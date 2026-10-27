//go:build windows

package executor

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// TestStrataGatedDeltaWindowParity holds Strata's gated-delta verify kernels (strata_gdn.cuh, from verify_kernels.cu)
// to gdn_parity.cpp's sections 5 and 6 and its rival readings, over two key heads and four value heads in our state's
// layout: for every window of 1 to 8 tokens the window's conv and its every-output recurrence give, bit for bit, the
// rows and outputs that one-token windows give in turn, each keeping its token (the conv's own commit, the kept form's
// count of one), and leave the history and the state as they were; the kept form over the window gives the same outputs
// and the one-token windows' state; a commit of the window's tokens, or of all but its last, leaves the history and the
// state those one-token windows leave; the rows, outputs and state are a double-precision transcription's within a
// bound; and the readings gdn_parity pins - value head h reads key head h modulo the key heads, the decay applied with
// the update, the conv's oldest tap on the oldest row, the norm's epsilon on the squared norm - each differ observably
// from their rival.
func TestStrataGatedDeltaWindowParity(t *testing.T) {
	const (
		keyHeads   = 2
		valueHeads = 4
		epsilon    = float32(1e-6)
		// bound is the relative L1 distance the kernels keep from the double-precision transcription.
		bound = 1e-5
		// rival is the relative L1 distance a rival reading must keep from the kernels.
		rival = 1e-3
	)
	width := strataGatedDeltaWidth
	channels := (2*keyHeads + valueHeads) * width
	valueWords, stateWords := width*valueHeads, width*width*valueHeads
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	draws := rand.New(rand.NewPCG(7, 11))
	fill := func(count int, draw func() float32) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = draw()
		}
		return values
	}
	gauss := func(scale float32) func() float32 {
		return func() float32 { return float32(draws.NormFloat64()) * scale }
	}
	state0 := fill(stateWords, gauss(0.1))
	history0 := fill(3*channels, gauss(0.5))
	weights := fill(4*channels, gauss(0.5))
	gamma := fill(width, func() float32 { return 1 + 0.1*float32(draws.NormFloat64()) })
	weightsBuffer, gammaBuffer := buffers.upload(t, float32Bytes(weights)), buffers.upload(t, float32Bytes(gamma))
	read := func(pointer driver.DevicePtr, words int) []float32 {
		return float32s(buffers.read(t, pointer, 4*words))
	}
	count := func(n int) driver.DevicePtr { return buffers.upload(t, int32Bytes([]int32{int32(n)})) }
	run := func(launch func(*device.State, functionSet) error) { buffers.run(t, launch) }
	row := func(values []float32, token, size int) []float32 { return values[token*size : (token+1)*size] }
	type window struct{ qkv, rows, gate, beta, z []float32 }
	upload := func(w window, tokens int) strataGatedDeltaWindow {
		return strataGatedDeltaWindow{
			qkv: buffers.upload(t, float32Bytes(w.qkv)), rows: buffers.upload(t, float32Bytes(w.rows)),
			gate: buffers.upload(t, float32Bytes(w.gate)), beta: buffers.upload(t, float32Bytes(w.beta)),
			z: buffers.upload(t, float32Bytes(w.z)), outputs: buffers.zeros(t, 4*valueWords*tokens),
		}
	}
	single := func(w window, token int) window {
		return window{
			qkv: row(w.qkv, token, channels), rows: row(w.rows, token, channels), gate: row(w.gate, token, valueHeads),
			beta: row(w.beta, token, valueHeads), z: row(w.z, token, valueWords),
		}
	}
	conv := func(history driver.DevicePtr, w strataGatedDeltaWindow, tokens int, commit bool) {
		run(func(state *device.State, functions functionSet) error {
			return launchStrataGatedDeltaConv(state, functions, history, weightsBuffer, w, int32(channels), keyHeads, epsilon, 0, int32(tokens), commit)
		})
	}
	step := func(stateBuffer, kept driver.DevicePtr, w strataGatedDeltaWindow, tokens int) {
		run(func(state *device.State, functions functionSet) error {
			return launchStrataGatedDeltaStep(state, functions, stateBuffer, gammaBuffer, kept, w, int32(channels), keyHeads, valueHeads, epsilon, int32(tokens), 0)
		})
	}
	commit := func(history, stateBuffer driver.DevicePtr, w strataGatedDeltaWindow, kept int) {
		keptBuffer := count(kept)
		run(func(state *device.State, functions functionSet) error {
			return launchStrataGatedDeltaCommit(state, functions, history, stateBuffer, keptBuffer, w, int32(channels), keyHeads, valueHeads)
		})
	}
	distance := func(got []float32, want []float64) float64 { return strataRelativeL1(float32To64(got), want) }

	for tokens := 1; tokens <= 8; tokens++ {
		w := window{
			qkv: fill(channels*tokens, gauss(1)), rows: fill(channels*tokens, gauss(0.1)),
			gate: fill(valueHeads*tokens, func() float32 { return -(0.5 + 3*draws.Float32()) }),
			beta: fill(valueHeads*tokens, draws.Float32), z: fill(valueWords*tokens, gauss(1)),
		}
		// The window: its conv into its own rows, its recurrence over the given rows, from the untouched history and state.
		convWindow, stepWindow := upload(w, tokens), upload(w, tokens)
		history, stateBuffer := buffers.upload(t, float32Bytes(history0)), buffers.upload(t, float32Bytes(state0))
		conv(history, convWindow, tokens, false)
		step(stateBuffer, 0, stepWindow, tokens)
		convRows, outputs := read(convWindow.rows, channels*tokens), read(stepWindow.outputs, valueWords*tokens)
		if !slices.Equal(read(history, len(history0)), history0) || !slices.Equal(read(stateBuffer, stateWords), state0) {
			t.Fatalf("T=%d: the window wrote its history or its state", tokens)
		}
		// One-token windows in turn, each keeping its token: the conv's own commit, the kept form's count of one.
		chainHistory, chainState := buffers.upload(t, float32Bytes(history0)), buffers.upload(t, float32Bytes(state0))
		histories, states := [][]float32{history0}, [][]float32{state0}
		for token := range tokens {
			one := upload(single(w, token), 1)
			conv(chainHistory, one, 1, true)
			if !slices.Equal(read(one.rows, channels), row(convRows, token, channels)) {
				t.Fatalf("T=%d: the window's conv at token %d differs from a one-token window's", tokens, token)
			}
			one = upload(single(w, token), 1)
			step(chainState, count(1), one, 1)
			if !slices.Equal(read(one.outputs, valueWords), row(outputs, token, valueWords)) {
				t.Fatalf("T=%d: the window's output at token %d differs from a one-token window's", tokens, token)
			}
			histories, states = append(histories, read(chainHistory, len(history0))), append(states, read(chainState, stateWords))
		}
		// The kept form over the window: its outputs and the one-token windows' state.
		keptWindow := upload(w, tokens)
		keptState := buffers.upload(t, float32Bytes(state0))
		step(keptState, count(tokens), keptWindow, tokens)
		if !slices.Equal(read(keptWindow.outputs, valueWords*tokens), outputs) || !slices.Equal(read(keptState, stateWords), states[tokens]) {
			t.Fatalf("T=%d: the kept form's outputs or state differ from the one-token windows'", tokens)
		}
		// A commit of the window's tokens, and of all but its last.
		for _, kept := range []int{tokens, tokens - 1} {
			if kept == 0 {
				continue
			}
			committedHistory, committedState := buffers.upload(t, float32Bytes(history0)), buffers.upload(t, float32Bytes(state0))
			commit(committedHistory, committedState, upload(w, tokens), kept)
			if !slices.Equal(read(committedHistory, len(history0)), histories[kept]) || !slices.Equal(read(committedState, stateWords), states[kept]) {
				t.Fatalf("T=%d: a commit of %d tokens differs from %d one-token windows", tokens, kept, kept)
			}
		}
		// The double-precision transcriptions and the rival readings.
		if d := distance(convRows, gdnConvTranscription(history0, w.qkv, weights, keyHeads, epsilon, tokens, false, false)); d > bound {
			t.Fatalf("T=%d: the conv is %.3g from the transcription", tokens, d)
		}
		wantState, wantOutputs := gdnTranscription(state0, w.rows, w.gate, w.beta, w.z, gamma, keyHeads, valueHeads, epsilon, tokens, true, true)
		if d := distance(outputs, wantOutputs); d > bound {
			t.Fatalf("T=%d: the outputs are %.3g from the transcription", tokens, d)
		}
		if d := distance(states[tokens], wantState); d > bound {
			t.Fatalf("T=%d: the state is %.3g from the transcription", tokens, d)
		}
		for name, reading := range map[string][2]bool{"interleaved heads": {false, true}, "the decay after the update": {true, false}} {
			rivalState, _ := gdnTranscription(state0, w.rows, w.gate, w.beta, w.z, gamma, keyHeads, valueHeads, epsilon, tokens, reading[0], reading[1])
			if d := distance(states[tokens], rivalState); d < rival {
				t.Fatalf("T=%d: the rival reading %q is %.3g from the kernels' state, not observable", tokens, name, d)
			}
		}
	}
	// The conv's slide and the norm's epsilon: the first query head's rows and history scaled so its squared norm nears
	// the epsilon, read with the taps reversed and with the epsilon on the mean.
	tiny, tinyHistory := fill(channels, gauss(1)), slices.Clone(history0)
	for channel := range width {
		tiny[channel] *= 1e-4
		for slot := range 3 {
			tinyHistory[channel*3+slot] *= 1e-4
		}
	}
	tinyWindow := upload(window{qkv: tiny, rows: tiny, gate: make([]float32, valueHeads), beta: make([]float32, valueHeads), z: make([]float32, valueWords)}, 1)
	conv(buffers.upload(t, float32Bytes(tinyHistory)), tinyWindow, 1, false)
	got := read(tinyWindow.rows, channels)
	if d := distance(got, gdnConvTranscription(tinyHistory, tiny, weights, keyHeads, epsilon, 1, false, false)); d > bound {
		t.Fatalf("the conv is %.3g from the transcription", d)
	}
	for name, reading := range map[string][2]bool{"the taps reversed": {true, false}, "the epsilon on the mean": {false, true}} {
		if d := distance(got, gdnConvTranscription(tinyHistory, tiny, weights, keyHeads, epsilon, 1, reading[0], reading[1])); d < rival {
			t.Fatalf("the rival reading %q is %.3g from the kernels' conv, not observable", name, d)
		}
	}
}

// TestStrataGatedDeltaCommitLayersMatchTheLayers holds Strata's commit of every gated-delta layer in one launch of each
// kernel (gdn_commit_layers: gdn_conv_commit_layers_kernel and gdn_step_commit_layers_kernel, a table of each layer's
// addresses where Strata strides) to each layer's own commit, bit for bit: over more layers than one launch's table
// holds, for windows of two and five tokens, a commit of the window's tokens, of all but its last and of none leaves every
// layer's history and state as that layer's own launches leave them.
func TestStrataGatedDeltaCommitLayersMatchTheLayers(t *testing.T) {
	const (
		keyHeads   = 2
		valueHeads = 4
		// layers spans a launch's table and part of the next.
		layers = strataGatedDeltaCommitLayers + 6
	)
	width := strataGatedDeltaWidth
	channels := (2*keyHeads + valueHeads) * width
	stateWords, historyWords := width*width*valueHeads, 3*channels
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	draws := rand.New(rand.NewPCG(13, 17))
	fill := func(count int, scale float32) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = float32(draws.NormFloat64()) * scale
		}
		return values
	}
	type layer struct {
		history, state []float32
		qkv, rows      []float32
		gate, beta     []float32
	}
	for _, tokens := range []int{2, 5} {
		fixtures := make([]layer, layers)
		for index := range fixtures {
			gate, beta := make([]float32, valueHeads*tokens), make([]float32, valueHeads*tokens)
			for entry := range gate {
				gate[entry], beta[entry] = -(0.5 + 3*draws.Float32()), draws.Float32()
			}
			fixtures[index] = layer{
				history: fill(historyWords, 0.5), state: fill(stateWords, 0.1),
				qkv: fill(channels*tokens, 1), rows: fill(channels*tokens, 0.1), gate: gate, beta: beta,
			}
		}
		for _, kept := range []int{tokens, tokens - 1, 0} {
			keptBuffer := buffers.upload(t, int32Bytes([]int32{int32(kept)}))
			upload := func() (histories, states []driver.DevicePtr, windows []strataGatedDeltaWindow) {
				for _, fixture := range fixtures {
					histories = append(histories, buffers.upload(t, float32Bytes(fixture.history)))
					states = append(states, buffers.upload(t, float32Bytes(fixture.state)))
					windows = append(windows, strataGatedDeltaWindow{
						qkv: buffers.upload(t, float32Bytes(fixture.qkv)), rows: buffers.upload(t, float32Bytes(fixture.rows)),
						gate: buffers.upload(t, float32Bytes(fixture.gate)), beta: buffers.upload(t, float32Bytes(fixture.beta)),
					})
				}
				return histories, states, windows
			}
			ownHistories, ownStates, ownWindows := upload()
			buffers.run(t, func(state *device.State, functions functionSet) error {
				for index := range fixtures {
					if err := launchStrataGatedDeltaCommit(state, functions, ownHistories[index], ownStates[index], keptBuffer,
						ownWindows[index], int32(channels), keyHeads, valueHeads); err != nil {
						return err
					}
				}
				return nil
			})
			histories, states, windows := upload()
			buffers.run(t, func(state *device.State, functions functionSet) error {
				return launchStrataGatedDeltaCommitLayers(state, functions, histories, states, keptBuffer, windows,
					int32(channels), keyHeads, valueHeads)
			})
			for index, fixture := range fixtures {
				ownHistory, history := buffers.read(t, ownHistories[index], 4*historyWords), buffers.read(t, histories[index], 4*historyWords)
				ownState, state := buffers.read(t, ownStates[index], 4*stateWords), buffers.read(t, states[index], 4*stateWords)
				if !slices.Equal(history, ownHistory) || !slices.Equal(state, ownState) {
					t.Fatalf("T=%d kept %d: layer %d's history or state differs from its own commit's", tokens, kept, index)
				}
				if kept > 0 && slices.Equal(float32s(state), fixture.state) {
					t.Fatalf("T=%d kept %d: layer %d's state was not advanced", tokens, kept, index)
				}
				if kept == 0 && (!slices.Equal(float32s(state), fixture.state) || !slices.Equal(float32s(history), fixture.history)) {
					t.Fatalf("T=%d: a commit of none changed layer %d", tokens, index)
				}
			}
		}
	}
}

// gdnTranscription is gdn_parity's ref_step in double precision over the window's tokens, its state in our layout (a
// value head's [value][key]), each output scaled by rsqrt(128), normed by its head's mean square (gamma, epsilon) and
// gated by sigmoid(z). modulo pairs value head h with key head h modulo the key heads, else h / (value heads / key
// heads); decayFirst decays the state before the rank-one update, else after it.
func gdnTranscription(state, rows, gate, beta, z, gamma []float32, keyHeads, valueHeads int, epsilon float32, tokens int,
	modulo, decayFirst bool) ([]float64, []float64) {
	width := strataGatedDeltaWidth
	channels := (2*keyHeads + valueHeads) * width
	values := float32To64(state)
	outputs := make([]float64, tokens*valueHeads*width)
	for token := range tokens {
		row := rows[token*channels : (token+1)*channels]
		for head := range valueHeads {
			source := head % keyHeads
			if !modulo {
				source = head / (valueHeads / keyHeads)
			}
			query, key := row[source*width:], row[keyHeads*width+source*width:]
			value := row[2*keyHeads*width+head*width:]
			decay, strength := math.Exp(float64(gate[token*valueHeads+head])), float64(beta[token*valueHeads+head])
			output := outputs[(token*valueHeads+head)*width : (token*valueHeads+head+1)*width]
			for column := range width {
				cell := values[(head*width+column)*width : (head*width+column+1)*width]
				if decayFirst {
					for index := range cell {
						cell[index] *= decay
					}
				}
				var projected float64
				for index := range cell {
					projected += cell[index] * float64(key[index])
				}
				delta := (float64(value[column]) - projected) * strength
				for index := range cell {
					cell[index] += float64(key[index]) * delta
					if !decayFirst {
						cell[index] *= decay
					}
					output[column] += cell[index] * float64(query[index])
				}
				output[column] /= math.Sqrt(float64(width))
			}
			var squares float64
			for _, value := range output {
				squares += value * value
			}
			scale := 1 / math.Sqrt(squares/float64(width)+float64(epsilon))
			for column := range output {
				gated := 1 / (1 + math.Exp(-float64(z[(token*valueHeads+head)*width+column])))
				output[column] *= scale * float64(gamma[column]) * gated
			}
		}
	}
	return values, outputs
}

// gdnConvTranscription is the window's conv over [history | qkv's tokens] in double precision, a channel's oldest tap
// on its oldest row (on its newest when reversed), each query and key head divided by sqrt(its squared norm + epsilon)
// (its mean square + epsilon when onMean).
func gdnConvTranscription(history, qkv, weights []float32, keyHeads int, epsilon float32, tokens int, reversed, onMean bool) []float64 {
	width := strataGatedDeltaWidth
	channels := len(qkv) / tokens
	output := make([]float64, len(qkv))
	for token := range tokens {
		row := output[token*channels : (token+1)*channels]
		for channel := range channels {
			var sum float64
			for tap := range 4 {
				source, value := token+tap, 0.0
				if source < 3 {
					value = float64(history[channel*3+source])
				} else {
					value = float64(qkv[(source-3)*channels+channel])
				}
				weight := tap
				if reversed {
					weight = 3 - tap
				}
				sum += value * float64(weights[channel*4+weight])
			}
			row[channel] = sum / (1 + math.Exp(-sum))
		}
		for head := range 2 * keyHeads {
			values := row[head*width : (head+1)*width]
			var squares float64
			for _, value := range values {
				squares += value * value
			}
			if onMean {
				squares /= float64(width)
			}
			for index := range values {
				values[index] /= math.Sqrt(squares + float64(epsilon))
			}
		}
	}
	return output
}
