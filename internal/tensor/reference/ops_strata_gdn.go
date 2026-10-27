package reference

import (
	"errors"
	"math"

	"overgo/internal/tensor"
)

// The reference of Strata's gated-delta verify kernels (verify_kernels.cu), in their arithmetic and our state's layout:
// a value head's state [value][key], so value column v's keys sit contiguous at state[(head*width + v)*width:].

// gatedDeltaWindowConv is gdn_conv_l2_body over every token of the window, the history left as it was.
func gatedDeltaWindowConv(node *tensor.Tensor, inputs []Value) (Value, error) {
	attributes, ok := node.Attrs.(tensor.GatedDeltaWindowAttributes)
	if !ok || len(inputs) != tensor.GatedDeltaWindowConvOperands {
		return Value{}, errors.New("gated-delta window convolution inputs are invalid")
	}
	history, qkv := inputs[tensor.GatedDeltaWindowConvHistory].Data, inputs[tensor.GatedDeltaWindowConvQKV].Data
	weights := inputs[tensor.GatedDeltaWindowConvWeights].Data
	channels, tokens := int(node.Shape.Dims[0]), int(node.Shape.Dims[1])
	output := make([]float32, channels*tokens)
	for token := range tokens {
		row := output[token*channels : (token+1)*channels]
		for channel := range row {
			row[channel] = gatedDeltaWindowConvolved(history, qkv, weights, channels, token, channel)
		}
		gatedDeltaWindowNormalize(row, int(attributes.KeyHeads), attributes.Epsilon)
	}
	if attributes.Commit {
		// A one-token window keeps its token: each channel's history becomes [hist1, hist2, x_0].
		rows := tensor.GatedDeltaWindowKernel - 1
		for channel := range channels {
			kept := history[channel*rows : (channel+1)*rows]
			copy(kept, kept[1:])
			kept[rows-1] = qkv[channel]
		}
	}
	return Value{Shape: node.Shape, Data: output}, nil
}

// gatedDeltaWindowConvolved is a channel's SiLU of the taps over the window of [history(3) | qkv's tokens] that ends at
// token.
func gatedDeltaWindowConvolved(history, qkv, weights []float32, channels, token, channel int) float32 {
	taps := tensor.GatedDeltaWindowKernel
	row := func(source int) float32 {
		if source < taps-1 {
			return history[channel*(taps-1)+source]
		}
		return qkv[(source-(taps-1))*channels+channel]
	}
	var sum float32
	for tap := range taps {
		sum += row(token+tap) * weights[channel*taps+tap]
	}
	return sum / (1 + float32(math.Exp(float64(-sum))))
}

// gatedDeltaWindowNormalize scales each of a row's query and key heads by rsqrt(its squared norm + epsilon).
func gatedDeltaWindowNormalize(row []float32, keyHeads int, epsilon float32) {
	width := tensor.GatedDeltaWindowWidth
	for head := range tensor.PairedExtent * keyHeads {
		values := row[head*width : (head+1)*width]
		var squares float32
		for _, value := range values {
			squares += value * value
		}
		scale := float32(1 / math.Sqrt(float64(squares+epsilon)))
		for index := range values {
			values[index] *= scale
		}
	}
}

// gatedDeltaWindowStep is gdn_step_norm_multi with every output over a copy of the state, which it leaves.
func gatedDeltaWindowStep(node *tensor.Tensor, inputs []Value) (Value, error) {
	attributes, ok := node.Attrs.(tensor.GatedDeltaWindowAttributes)
	if !ok || len(inputs) < tensor.GatedDeltaWindowStepOperands || len(inputs) > tensor.GatedDeltaWindowStepOperands+1 {
		return Value{}, errors.New("gated-delta window step inputs are invalid")
	}
	// The kept form runs over the kept tokens on the state itself, which it leaves after them; the every-output form on
	// a copy, which it drops.
	state := append([]float32(nil), inputs[tensor.GatedDeltaWindowStepState].Data...)
	tokens := int(inputs[tensor.GatedDeltaWindowStepRows].Shape.Dims[1])
	if len(inputs) > tensor.GatedDeltaWindowStepOperands {
		state, tokens = inputs[tensor.GatedDeltaWindowStepState].Data, int(inputs[tensor.GatedDeltaWindowStepOperands].Data[0])
	}
	rows, gate := inputs[tensor.GatedDeltaWindowStepRows].Data, inputs[tensor.GatedDeltaWindowStepGate].Data
	beta, z := inputs[tensor.GatedDeltaWindowStepBeta].Data, inputs[tensor.GatedDeltaWindowStepZ].Data
	gamma := inputs[tensor.GatedDeltaWindowStepGamma].Data
	output := make([]float32, len(z))
	gatedDeltaWindowRecur(state, rows, gate, beta, attributes, tokens, func(token, head int, values []float32) {
		width := tensor.GatedDeltaWindowWidth
		var squares float32
		for _, value := range values {
			squares += value * value
		}
		scale := float32(1 / math.Sqrt(float64(squares/float32(width)+attributes.Epsilon)))
		base := (token*int(attributes.ValueHeads) + head) * width
		for column, value := range values {
			gated := 1 / (1 + float32(math.Exp(float64(-z[base+column]))))
			output[base+column] = value * scale * gamma[column] * gated
		}
	})
	return Value{Shape: node.Shape, Data: output}, nil
}

// gatedDeltaCommit is gdn_conv_commit and gdn_step_commit over the kept count, in place on the history and the state.
func gatedDeltaCommit(node *tensor.Tensor, inputs []Value) (Value, error) {
	attributes, ok := node.Attrs.(tensor.GatedDeltaWindowAttributes)
	layers := attributes.CommitLayers()
	if !ok || len(inputs) < layers*tensor.GatedDeltaCommitKept+tensor.SingletonExtent {
		return Value{}, errors.New("gated-delta commit inputs are invalid")
	}
	kept := int(inputs[layers*tensor.GatedDeltaCommitKept].Data[0])
	for layer := range layers {
		operands := inputs[layer*tensor.GatedDeltaCommitKept:]
		history, state := operands[tensor.GatedDeltaCommitHistory].Data, operands[tensor.GatedDeltaCommitState].Data
		qkv, rows := operands[tensor.GatedDeltaCommitQKV].Data, operands[tensor.GatedDeltaCommitRows].Data
		gate, beta := operands[tensor.GatedDeltaCommitGate].Data, operands[tensor.GatedDeltaCommitBeta].Data
		shape := operands[tensor.GatedDeltaCommitRows].Shape
		channels, tokens := int(shape.Dims[0]), int(shape.Dims[1])
		if kept < 0 || kept > tokens {
			return Value{}, errors.New("gated-delta commit keeps more tokens than its window holds")
		}
		if kept == 0 {
			continue
		}
		taps := tensor.GatedDeltaWindowKernel
		for channel := range channels {
			var window [tensor.GatedDeltaWindowKernel - 1]float32
			for row := range window {
				if source := kept + row; source < taps-1 {
					window[row] = history[channel*(taps-1)+source]
				} else {
					window[row] = qkv[(source-(taps-1))*channels+channel]
				}
			}
			copy(history[channel*(taps-1):], window[:])
		}
		gatedDeltaWindowRecur(state, rows, gate, beta, attributes, kept, nil)
	}
	return Value{Shape: node.Shape, Data: []float32{math.Float32frombits(uint32(kept))}}, nil
}

// gatedDeltaWindowRecur runs the recurrence over the first tokens rows in place on state, handing each token's and
// value head's outputs, scaled by rsqrt(width), to emit when it is set: the decay applied with the update, as
// delta = (v - g·(S k)) beta and S = g·S + k delta.
func gatedDeltaWindowRecur(state, rows, gate, beta []float32, attributes tensor.GatedDeltaWindowAttributes, tokens int,
	emit func(token, head int, values []float32)) {
	width := tensor.GatedDeltaWindowWidth
	keyHeads, valueHeads := int(attributes.KeyHeads), int(attributes.ValueHeads)
	channels := (tensor.PairedExtent*keyHeads + valueHeads) * width
	queryKey := keyHeads * width
	scale := float32(1 / math.Sqrt(float64(width)))
	values := make([]float32, width)
	for token := range tokens {
		row := rows[token*channels : (token+1)*channels]
		for head := range valueHeads {
			keyHead := head % keyHeads
			query, key := row[keyHead*width:(keyHead+1)*width], row[queryKey+keyHead*width:queryKey+(keyHead+1)*width]
			value := row[tensor.PairedExtent*queryKey+head*width : tensor.PairedExtent*queryKey+(head+1)*width]
			decay := float32(math.Exp(float64(gate[token*valueHeads+head])))
			strength := beta[token*valueHeads+head]
			for column := range width {
				keys := state[(head*width+column)*width : (head*width+column+1)*width]
				var projected float32
				for index, weight := range keys {
					projected += weight * key[index]
				}
				delta := (value[column] - decay*projected) * strength
				var output float32
				for index := range keys {
					keys[index] = decay*keys[index] + key[index]*delta
					output += keys[index] * query[index]
				}
				values[column] = output * scale
			}
			if emit != nil {
				emit(token, head, values)
			}
		}
	}
}

// copyIndexed is copy_indexed: the target's values become the source's from the index's row on, in place.
func copyIndexed(node *tensor.Tensor, inputs []Value) (Value, error) {
	attributes, ok := node.Attrs.(tensor.CopyIndexedAttributes)
	if !ok || len(inputs) < tensor.CopyIndexedOperands {
		return Value{}, errors.New("indexed copy inputs are invalid")
	}
	target, source := inputs[tensor.CopyIndexedTarget].Data, inputs[tensor.CopyIndexedSource].Data
	index := int(inputs[tensor.CopyIndexedIndex].Data[0])
	if index >= 0 {
		first := index * int(attributes.Stride)
		if first+len(target) > len(source) {
			return Value{}, errors.New("indexed copy reads past its source")
		}
		copy(target, source[first:first+len(target)])
	}
	return Value{Shape: node.Shape, Data: []float32{math.Float32frombits(uint32(int32(index)))}}, nil
}
