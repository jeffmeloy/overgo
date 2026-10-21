package reference

import (
	"errors"
	"math"

	"overgo/internal/tensor"
)

// gatedDeltaConv is Builder.GatedDeltaConv on the host, rounding as the composed preparation does: ssmConv's float32
// taps in order, the SiLU, and l2Norm's float64 sum and inverse, so its bits are the composed graph's.
func gatedDeltaConv(node *tensor.Tensor, inputs []Value) (Value, error) {
	attributes, ok := node.Attrs.(tensor.GatedDeltaConvAttributes)
	if !ok || len(inputs) != tensor.GatedDeltaConvOperands {
		return Value{}, errors.New("gated-delta convolution operands are invalid")
	}
	state, qkv, weights := inputs[tensor.GatedDeltaConvState], inputs[tensor.GatedDeltaConvQKV], inputs[tensor.GatedDeltaConvTaps]
	layout := tensor.GatedDeltaConvLayout{
		StateWidth: uint64(attributes.StateWidth), KeyHeads: uint64(attributes.KeyHeads), ValueHeads: uint64(attributes.ValueHeads),
		Kernel: weights.Shape.Dims[0], Tokens: qkv.Shape.Dims[1],
	}
	channels, kernel, tokens, width := int(layout.Channels()), int(layout.Kernel), int(layout.Tokens), int(layout.StateWidth)
	history := kernel - 1
	if len(state.Data) != history*channels || len(qkv.Data) != channels*tokens || len(weights.Data) != kernel*channels {
		return Value{}, errors.New("gated-delta convolution shapes are invalid")
	}
	// row is the convolution input's row at time index of a channel: the history's rows, then the tokens'.
	row := func(index, channel int) float32 {
		if index < history {
			return state.Data[channel*history+index]
		}
		return qkv.Data[(index-history)*channels+channel]
	}
	output := make([]float32, layout.Elements())
	keyChannels := int(layout.KeyHeads) * width
	for token := range tokens {
		convolved := make([]float32, channels)
		for channel := range channels {
			var sum float32
			for tap := range kernel {
				sum += row(token+tap, channel) * weights.Data[channel*kernel+tap]
			}
			convolved[channel] = sum / (1 + float32(math.Exp(float64(-sum))))
			for index := range history {
				output[int(layout.Window(uint64(token)))+channel*history+index] = row(token+1+index, channel)
			}
		}
		// The query's and the key's heads take their L2 norms; the value's are as convolved.
		for head := range channels / width {
			values := convolved[head*width : (head+1)*width]
			if head*width < tensor.PairedExtent*keyChannels {
				var sumSquares float64
				for _, value := range values {
					sumSquares += float64(value) * float64(value)
				}
				inverse := float32(1 / max(math.Sqrt(sumSquares), float64(attributes.Epsilon)))
				for index, value := range values {
					values[index] = value * inverse
				}
			}
		}
		for channel, value := range convolved {
			switch {
			case channel < keyChannels:
				output[token*keyChannels+channel] = value
			case channel < tensor.PairedExtent*keyChannels:
				output[int(layout.Key())+token*keyChannels+channel-keyChannels] = value
			default:
				valueChannels := channels - tensor.PairedExtent*keyChannels
				output[int(layout.Value())+token*valueChannels+channel-tensor.PairedExtent*keyChannels] = value
			}
		}
	}
	return Value{Shape: node.Shape, Data: output}, nil
}

// gatedDeltaGates is Builder.GatedDeltaGates on the host, rounding as the composed gates do: mulMat's float64 sums,
// the time step's addition, Softplus, the decay's product, and Sigmoid.
func gatedDeltaGates(node *tensor.Tensor, inputs []Value) (Value, error) {
	if len(inputs) != tensor.GatedDeltaGatesOperands {
		return Value{}, errors.New("gated-delta gate operands are invalid")
	}
	input, alpha, beta := inputs[tensor.GatedDeltaGatesInput], inputs[tensor.GatedDeltaGatesAlpha], inputs[tensor.GatedDeltaGatesBeta]
	timeStep, decay := inputs[tensor.GatedDeltaGatesTimeStep], inputs[tensor.GatedDeltaGatesDecay]
	embedding, heads, tokens := int(input.Shape.Dims[0]), int(alpha.Shape.Dims[1]), int(input.Shape.Dims[1])
	if len(alpha.Data) != embedding*heads || len(beta.Data) != embedding*heads || len(input.Data) != embedding*tokens ||
		len(timeStep.Data) != heads || len(decay.Data) != heads {
		return Value{}, errors.New("gated-delta gate shapes are invalid")
	}
	output := make([]float32, tensor.PairedExtent*heads*tokens)
	for token := range tokens {
		column := input.Data[token*embedding : (token+1)*embedding]
		for head := range heads {
			value := productDot(alpha.Data[head*embedding:(head+1)*embedding], column) + timeStep.Data[head]
			softplus := float32(max(float64(value), 0) + math.Log1p(math.Exp(-math.Abs(float64(value)))))
			output[token*heads+head] = softplus * decay.Data[head]
			product := productDot(beta.Data[head*embedding:(head+1)*embedding], column)
			output[heads*tokens+token*heads+head] = 1 / (1 + float32(math.Exp(float64(-product))))
		}
	}
	return Value{Shape: node.Shape, Data: output}, nil
}

// productDot is one product element as mulMat sums it.
func productDot(weights, column []float32) float32 {
	var sum float64
	for index, weight := range weights {
		sum += float64(weight) * float64(column[index])
	}
	return float32(sum)
}
