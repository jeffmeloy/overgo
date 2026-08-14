//go:build windows

package scratchmodel

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/cuda/device"
	"overgo/internal/devicemath"
	"overgo/internal/hostmath"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// DeviceLossAndGrad composes shared device VJPs over tensor-forward caches.
func (g ForwardGraph) DeviceLossAndGrad(
	worker *device.Worker,
	c Construction,
	weights []float32,
	tokens []int,
	values map[*tensor.Tensor]reference.Value,
) (float64, []float32, error) {
	if worker == nil || len(weights) != len(c.weights) || len(tokens) < g.positions+1 || len(g.layers) != c.config.LayerCount {
		return 0, nil, errors.New("scratch model: device VJP inputs differ")
	}
	data := func(node *tensor.Tensor, size int) ([]float32, error) {
		value, ok := values[node]
		if !ok || len(value.Data) != size {
			return nil, errors.New("scratch model: device VJP cache differs")
		}
		return value.Data, nil
	}
	weight := func(name string) ([]float32, error) {
		binding, ok := c.bindings[name]
		if !ok {
			return nil, fmt.Errorf("scratch model: device VJP weight %q absent", name)
		}
		return weights[binding.start:binding.end], nil
	}
	gradient := make([]float32, len(weights))
	put := func(name string, source []float32) error {
		binding, ok := c.bindings[name]
		if !ok || binding.end-binding.start != len(source) {
			return fmt.Errorf("scratch model: device VJP gradient %q differs", name)
		}
		for index, value := range source {
			gradient[binding.start+index] += value
		}
		return nil
	}

	positions, hidden, vocab := g.positions, c.config.Embedding, c.config.VocabSize
	logits, err := data(g.Output, positions*vocab)
	if err != nil {
		return 0, nil, err
	}
	finalHidden, err := data(g.Hidden, positions*hidden)
	if err != nil {
		return 0, nil, err
	}
	dLogits := make([]float32, len(logits))
	loss := hostmath.SoftmaxCrossEntropy(dLogits, logits, tokens[1:positions+1], positions, vocab)
	head, err := weight("lm_head")
	if err != nil {
		return 0, nil, err
	}
	dHidden, dHead, err := devicemath.LinearBackwardT(worker, finalHidden, head, dLogits, positions, hidden, vocab)
	if err != nil {
		return 0, nil, err
	}
	if err := put("lm_head", dHead); err != nil {
		return 0, nil, err
	}

	for layer := len(g.layers) - 1; layer >= 0; layer-- {
		cache := g.layers[layer]
		prefix := fmt.Sprintf("l%d.", layer)
		attentionOutput, err := data(cache.attentionOutput, positions*hidden)
		if err != nil {
			return 0, nil, err
		}
		activation, err := data(cache.activation, positions*c.config.MLPWidth)
		if err != nil {
			return 0, nil, err
		}
		preactivation, err := data(cache.preactivation, len(activation))
		if err != nil {
			return 0, nil, err
		}
		mlpNorm, err := data(cache.mlpNorm, positions*hidden)
		if err != nil {
			return 0, nil, err
		}
		w2, err := weight(prefix + "w2")
		if err != nil {
			return 0, nil, err
		}
		dActivation, dW2, err := devicemath.LinearBackwardT(worker, activation, w2, dHidden, positions, c.config.MLPWidth, hidden)
		if err != nil {
			return 0, nil, err
		}
		if err := put(prefix+"w2", dW2); err != nil {
			return 0, nil, err
		}
		for index := range dActivation {
			if preactivation[index] <= 0 {
				dActivation[index] = 0
			}
		}
		w1, err := weight(prefix + "w1")
		if err != nil {
			return 0, nil, err
		}
		dMLPNorm, dW1, err := devicemath.LinearBackwardT(worker, mlpNorm, w1, dActivation, positions, hidden, c.config.MLPWidth)
		if err != nil {
			return 0, nil, err
		}
		if err := put(prefix+"w1", dW1); err != nil {
			return 0, nil, err
		}
		dMLPInput, err := devicemath.MADNormBackward(worker, dMLPNorm, attentionOutput, positions, hidden, c.config.Epsilon)
		if err != nil {
			return 0, nil, err
		}
		dAttentionOutput := slices.Clone(dHidden)
		addF32(dAttentionOutput, dMLPInput)

		attention, err := data(cache.attention, positions*hidden)
		if err != nil {
			return 0, nil, err
		}
		wo, err := weight(prefix + "wo")
		if err != nil {
			return 0, nil, err
		}
		dAttention, dWO, err := devicemath.LinearBackwardT(worker, attention, wo, dAttentionOutput, positions, hidden, hidden)
		if err != nil {
			return 0, nil, err
		}
		if err := put(prefix+"wo", dWO); err != nil {
			return 0, nil, err
		}
		dInput := slices.Clone(dAttentionOutput)
		dQKV := make([]float32, positions*3*hidden)
		dBias := make([]float32, c.config.BlockSize)
		dTemperature := make([]float32, c.config.HeadCount)
		temperature, err := weight("lt")
		if err != nil {
			return 0, nil, err
		}
		for headIndex, headCache := range cache.heads {
			q, err := data(headCache.query, positions*c.config.HeadDim)
			if err != nil {
				return 0, nil, err
			}
			k, err := data(headCache.key, len(q))
			if err != nil {
				return 0, nil, err
			}
			v, err := data(headCache.value, len(q))
			if err != nil {
				return 0, nil, err
			}
			probability, err := data(headCache.probability, positions*positions)
			if err != nil {
				return 0, nil, err
			}
			dHead := extractScratchHead(dAttention, positions, c.config.HeadCount, c.config.HeadDim, headIndex)
			scale := math.Exp(-float64(temperature[layer*c.config.HeadCount+headIndex])) / math.Sqrt(float64(c.config.HeadDim))
			grads, err := devicemath.AttentionCoreBackward(worker, q, k, v, probability, dHead, positions, c.config.HeadDim, scale)
			if err != nil {
				return 0, nil, err
			}
			insertScratchQKV(dQKV, grads.DQ, positions, 3*hidden, headIndex*c.config.HeadDim)
			insertScratchQKV(dQKV, grads.DK, positions, 3*hidden, hidden+headIndex*c.config.HeadDim)
			insertScratchQKV(dQKV, grads.DV, positions, 3*hidden, 2*hidden+headIndex*c.config.HeadDim)
			for query := range positions {
				for key := range positions {
					dScore := grads.DScores[query*positions+key]
					if dScore == 0 {
						continue
					}
					lag := min(c.config.BlockSize-1, query-key)
					dBias[lag] += dScore
					var dot float64
					for channel := range c.config.HeadDim {
						dot += float64(q[query*c.config.HeadDim+channel]) * float64(k[key*c.config.HeadDim+channel])
					}
					dTemperature[headIndex] += float32(-scale * float64(dScore) * dot)
				}
			}
		}
		if err := put("pos_bias", dBias); err != nil {
			return 0, nil, err
		}
		ltBinding := c.bindings["lt"]
		for headIndex, value := range dTemperature {
			gradient[ltBinding.start+layer*c.config.HeadCount+headIndex] += value
		}
		qkvNorm, err := data(cache.qkvNorm, positions*hidden)
		if err != nil {
			return 0, nil, err
		}
		wqkv, err := weight(prefix + "wqkv")
		if err != nil {
			return 0, nil, err
		}
		dQKVNorm, dWQKV, err := devicemath.LinearBackwardT(worker, qkvNorm, wqkv, dQKV, positions, hidden, 3*hidden)
		if err != nil {
			return 0, nil, err
		}
		if err := put(prefix+"wqkv", dWQKV); err != nil {
			return 0, nil, err
		}
		input, err := data(cache.input, positions*hidden)
		if err != nil {
			return 0, nil, err
		}
		dQKVInput, err := devicemath.MADNormBackward(worker, dQKVNorm, input, positions, hidden, c.config.Epsilon)
		if err != nil {
			return 0, nil, err
		}
		addF32(dInput, dQKVInput)
		dHidden = dInput
	}

	embedding, err := data(g.embedding, positions*hidden)
	if err != nil {
		return 0, nil, err
	}
	dEmbedding, err := devicemath.MADNormBackward(worker, dHidden, embedding, positions, hidden, c.config.Epsilon)
	if err != nil {
		return 0, nil, err
	}
	wte := c.bindings["wte"]
	wpe := c.bindings["wpe"]
	for position := range positions {
		for channel := range hidden {
			value := dEmbedding[position*hidden+channel]
			gradient[wte.start+tokens[position]*hidden+channel] += value
			gradient[wpe.start+position*hidden+channel] += value
		}
	}
	return loss, gradient, nil
}

func addF32(destination, source []float32) {
	for index, value := range source {
		destination[index] += value
	}
}

func extractScratchHead(source []float32, positions, heads, headDim, head int) []float32 {
	result := make([]float32, positions*headDim)
	for position := range positions {
		copy(result[position*headDim:(position+1)*headDim], source[position*heads*headDim+head*headDim:position*heads*headDim+(head+1)*headDim])
	}
	return result
}

func insertScratchQKV(destination, source []float32, positions, stride, offset int) {
	width := len(source) / positions
	for position := range positions {
		copy(destination[position*stride+offset:position*stride+offset+width], source[position*width:(position+1)*width])
	}
}
