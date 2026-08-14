package hostmath

import (
	"errors"
	"math"
)

// MADTransformerConfig is a causal transformer using rank/MAD normalization.
type MADTransformerConfig struct {
	Vocab, Block, Hidden, Heads, HeadDim, Layers, Intermediate, Window int
	Epsilon                                                            float64
}

// MADTransformerLayer binds one layer's parameter slabs.
type MADTransformerLayer struct {
	QKV, Output, Expand, Contract []float32
}

// MADTransformer binds parameter or gradient slabs without copying.
type MADTransformer struct {
	Config                    MADTransformerConfig
	Token, Head, Position     []float32
	PositionBias, Temperature []float32
	Layer                     []MADTransformerLayer
}

type madState struct {
	output, a, b []float32
	c, inverse   []float64
}

type madLayerCache struct {
	input, qkv, attention, attentionOutput, activated []float32
	qkvNorm, mlpNorm                                  madState
	probabilities                                     [][]float32
}

type madForward struct {
	loss        float64
	hidden      []float32
	initial     madState
	layers      []madLayerCache
	probability []float32
}

// MADNorm applies rank/MAD normalization independently to each row.
func MADNorm(input []float32, rows, width int, epsilon float64) ([]float32, error) {
	if rows <= 0 || width <= 0 || len(input) != rows*width || epsilon <= 0 || math.IsNaN(epsilon) || math.IsInf(epsilon, 0) {
		return nil, errors.New("mad norm: invalid geometry or epsilon")
	}
	output, _ := madForwardRows(input, rows, width, epsilon)
	return output, nil
}

// MADNormBackward returns the input VJP for rank/MAD normalization.
func MADNormBackward(incoming, input []float32, rows, width int, epsilon float64) ([]float32, error) {
	if rows <= 0 || width <= 0 || len(input) != rows*width || epsilon <= 0 || math.IsNaN(epsilon) || math.IsInf(epsilon, 0) {
		return nil, errors.New("mad norm: invalid geometry or epsilon")
	}
	if len(incoming) != len(input) {
		return nil, errors.New("mad norm: incoming geometry differs")
	}
	_, state := madForwardRows(input, rows, width, epsilon)
	return madBackwardRows(incoming, state, rows, width), nil
}

// MADTransformerLoss evaluates mean causal cross entropy.
func MADTransformerLoss(model MADTransformer, tokens []int) (float64, error) {
	forward, err := madTransformerForward(model, tokens)
	return forward.loss, err
}

// MADTransformerLossAndGrad evaluates and replaces the complete gradient slab.
func MADTransformerLossAndGrad(model, gradient MADTransformer, tokens []int) (float64, error) {
	if err := validateMADTransformer(model); err != nil {
		return 0, err
	}
	if err := validateMADTransformer(gradient); err != nil {
		return 0, errors.New("mad transformer: gradient geometry differs")
	}
	if model.Config != gradient.Config {
		return 0, errors.New("mad transformer: gradient config differs")
	}
	clearMADTransformer(gradient)
	forward, err := madTransformerForward(model, tokens)
	if err != nil {
		return 0, err
	}
	cfg := model.Config
	positions := len(tokens) - 1
	dHidden := make([]float32, positions*cfg.Hidden)
	for position := range positions {
		hidden := forward.hidden[position*cfg.Hidden : (position+1)*cfg.Hidden]
		probability := forward.probability[position*cfg.Vocab : (position+1)*cfg.Vocab]
		for token := range cfg.Vocab {
			dLogit := float64(probability[token])
			if token == tokens[position+1] {
				dLogit--
			}
			dLogit /= float64(positions)
			head := model.Head[token*cfg.Hidden : (token+1)*cfg.Hidden]
			dHead := gradient.Head[token*cfg.Hidden : (token+1)*cfg.Hidden]
			for channel := range cfg.Hidden {
				dHead[channel] += float32(dLogit * float64(hidden[channel]))
				dHidden[position*cfg.Hidden+channel] += float32(dLogit * float64(head[channel]))
			}
		}
	}

	for layer := cfg.Layers - 1; layer >= 0; layer-- {
		weights, grads, cache := model.Layer[layer], gradient.Layer[layer], forward.layers[layer]
		dAttentionOutput := append([]float32(nil), dHidden...)
		dActivated := make([]float32, positions*cfg.Intermediate)
		linearBackward(dActivated, grads.Contract, cache.activated, weights.Contract, dHidden, positions, cfg.Intermediate, cfg.Hidden)
		for index := range dActivated {
			if cache.activated[index] == 0 {
				dActivated[index] = 0
			}
		}
		dMLPNorm := make([]float32, positions*cfg.Hidden)
		linearBackward(dMLPNorm, grads.Expand, cache.mlpNorm.output, weights.Expand, dActivated, positions, cfg.Hidden, cfg.Intermediate)
		addInPlace(dAttentionOutput, madBackwardRows(dMLPNorm, cache.mlpNorm, positions, cfg.Hidden))

		dInput := append([]float32(nil), dAttentionOutput...)
		dAttention := make([]float32, positions*cfg.Hidden)
		linearBackward(dAttention, grads.Output, cache.attention, weights.Output, dAttentionOutput, positions, cfg.Hidden, cfg.Hidden)
		dQKV := attentionBackward(model, gradient, layer, cache, dAttention, positions)
		dQKVNorm := make([]float32, positions*cfg.Hidden)
		linearBackward(dQKVNorm, grads.QKV, cache.qkvNorm.output, weights.QKV, dQKV, positions, cfg.Hidden, 3*cfg.Hidden)
		addInPlace(dInput, madBackwardRows(dQKVNorm, cache.qkvNorm, positions, cfg.Hidden))
		dHidden = dInput
	}

	dEmbedding := madBackwardRows(dHidden, forward.initial, positions, cfg.Hidden)
	for position := range positions {
		token := tokens[position]
		for channel := range cfg.Hidden {
			value := dEmbedding[position*cfg.Hidden+channel]
			gradient.Token[token*cfg.Hidden+channel] += value
			gradient.Position[position*cfg.Hidden+channel] += value
		}
	}
	return forward.loss, nil
}

func madTransformerForward(model MADTransformer, tokens []int) (madForward, error) {
	if err := validateMADTransformer(model); err != nil {
		return madForward{}, err
	}
	cfg := model.Config
	if len(tokens) < 2 {
		return madForward{}, errors.New("mad transformer: at least two tokens required")
	}
	positions := min(cfg.Block, len(tokens)-1)
	for _, token := range tokens[:positions+1] {
		if token < 0 || token >= cfg.Vocab {
			return madForward{}, errors.New("mad transformer: token outside vocabulary")
		}
	}
	embedding := make([]float32, positions*cfg.Hidden)
	for position := range positions {
		for channel := range cfg.Hidden {
			embedding[position*cfg.Hidden+channel] = model.Token[tokens[position]*cfg.Hidden+channel] + model.Position[position*cfg.Hidden+channel]
		}
	}
	hidden, initial := madForwardRows(embedding, positions, cfg.Hidden, cfg.Epsilon)
	result := madForward{initial: initial, layers: make([]madLayerCache, cfg.Layers)}
	for layer := range cfg.Layers {
		weights := model.Layer[layer]
		cache := madLayerCache{input: hidden}
		_, cache.qkvNorm = madForwardRows(hidden, positions, cfg.Hidden, cfg.Epsilon)
		cache.qkv = linearForward(cache.qkvNorm.output, weights.QKV, positions, cfg.Hidden, 3*cfg.Hidden)
		cache.attention, cache.probabilities = attentionForward(model, layer, cache.qkv, positions)
		projected := linearForward(cache.attention, weights.Output, positions, cfg.Hidden, cfg.Hidden)
		cache.attentionOutput = addCopy(hidden, projected)
		_, cache.mlpNorm = madForwardRows(cache.attentionOutput, positions, cfg.Hidden, cfg.Epsilon)
		cache.activated = linearForward(cache.mlpNorm.output, weights.Expand, positions, cfg.Hidden, cfg.Intermediate)
		for index := range cache.activated {
			cache.activated[index] = max(cache.activated[index], 0)
		}
		hidden = addCopy(cache.attentionOutput, linearForward(cache.activated, weights.Contract, positions, cfg.Intermediate, cfg.Hidden))
		result.layers[layer] = cache
	}
	result.hidden = hidden
	result.probability = make([]float32, positions*cfg.Vocab)
	for position := range positions {
		logits := make([]float64, cfg.Vocab)
		maximum := -math.MaxFloat64
		for token := range cfg.Vocab {
			for channel := range cfg.Hidden {
				logits[token] += float64(model.Head[token*cfg.Hidden+channel]) * float64(hidden[position*cfg.Hidden+channel])
			}
			maximum = max(maximum, logits[token])
		}
		var denominator float64
		for token := range cfg.Vocab {
			logits[token] = math.Exp(logits[token] - maximum)
			denominator += logits[token]
		}
		for token := range cfg.Vocab {
			probability := logits[token] / denominator
			result.probability[position*cfg.Vocab+token] = float32(probability)
			if token == tokens[position+1] {
				result.loss -= math.Log(max(probability, cfg.Epsilon)) / float64(positions)
			}
		}
	}
	return result, nil
}

func attentionForward(model MADTransformer, layer int, qkv []float32, positions int) ([]float32, [][]float32) {
	cfg := model.Config
	output := make([]float32, positions*cfg.Hidden)
	probabilities := make([][]float32, positions)
	copy(output[:cfg.Hidden], qkv[2*cfg.Hidden:3*cfg.Hidden])
	for position := 1; position < positions; position++ {
		start := max(0, position+1-cfg.Window)
		window := position + 1 - start
		probabilities[position] = make([]float32, cfg.Heads*window)
		for head := range cfg.Heads {
			scale := math.Exp(-float64(model.Temperature[layer*cfg.Heads+head])) / math.Sqrt(float64(cfg.HeadDim))
			row := probabilities[position][head*window : (head+1)*window]
			maximum := -math.MaxFloat64
			for offset := range window {
				keyPosition := start + offset
				var dot float64
				for channel := range cfg.HeadDim {
					q := qkv[position*3*cfg.Hidden+head*cfg.HeadDim+channel]
					k := qkv[keyPosition*3*cfg.Hidden+cfg.Hidden+head*cfg.HeadDim+channel]
					dot += float64(q) * float64(k)
				}
				lag := min(cfg.Block-1, position-keyPosition)
				logit := dot*scale + float64(model.PositionBias[lag])
				row[offset] = float32(logit)
				maximum = max(maximum, logit)
			}
			var denominator float64
			for offset := range window {
				value := math.Exp(float64(row[offset]) - maximum)
				row[offset] = float32(value)
				denominator += value
			}
			for offset := range window {
				row[offset] = float32(float64(row[offset]) / denominator)
				keyPosition := start + offset
				for channel := range cfg.HeadDim {
					out := position*cfg.Hidden + head*cfg.HeadDim + channel
					value := qkv[keyPosition*3*cfg.Hidden+2*cfg.Hidden+head*cfg.HeadDim+channel]
					output[out] += row[offset] * value
				}
			}
		}
	}
	return output, probabilities
}

func attentionBackward(model, gradient MADTransformer, layer int, cache madLayerCache, incoming []float32, positions int) []float32 {
	cfg := model.Config
	result := make([]float32, len(cache.qkv))
	copy(result[2*cfg.Hidden:3*cfg.Hidden], incoming[:cfg.Hidden])
	for position := positions - 1; position >= 1; position-- {
		start := max(0, position+1-cfg.Window)
		window := position + 1 - start
		for head := range cfg.Heads {
			scale := math.Exp(-float64(model.Temperature[layer*cfg.Heads+head])) / math.Sqrt(float64(cfg.HeadDim))
			var outputDot float64
			for channel := range cfg.HeadDim {
				index := position*cfg.Hidden + head*cfg.HeadDim + channel
				outputDot += float64(cache.attention[index]) * float64(incoming[index])
			}
			for offset := range window {
				keyPosition := start + offset
				probability := float64(cache.probabilities[position][head*window+offset])
				var valueDot, queryKeyDot float64
				for channel := range cfg.HeadDim {
					out := position*cfg.Hidden + head*cfg.HeadDim + channel
					valueIndex := keyPosition*3*cfg.Hidden + 2*cfg.Hidden + head*cfg.HeadDim + channel
					valueDot += float64(cache.qkv[valueIndex]) * float64(incoming[out])
					result[valueIndex] += float32(probability * float64(incoming[out]))
				}
				dLogit := probability * (valueDot - outputDot)
				lag := min(cfg.Block-1, position-keyPosition)
				gradient.PositionBias[lag] += float32(dLogit)
				for channel := range cfg.HeadDim {
					queryIndex := position*3*cfg.Hidden + head*cfg.HeadDim + channel
					keyIndex := keyPosition*3*cfg.Hidden + cfg.Hidden + head*cfg.HeadDim + channel
					query, key := cache.qkv[queryIndex], cache.qkv[keyIndex]
					queryKeyDot += float64(query) * float64(key)
					result[queryIndex] += float32(dLogit * scale * float64(key))
					result[keyIndex] += float32(dLogit * scale * float64(query))
				}
				gradient.Temperature[layer*cfg.Heads+head] += float32(-scale * dLogit * queryKeyDot)
			}
		}
	}
	return result
}

func madForwardRows(input []float32, rows, width int, epsilon float64) ([]float32, madState) {
	normalizedWeights, rawWeights := rankWeights(width)
	state := madState{
		output: make([]float32, len(input)), a: make([]float32, len(input)), b: make([]float32, len(input)),
		c: make([]float64, rows), inverse: make([]float64, rows),
	}
	for row := range rows {
		start := row * width
		out, a, b, c, inverse := madForwardRow(input[start:start+width], normalizedWeights, rawWeights, epsilon)
		copy(state.output[start:], out)
		copy(state.a[start:], a)
		copy(state.b[start:], b)
		state.c[row], state.inverse[row] = c, inverse
	}
	return state.output, state
}

func madForwardRow(input, normalizedWeights, rawWeights []float32, epsilon float64) ([]float32, []float32, []float32, float64, float64) {
	type ranked struct {
		value float32
		index int
	}
	rankedValues := make([]ranked, len(input))
	for index, value := range input {
		rankedValues[index] = ranked{value, index}
	}
	for left := 1; left < len(rankedValues); left++ {
		for right := left; right > 0 && rankedValues[right].value < rankedValues[right-1].value; right-- {
			rankedValues[right], rankedValues[right-1] = rankedValues[right-1], rankedValues[right]
		}
	}
	var location float64
	for index, value := range rankedValues {
		location += float64(value.value) * float64(normalizedWeights[index])
	}
	split := 0
	for split < len(input) && float64(rankedValues[split].value) < location {
		split++
	}
	type distance struct {
		value float64
		index int
	}
	distances := make([]distance, len(input))
	left, right := split-1, split
	for index := range distances {
		switch {
		case left < 0:
			distances[index] = distance{float64(rankedValues[right].value) - location, right}
			right++
		case right >= len(input):
			distances[index] = distance{location - float64(rankedValues[left].value), left}
			left--
		case location-float64(rankedValues[left].value) <= float64(rankedValues[right].value)-location:
			distances[index] = distance{location - float64(rankedValues[left].value), left}
			left--
		default:
			distances[index] = distance{float64(rankedValues[right].value) - location, right}
			right++
		}
	}
	twoOverSize := 2 / float64(len(input))
	var scale float64
	for index, distance := range distances {
		scale += distance.value * float64(rawWeights[index])
	}
	inverse := 1 / (scale*twoOverSize + epsilon)
	output := make([]float32, len(input))
	for index := range output {
		output[index] = float32((float64(input[index]) - location) * inverse)
	}
	distanceRank := make([]int, len(input))
	for index, distance := range distances {
		distanceRank[distance.index] = index
	}
	bc := make([]float64, len(input))
	var c float64
	for index := range bc {
		boundary := -1.0
		if float64(rankedValues[index].value) >= location {
			boundary = 1
		}
		bc[index] = float64(rawWeights[distanceRank[index]]) * twoOverSize * boundary
		c += bc[index]
	}
	originalRank := make([]int, len(input))
	for index, value := range rankedValues {
		originalRank[value.index] = index
	}
	a, b := make([]float32, len(input)), make([]float32, len(input))
	for index := range input {
		a[index] = normalizedWeights[originalRank[index]]
		b[index] = float32(bc[originalRank[index]])
	}
	return output, a, b, c, inverse
}

func madBackwardRows(incoming []float32, state madState, rows, width int) []float32 {
	result := make([]float32, len(incoming))
	for row := range rows {
		start := row * width
		var sum, weighted float64
		for index := range width {
			sum += float64(incoming[start+index])
			weighted += float64(incoming[start+index]) * float64(state.output[start+index])
		}
		t1 := state.inverse[row] * (sum + state.c[row]*weighted)
		t2 := state.inverse[row] * weighted
		for index := range width {
			result[start+index] = float32(state.inverse[row]*float64(incoming[start+index]) - float64(state.a[start+index])*t1 - float64(state.b[start+index])*t2)
		}
	}
	return result
}

func linearForward(input, weights []float32, rows, in, out int) []float32 {
	result := make([]float32, rows*out)
	for row := range rows {
		for output := range out {
			var sum float64
			for column := range in {
				sum += float64(input[row*in+column]) * float64(weights[output*in+column])
			}
			result[row*out+output] = float32(sum)
		}
	}
	return result
}

func linearBackward(dInput, dWeights, input, weights, incoming []float32, rows, in, out int) {
	for row := range rows {
		for output := range out {
			gradient := incoming[row*out+output]
			for column := range in {
				dInput[row*in+column] += weights[output*in+column] * gradient
				dWeights[output*in+column] += input[row*in+column] * gradient
			}
		}
	}
}

func rankWeights(size int) ([]float32, []float32) {
	normalized, raw := make([]float32, size), make([]float32, size)
	for index := range size {
		raw[index] = float32(2*(float64(index)+0.5)/float64(size) - 1)
		normalized[index] = raw[index] / float32(size)
	}
	return normalized, raw
}

func addCopy(left, right []float32) []float32 {
	result := make([]float32, len(left))
	for index := range result {
		result[index] = left[index] + right[index]
	}
	return result
}

func addInPlace(left, right []float32) {
	for index := range left {
		left[index] += right[index]
	}
}

func validateMADTransformer(model MADTransformer) error {
	cfg := model.Config
	if cfg.Vocab <= 1 || cfg.Block <= 1 || cfg.Hidden <= 0 || cfg.Heads <= 0 || cfg.HeadDim <= 0 ||
		cfg.Heads*cfg.HeadDim != cfg.Hidden || cfg.Layers <= 0 || cfg.Intermediate <= 0 || cfg.Window <= 0 ||
		cfg.Window > cfg.Block || cfg.Epsilon <= 0 || math.IsNaN(cfg.Epsilon) || math.IsInf(cfg.Epsilon, 0) ||
		len(model.Token) != cfg.Vocab*cfg.Hidden || len(model.Head) != cfg.Vocab*cfg.Hidden ||
		len(model.Position) != cfg.Block*cfg.Hidden || len(model.PositionBias) != cfg.Block ||
		len(model.Temperature) != cfg.Layers*cfg.Heads || len(model.Layer) != cfg.Layers {
		return errors.New("mad transformer: invalid model geometry")
	}
	for _, layer := range model.Layer {
		if len(layer.QKV) != 3*cfg.Hidden*cfg.Hidden || len(layer.Output) != cfg.Hidden*cfg.Hidden ||
			len(layer.Expand) != cfg.Intermediate*cfg.Hidden || len(layer.Contract) != cfg.Hidden*cfg.Intermediate {
			return errors.New("mad transformer: invalid layer geometry")
		}
	}
	return nil
}

func clearMADTransformer(model MADTransformer) {
	clear(model.Token)
	clear(model.Head)
	clear(model.Position)
	clear(model.PositionBias)
	clear(model.Temperature)
	for _, layer := range model.Layer {
		clear(layer.QKV)
		clear(layer.Output)
		clear(layer.Expand)
		clear(layer.Contract)
	}
}
