package reference

import (
	"errors"
	"math"
	"slices"
	"sort"

	"overgo/internal/tensor"
)

func deepSeek4HCInit(shape tensor.Shape, inputs []Value, attributes tensor.DeepSeek4HCAttributes) (Value, error) {
	if len(inputs) != 1 || attributes.HyperConnections == 0 {
		return Value{}, errors.New("invalid DeepSeek 4 HC init")
	}
	hidden := int(inputs[0].Shape.Dims[0])
	tokens := int(inputs[0].Shape.Dims[1])
	hc := int(attributes.HyperConnections)
	output := make([]float32, hidden*hc*tokens)
	for token := range tokens {
		row := inputs[0].Data[token*hidden : (token+1)*hidden]
		for stream := range hc {
			copy(output[(token*hc+stream)*hidden:], row)
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func deepSeek4HCMixes(input, fn Value, hc int, epsilon float32) ([]float64, error) {
	hidden := int(input.Shape.Dims[0])
	tokens := int(input.Shape.Dims[2])
	hcDim := hidden * hc
	mixDim := int(fn.Shape.Dims[1])
	if hc <= 0 || int(input.Shape.Dims[1]) != hc || int(fn.Shape.Dims[0]) != hcDim {
		return nil, errors.New("invalid DeepSeek 4 HC mix dimensions")
	}
	mixes := make([]float64, mixDim*tokens)
	for token := range tokens {
		flat := input.Data[token*hcDim : (token+1)*hcDim]
		var meanSquare float64
		for _, value := range flat {
			meanSquare += float64(value) * float64(value)
		}
		inverseRMS := 1 / math.Sqrt(meanSquare/float64(hcDim)+float64(epsilon))
		for mix := range mixDim {
			var sum float64
			base := mix * hcDim
			for channel, value := range flat {
				sum += float64(value) * inverseRMS * float64(fn.Data[base+channel])
			}
			mixes[token*mixDim+mix] = sum
		}
	}
	return mixes, nil
}

func deepSeek4HCPre(shape tensor.Shape, inputs []Value, attributes tensor.DeepSeek4HCAttributes) (Value, error) {
	if len(inputs) != 4 {
		return Value{}, errors.New("invalid DeepSeek 4 HC pre inputs")
	}
	input, fn, scale, base := inputs[0], inputs[1], inputs[2], inputs[3]
	hidden := int(input.Shape.Dims[0])
	hc := int(attributes.HyperConnections)
	tokens := int(input.Shape.Dims[2])
	mixes, err := deepSeek4HCMixes(input, fn, hc, attributes.NormEpsilon)
	if err != nil {
		return Value{}, err
	}
	mixDim := int(fn.Shape.Dims[1])
	output := make([]float32, hidden*tokens)
	for token := range tokens {
		for stream := range hc {
			weight := 1/(1+math.Exp(-(mixes[token*mixDim+stream]*float64(scale.Data[0])+float64(base.Data[stream])))) + float64(attributes.Epsilon)
			inputBase := (token*hc + stream) * hidden
			for channel := range hidden {
				output[token*hidden+channel] += float32(float64(input.Data[inputBase+channel]) * weight)
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func deepSeek4HCPost(shape tensor.Shape, inputs []Value, attributes tensor.DeepSeek4HCAttributes) (Value, error) {
	if len(inputs) != 5 {
		return Value{}, errors.New("invalid DeepSeek 4 HC post inputs")
	}
	branch, residual, fn, scale, base := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
	hidden := int(residual.Shape.Dims[0])
	hc := int(attributes.HyperConnections)
	tokens := int(residual.Shape.Dims[2])
	mixes, err := deepSeek4HCMixes(residual, fn, hc, attributes.NormEpsilon)
	if err != nil {
		return Value{}, err
	}
	mixDim := int(fn.Shape.Dims[1])
	output := make([]float32, hidden*hc*tokens)
	comb := make([]float64, hc*hc)
	for token := range tokens {
		mixBase := token * mixDim
		for src := range hc {
			maximum := math.Inf(-1)
			for dst := range hc {
				index := dst + hc*src
				value := mixes[mixBase+2*hc+index]*float64(scale.Data[2]) + float64(base.Data[2*hc+index])
				comb[index] = value
				maximum = math.Max(maximum, value)
			}
			var sum float64
			for dst := range hc {
				index := dst + hc*src
				comb[index] = math.Exp(comb[index] - maximum)
				sum += comb[index]
			}
			for dst := range hc {
				index := dst + hc*src
				comb[index] = comb[index]/sum + float64(attributes.Epsilon)
			}
		}
		normalizeDeepSeek4HCColumns(comb, hc, attributes.Epsilon)
		for iteration := uint32(1); iteration < attributes.SinkhornIterations; iteration++ {
			for src := range hc {
				var sum float64
				for dst := range hc {
					sum += comb[dst+hc*src]
				}
				sum += float64(attributes.Epsilon)
				for dst := range hc {
					comb[dst+hc*src] /= sum
				}
			}
			normalizeDeepSeek4HCColumns(comb, hc, attributes.Epsilon)
		}
		for dst := range hc {
			post := 2 / (1 + math.Exp(-(mixes[mixBase+hc+dst]*float64(scale.Data[1]) + float64(base.Data[hc+dst]))))
			for channel := range hidden {
				value := float64(branch.Data[token*hidden+channel]) * post
				for src := range hc {
					value += float64(residual.Data[(token*hc+src)*hidden+channel]) * comb[dst+hc*src]
				}
				output[(token*hc+dst)*hidden+channel] = float32(value)
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func normalizeDeepSeek4HCColumns(comb []float64, hc int, epsilon float32) {
	for dst := range hc {
		var sum float64
		for src := range hc {
			sum += comb[dst+hc*src]
		}
		sum += float64(epsilon)
		for src := range hc {
			comb[dst+hc*src] /= sum
		}
	}
}

func deepSeek4HCHead(shape tensor.Shape, inputs []Value, attributes tensor.DeepSeek4HCAttributes) (Value, error) {
	if len(inputs) != 4 {
		return Value{}, errors.New("invalid DeepSeek 4 HC head inputs")
	}
	input, fn, scale, base := inputs[0], inputs[1], inputs[2], inputs[3]
	hidden := int(input.Shape.Dims[0])
	hc := int(attributes.HyperConnections)
	tokens := int(input.Shape.Dims[2])
	mixes, err := deepSeek4HCMixes(input, fn, hc, attributes.NormEpsilon)
	if err != nil {
		return Value{}, err
	}
	output := make([]float32, hidden*tokens)
	for token := range tokens {
		for stream := range hc {
			weight := 1/(1+math.Exp(-(mixes[token*hc+stream]*float64(scale.Data[0])+float64(base.Data[stream])))) + float64(attributes.Epsilon)
			for channel := range hidden {
				output[token*hidden+channel] += float32(float64(input.Data[(token*hc+stream)*hidden+channel]) * weight)
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

type deepSeek4CompressedBlock struct {
	position uint32
	value    []float32
}

func deepSeek4CompressedBlocks(
	kv, score, norm Value,
	ratio, width uint32,
	positions []uint32,
	attributes tensor.DeepSeek4AttentionAttributes,
) ([]deepSeek4CompressedBlock, error) {
	tokens := uint32(kv.Shape.Dims[2])
	overlap := ratio == 4
	coefficient := uint32(1)
	if overlap {
		coefficient = 2
	}
	if uint32(kv.Shape.Dims[0]) != coefficient*width || !score.Shape.Equal(kv.Shape) || uint32(norm.Shape.Dims[0]) != width {
		return nil, errors.New("invalid DeepSeek 4 compressor state")
	}
	if len(positions) != int(tokens) {
		return nil, errors.New("invalid DeepSeek 4 compressor positions")
	}
	rows := make(map[uint32]uint32, len(positions))
	for row, position := range positions {
		rows[position] = uint32(row)
	}
	blocks := make([]deepSeek4CompressedBlock, 0, tokens/ratio)
	for _, start := range positions {
		if start%ratio != 0 {
			continue
		}
		complete := true
		for offset := uint32(0); offset < ratio; offset++ {
			if _, present := rows[start+offset]; !present {
				complete = false
				break
			}
		}
		if !complete {
			continue
		}
		result := make([]float32, width)
		for channel := uint32(0); channel < width; channel++ {
			maximum := math.Inf(-1)
			type candidate struct{ value, score float64 }
			candidates := make([]candidate, 0, coefficient*ratio)
			if overlap {
				previous := int64(start) - int64(ratio)
				for offset := uint32(0); offset < ratio; offset++ {
					position := previous + int64(offset)
					item := candidate{value: 0, score: math.Inf(-1)}
					if row, present := rows[uint32(position)]; position >= 0 && present {
						item.value = float64(kv.Data[int(row*2*width+channel)])
						item.score = float64(score.Data[int(row*2*width+channel)])
					}
					maximum = math.Max(maximum, item.score)
					candidates = append(candidates, item)
				}
			}
			for offset := uint32(0); offset < ratio; offset++ {
				row := rows[start+offset]
				column := channel
				if overlap {
					column += width
				}
				item := candidate{
					value: float64(kv.Data[int(row*coefficient*width+column)]),
					score: float64(score.Data[int(row*coefficient*width+column)]),
				}
				maximum = math.Max(maximum, item.score)
				candidates = append(candidates, item)
			}
			var numerator, denominator float64
			for _, item := range candidates {
				weight := math.Exp(item.score - maximum)
				numerator += item.value * weight
				denominator += weight
			}
			result[channel] = float32(numerator / denominator)
		}
		var meanSquare float64
		for _, value := range result {
			meanSquare += float64(value) * float64(value)
		}
		inverseRMS := 1 / math.Sqrt(meanSquare/float64(width)+float64(attributes.NormEpsilon))
		for channel := range result {
			result[channel] = float32(float64(result[channel]) * inverseRMS * float64(norm.Data[channel]))
		}
		deepSeek4RotateTail(result, start, false, attributes)
		blocks = append(blocks, deepSeek4CompressedBlock{position: start, value: result})
	}
	return blocks, nil
}

func deepSeek4RotateTail(vector []float32, position uint32, inverse bool, attributes tensor.DeepSeek4AttentionAttributes) {
	rotary := int(attributes.RotaryDimensions)
	start := len(vector) - rotary
	rope := tensor.RoPEAttributes{
		RotaryDimensions: attributes.RotaryDimensions, FrequencyBase: attributes.FrequencyBase,
		FrequencyScale: attributes.FrequencyScale, OriginalContext: attributes.OriginalContext,
		ExtFactor: attributes.ExtFactor, AttentionFactor: attributes.AttentionFactor,
		BetaFast: attributes.BetaFast, BetaSlow: attributes.BetaSlow,
	}
	for pair := 0; pair < rotary/2; pair++ {
		cosine, sine := ropeCosSin(rope, pair, rotary, position, 1)
		if inverse {
			sine = -sine
		}
		index := start + 2*pair
		first, second := vector[index], vector[index+1]
		vector[index] = first*cosine - second*sine
		vector[index+1] = first*sine + second*cosine
	}
}

func deepSeek4FWHT(vector []float32) {
	for stride := 1; stride < len(vector); stride *= 2 {
		for base := 0; base < len(vector); base += 2 * stride {
			for offset := 0; offset < stride; offset++ {
				first, second := base+offset, base+offset+stride
				a, b := vector[first], vector[second]
				vector[first], vector[second] = a+b, a-b
			}
		}
	}
	scale := float32(1 / math.Sqrt(float64(len(vector))))
	for index := range vector {
		vector[index] *= scale
	}
}

func deepSeek4Attention(shape tensor.Shape, inputs []Value, attributes tensor.DeepSeek4AttentionAttributes) (Value, error) {
	if len(inputs) < 4 || len(attributes.Positions) == 0 {
		return Value{}, errors.New("invalid DeepSeek 4 attention inputs")
	}
	query, cache, positionState, sinks := inputs[0], inputs[1], inputs[2], inputs[3]
	width := uint32(query.Shape.Dims[0])
	heads := uint32(query.Shape.Dims[1])
	newTokens := uint32(query.Shape.Dims[2])
	totalTokens := uint32(cache.Shape.Dims[2])
	if totalTokens < newTokens {
		return Value{}, errors.New("DeepSeek 4 attention cache is shorter than query")
	}
	if len(positionState.Data) != int(totalTokens) {
		return Value{}, errors.New("invalid DeepSeek 4 cache position shape")
	}
	cachePositions := make([]uint32, totalTokens)
	for index, value := range positionState.Data {
		position := uint32(value)
		if value < 0 || float32(position) != value ||
			(index > 0 && position <= cachePositions[index-1]) {
			return Value{}, errors.New("invalid DeepSeek 4 cache positions")
		}
		cachePositions[index] = position
	}
	pastTokens := totalTokens - newTokens
	for index, position := range attributes.Positions {
		if cachePositions[int(pastTokens)+index] != position {
			return Value{}, errors.New("DeepSeek 4 appended positions differ from cache state")
		}
	}
	var compressed, indexerCompressed []deepSeek4CompressedBlock
	var err error
	if attributes.Ratio.Enabled() {
		compressed, err = deepSeek4CompressedBlocks(
			inputs[4], inputs[5], inputs[6], uint32(attributes.Ratio), width, cachePositions, attributes,
		)
		if err != nil {
			return Value{}, err
		}
	}
	if attributes.Ratio.UsesIndexer() {
		indexerWidth := uint32(inputs[7].Shape.Dims[0])
		indexerCompressed, err = deepSeek4CompressedBlocks(
			inputs[9], inputs[10], inputs[11],
			uint32(tensor.DeepSeek4CompressionOverlap), indexerWidth, cachePositions, attributes,
		)
		if err != nil {
			return Value{}, err
		}
		for index := range indexerCompressed {
			deepSeek4FWHT(indexerCompressed[index].value)
		}
	}
	output := make([]float32, int(width*heads*newTokens))
	scale := 1 / math.Sqrt(float64(width))
	for token := uint32(0); token < newTokens; token++ {
		position := attributes.Positions[token]
		visibleCompressed := make([]int, 0, len(compressed))
		for index, block := range compressed {
			if block.position+uint32(attributes.Ratio) <= position+1 {
				visibleCompressed = append(visibleCompressed, index)
			}
		}
		if attributes.Ratio.UsesIndexer() && len(visibleCompressed) > 0 {
			type scored struct {
				index int
				score float64
			}
			scores := make([]scored, len(visibleCompressed))
			indexerWidth := uint32(inputs[7].Shape.Dims[0])
			for item, blockIndex := range visibleCompressed {
				var score float64
				for head := uint32(0); head < attributes.IndexerHeads; head++ {
					queryVector := slices.Clone(inputs[7].Data[int((token*attributes.IndexerHeads+head)*indexerWidth):int((token*attributes.IndexerHeads+head+1)*indexerWidth)])
					deepSeek4RotateTail(queryVector, position, false, attributes)
					deepSeek4FWHT(queryVector)
					var dot float64
					for channel := uint32(0); channel < indexerWidth; channel++ {
						dot += float64(queryVector[channel]) * float64(indexerCompressed[blockIndex].value[channel])
					}
					if dot > 0 {
						score += dot * float64(inputs[8].Data[token*attributes.IndexerHeads+head])
					}
				}
				scores[item] = scored{index: blockIndex, score: score}
			}
			sort.SliceStable(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
			limit := int(attributes.IndexerTopK)
			if limit > len(scores) {
				limit = len(scores)
			}
			visibleCompressed = visibleCompressed[:limit]
			for index := range limit {
				visibleCompressed[index] = scores[index].index
			}
		}
		rawFirst := uint32(0)
		if position+1 > attributes.Window {
			rawFirst = max(rawFirst, position+1-attributes.Window)
		}
		rawLast := position + 1
		for head := uint32(0); head < heads; head++ {
			queryBase := int((token*heads + head) * width)
			queryVector := slices.Clone(query.Data[queryBase : queryBase+int(width)])
			deepSeek4RotateTail(queryVector, position, false, attributes)
			maximum := float64(sinks.Data[head])
			type candidate struct {
				value []float32
				logit float64
			}
			candidates := make([]candidate, 0, int(rawLast-rawFirst)+len(visibleCompressed))
			add := func(vector []float32) {
				var dot float64
				for channel := uint32(0); channel < width; channel++ {
					dot += float64(queryVector[channel]) * float64(vector[channel])
				}
				logit := dot * scale
				maximum = math.Max(maximum, logit)
				candidates = append(candidates, candidate{value: vector, logit: logit})
			}
			for row, rawPosition := range cachePositions {
				if rawPosition < rawFirst || rawPosition >= rawLast {
					continue
				}
				base := row * int(width)
				raw := slices.Clone(cache.Data[base : base+int(width)])
				deepSeek4RotateTail(raw, rawPosition, false, attributes)
				add(raw)
			}
			for _, blockIndex := range visibleCompressed {
				add(compressed[blockIndex].value)
			}
			denominator := math.Exp(float64(sinks.Data[head]) - maximum)
			for _, candidate := range candidates {
				denominator += math.Exp(candidate.logit - maximum)
			}
			result := output[queryBase : queryBase+int(width)]
			for _, candidate := range candidates {
				weight := math.Exp(candidate.logit-maximum) / denominator
				for channel := uint32(0); channel < width; channel++ {
					result[channel] += float32(weight * float64(candidate.value[channel]))
				}
			}
			deepSeek4RotateTail(result, position, true, attributes)
		}
	}
	return Value{Shape: shape, Data: output}, nil
}
