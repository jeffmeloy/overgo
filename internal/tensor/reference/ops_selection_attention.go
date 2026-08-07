package reference

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"llamacpp2go/internal/tensor"
)

func fwht(shape tensor.Shape, input Value) (Value, error) {
	width := int(shape.Dims[0])
	if width == 0 || width&(width-1) != 0 || len(input.Data)%width != 0 {
		return Value{}, errors.New("invalid FWHT dimensions")
	}
	output := slices.Clone(input.Data)
	for row := 0; row < len(output); row += width {
		for stride := 1; stride < width; stride *= 2 {
			for base := 0; base < width; base += 2 * stride {
				for offset := 0; offset < stride; offset++ {
					first := row + base + offset
					second := first + stride
					a, b := output[first], output[second]
					output[first], output[second] = a+b, a-b
				}
			}
		}
	}
	scale := float32(1 / math.Sqrt(float64(width)))
	for index := range output {
		output[index] *= scale
	}
	return Value{Shape: shape, Data: output}, nil
}

func topK(
	shape tensor.Shape,
	input Value,
	attributes tensor.TopKAttributes,
) (Value, error) {
	width := int(input.Shape.Dims[0])
	k := int(attributes.K)
	if width == 0 || k <= 0 || k > width || len(input.Data)%width != 0 {
		return Value{}, errors.New("invalid TopK dimensions")
	}
	rows := len(input.Data) / width
	output := make([]float32, rows*k)
	for row := range rows {
		selected := output[row*k : (row+1)*k]
		for index := range selected {
			selected[index] = -1
		}
		for candidate := range width {
			candidateValue := input.Data[row*width+candidate]
			insert := k
			for slot := range k {
				selectedIndex := int(selected[slot])
				if selectedIndex < 0 || topKBefore(
					candidateValue, candidate,
					input.Data[row*width+selectedIndex], selectedIndex,
				) {
					insert = slot
					break
				}
			}
			if insert == k {
				continue
			}
			copy(selected[insert+1:], selected[insert:k-1])
			selected[insert] = float32(candidate)
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func topKBefore(left float32, leftIndex int, right float32, rightIndex int) bool {
	leftNaN, rightNaN := math.IsNaN(float64(left)), math.IsNaN(float64(right))
	if leftNaN != rightNaN {
		return !leftNaN
	}
	if left == right || leftNaN {
		return leftIndex < rightIndex
	}
	return left > right
}

func gatherLast(shape tensor.Shape, input, indices Value) (Value, error) {
	last := int(input.Shape.Rank) - 1
	rows := int(input.Shape.Dims[last])
	inner := 1
	for _, dimension := range input.Shape.Slice()[:last] {
		inner *= int(dimension)
	}
	if rows <= 0 || inner <= 0 || len(input.Data) != rows*inner {
		return Value{}, errors.New("invalid GatherLast input dimensions")
	}
	output := make([]float32, inner*len(indices.Data))
	for indexPosition, raw := range indices.Data {
		row, err := exactTensorIndex(raw, rows)
		if err != nil {
			return Value{}, fmt.Errorf("GatherLast index %d: %w", indexPosition, err)
		}
		copy(output[indexPosition*inner:(indexPosition+1)*inner], input.Data[row*inner:(row+1)*inner])
	}
	return Value{Shape: shape, Data: output}, nil
}

func sparseAttention(
	shape tensor.Shape,
	inputs []Value,
	attributes tensor.SparseAttentionAttributes,
) (Value, error) {
	query, key, value, indices := inputs[0], inputs[1], inputs[2], inputs[3]
	keyWidth := int(query.Shape.Dims[0])
	valueWidth := int(value.Shape.Dims[0])
	queryHeads := int(query.Shape.Dims[1])
	keyValueHeads := int(key.Shape.Dims[1])
	queryTokens := int(query.Shape.Dims[2])
	keyValueTokens := int(key.Shape.Dims[2])
	selected := int(indices.Shape.Dims[0])
	if keyWidth <= 0 || valueWidth <= 0 || queryHeads <= 0 || keyValueHeads <= 0 ||
		queryTokens <= 0 || keyValueTokens <= 0 || selected <= 0 || queryHeads%keyValueHeads != 0 {
		return Value{}, errors.New("invalid SparseAttention dimensions")
	}
	indexRows := make([]int, len(indices.Data))
	for index, raw := range indices.Data {
		row, err := exactTensorIndex(raw, keyValueTokens)
		if err != nil {
			return Value{}, fmt.Errorf("SparseAttention index %d: %w", index, err)
		}
		indexRows[index] = row
	}
	output := make([]float32, valueWidth*queryHeads*queryTokens)
	groupSize := queryHeads / keyValueHeads
	for queryToken := range queryTokens {
		selectedTokens := make([]int, 0, selected)
		for slot := range selected {
			keyToken := indexRows[queryToken*selected+slot]
			if attributes.Causal && keyToken > int(attributes.QueryStart)+queryToken {
				continue
			}
			selectedTokens = append(selectedTokens, keyToken)
		}
		if len(selectedTokens) == 0 {
			return Value{}, fmt.Errorf("SparseAttention query token %d has no valid keys", queryToken)
		}
		scores := make([]float64, len(selectedTokens))
		for queryHead := range queryHeads {
			keyValueHead := queryHead / groupSize
			queryOffset := (queryToken*queryHeads + queryHead) * keyWidth
			maximum := math.Inf(-1)
			for slot, keyToken := range selectedTokens {
				keyOffset := (keyToken*keyValueHeads + keyValueHead) * keyWidth
				var dot float64
				for channel := range keyWidth {
					dot += float64(query.Data[queryOffset+channel]) * float64(key.Data[keyOffset+channel])
				}
				scores[slot] = dot * float64(attributes.Scale)
				maximum = max(maximum, scores[slot])
			}
			var sum float64
			for slot := range selectedTokens {
				scores[slot] = math.Exp(scores[slot] - maximum)
				sum += scores[slot]
			}
			outputOffset := (queryToken*queryHeads + queryHead) * valueWidth
			for channel := range valueWidth {
				var weighted float64
				for slot, keyToken := range selectedTokens {
					valueOffset := (keyToken*keyValueHeads + keyValueHead) * valueWidth
					weighted += scores[slot] * float64(value.Data[valueOffset+channel])
				}
				output[outputOffset+channel] = float32(weighted / sum)
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func exactTensorIndex(value float32, limit int) (int, error) {
	if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || value < 0 || value >= float32(limit) {
		return 0, errors.New("index is outside tensor")
	}
	index := int(value)
	if float32(index) != value {
		return 0, errors.New("index is not an exact integer")
	}
	return index, nil
}

func indexerScore(
	shape tensor.Shape,
	inputs []Value,
	attributes tensor.IndexerScoreAttributes,
) (Value, error) {
	query, key, weights := inputs[0], inputs[1], inputs[2]
	width := int(query.Shape.Dims[0])
	heads := int(query.Shape.Dims[1])
	queryTokens := int(query.Shape.Dims[2])
	keyTokens := int(key.Shape.Dims[2])
	if width <= 0 || heads <= 0 || queryTokens <= 0 || keyTokens <= 0 {
		return Value{}, errors.New("invalid IndexerScore dimensions")
	}
	output := make([]float32, keyTokens*queryTokens)
	for queryToken := range queryTokens {
		for keyToken := range keyTokens {
			if keyToken > int(attributes.QueryStart)+queryToken {
				output[queryToken*keyTokens+keyToken] = float32(math.Inf(-1))
				continue
			}
			var score float64
			for head := range heads {
				queryOffset := (queryToken*heads + head) * width
				keyOffset := keyToken * width
				var dot float64
				for channel := range width {
					dot += float64(query.Data[queryOffset+channel]) * float64(key.Data[keyOffset+channel])
				}
				if dot > 0 {
					score += dot * float64(weights.Data[queryToken*heads+head])
				}
			}
			output[queryToken*keyTokens+keyToken] = float32(score * float64(attributes.Scale))
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func attention(
	shape tensor.Shape,
	query, key, value Value,
	bias, sinks, blockIDs *Value,
	attributes tensor.AttentionAttributes,
) (Value, error) {
	keyWidth := int(query.Shape.Dims[0])
	valueWidth := int(value.Shape.Dims[0])
	queryHeads := int(query.Shape.Dims[1])
	keyValueHeads := int(key.Shape.Dims[1])
	queryTokens := int(query.Shape.Dims[2])
	keyValueTokens := int(key.Shape.Dims[2])
	sequences := 1
	if query.Shape.Rank == 4 {
		sequences = int(query.Shape.Dims[3])
	}
	if keyWidth <= 0 || valueWidth <= 0 || queryHeads <= 0 ||
		keyValueHeads <= 0 || queryTokens <= 0 || keyValueTokens <= 0 {
		return Value{}, errors.New("invalid attention dimensions")
	}
	if queryHeads%keyValueHeads != 0 {
		return Value{}, errors.New("attention query heads are not divisible by KV heads")
	}
	if attributes.Causal && uint64(attributes.QueryStart)+uint64(queryTokens) > uint64(keyValueTokens) {
		return Value{}, errors.New("attention query range exceeds KV tokens")
	}
	if blockIDs != nil && len(blockIDs.Data) != keyValueTokens {
		return Value{}, errors.New("attention block ID count differs from KV tokens")
	}
	output := make([]float32, valueWidth*queryHeads*queryTokens*sequences)
	groupSize := queryHeads / keyValueHeads
	nHeadLog2 := 1
	for nHeadLog2*2 <= queryHeads {
		nHeadLog2 *= 2
	}
	m0 := math.Pow(2, -float64(attributes.MaxALiBiBias)/float64(nHeadLog2))
	m1 := math.Pow(2, -float64(attributes.MaxALiBiBias/2)/float64(nHeadLog2))
	scores := make([]float64, keyValueTokens)
	for sequence := 0; sequence < sequences; sequence++ {
		for queryToken := 0; queryToken < queryTokens; queryToken++ {
			queryPosition := int(attributes.QueryStart) + queryToken
			causalLimit := queryPosition + 1
			keyLimit := keyValueTokens
			if attributes.Causal {
				keyLimit = causalLimit
				if blockIDs != nil && blockIDs.Data[queryPosition] >= 0 {
					keyLimit = keyValueTokens
				}
			}
			keyFirst := 0
			if attributes.ChunkedWindow {
				queryPosition := int(attributes.QueryStart) + queryToken
				keyFirst = queryPosition / int(attributes.Window) * int(attributes.Window)
			} else if attributes.SymmetricWindow {
				halfWindow := int(attributes.Window / 2)
				queryPosition := int(attributes.QueryStart) + queryToken
				keyFirst = max(0, queryPosition-halfWindow)
				keyLimit = min(keyValueTokens, queryPosition+halfWindow+1)
			} else if attributes.Window > 0 && causalLimit > int(attributes.Window) {
				keyFirst = causalLimit - int(attributes.Window)
			}
			for queryHead := 0; queryHead < queryHeads; queryHead++ {
				keyValueHead := queryHead / groupSize
				alibiSlope := 0.0
				if attributes.MaxALiBiBias > 0 {
					if queryHead < nHeadLog2 {
						alibiSlope = math.Pow(m0, float64(queryHead+1))
					} else {
						alibiSlope = math.Pow(m1, float64(2*(queryHead-nHeadLog2)+1))
					}
				}
				maximum := math.Inf(-1)
				if sinks != nil {
					maximum = float64(sinks.Data[queryHead])
				}
				queryOffset := ((sequence*queryTokens+queryToken)*queryHeads + queryHead) * keyWidth
				for keyToken := keyFirst; keyToken < keyLimit; keyToken++ {
					if attributes.Causal && keyToken >= causalLimit && !sameAttentionBlock(blockIDs, queryPosition, keyToken) {
						continue
					}
					keyOffset := ((sequence*keyValueTokens+keyToken)*keyValueHeads + keyValueHead) * keyWidth
					var dot float64
					for channel := 0; channel < keyWidth; channel++ {
						dot += float64(query.Data[queryOffset+channel]) * float64(key.Data[keyOffset+channel])
					}
					score := dot * float64(attributes.Scale)
					if alibiSlope != 0 {
						queryPosition := int(attributes.QueryStart) + queryToken
						score -= math.Abs(float64(queryPosition-keyToken)) * alibiSlope
					}
					if bias != nil {
						bucket := relativePositionBucket(
							int(attributes.QueryStart)+queryToken,
							keyToken,
							int(attributes.RelativeBuckets),
							attributes.RelativeBidirectional,
						)
						score += float64(bias.Data[bucket*queryHeads+queryHead])
					}
					if attributes.Softcap > 0 {
						cap := float64(attributes.Softcap)
						score = cap * math.Tanh(score/cap)
					}
					scores[keyToken] = score
					if score > maximum {
						maximum = score
					}
				}
				var sum float64
				if sinks != nil {
					sum = math.Exp(float64(sinks.Data[queryHead]) - maximum)
				}
				for keyToken := keyFirst; keyToken < keyLimit; keyToken++ {
					if attributes.Causal && keyToken >= causalLimit && !sameAttentionBlock(blockIDs, queryPosition, keyToken) {
						continue
					}
					probability := math.Exp(scores[keyToken] - maximum)
					scores[keyToken] = probability
					sum += probability
				}
				outputOffset := ((sequence*queryTokens+queryToken)*queryHeads + queryHead) * valueWidth
				for channel := 0; channel < valueWidth; channel++ {
					var weighted float64
					for keyToken := keyFirst; keyToken < keyLimit; keyToken++ {
						if attributes.Causal && keyToken >= causalLimit && !sameAttentionBlock(blockIDs, queryPosition, keyToken) {
							continue
						}
						valueOffset := ((sequence*keyValueTokens+keyToken)*keyValueHeads + keyValueHead) * valueWidth
						weighted += scores[keyToken] * float64(value.Data[valueOffset+channel])
					}
					output[outputOffset+channel] = float32(weighted / sum)
				}
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func sameAttentionBlock(blockIDs *Value, query, key int) bool {
	return blockIDs != nil && blockIDs.Data[query] >= 0 && blockIDs.Data[key] == blockIDs.Data[query]
}

func relativePositionBucket(query, key, buckets int, bidirectional bool) int {
	relative := key - query
	bucket := 0
	if bidirectional {
		half := buckets / 2
		if relative > 0 {
			bucket += half
		}
		buckets = half
	} else if relative > 0 {
		relative = 0
	}
	if relative < 0 {
		relative = -relative
	}
	maxExact := buckets / 2
	if relative < maxExact {
		return bucket + relative
	}
	large := maxExact + int(math.Floor(
		math.Log(float64(relative)/float64(maxExact))*
			float64(buckets-maxExact)/math.Log(128.0/float64(maxExact)),
	))
	if large >= buckets {
		large = buckets - 1
	}
	return bucket + large
}

func concat(shape tensor.Shape, left, right Value, axis uint32) (Value, error) {
	leftElements, err := left.Shape.Elements()
	if err != nil {
		return Value{}, err
	}
	rightElements, err := right.Shape.Elements()
	if err != nil {
		return Value{}, err
	}
	if leftElements > uint64(len(left.Data)) || rightElements > uint64(len(right.Data)) {
		return Value{}, errors.New("invalid concat input storage")
	}
	outputElements, shapeErr := shape.Elements()
	if shapeErr != nil || outputElements > uint64(math.MaxInt) {
		return Value{}, errors.New("invalid concat output size")
	}
	inner := 1
	for dimension := uint32(0); dimension < axis; dimension++ {
		inner *= int(left.Shape.Dims[dimension])
	}
	leftAxis := int(left.Shape.Dims[axis])
	rightAxis := int(right.Shape.Dims[axis])
	outputAxis := leftAxis + rightAxis
	outer := int(outputElements) / (inner * outputAxis)
	output := make([]float32, int(outputElements))
	for group := 0; group < outer; group++ {
		outputBase := group * outputAxis * inner
		leftBase := group * leftAxis * inner
		rightBase := group * rightAxis * inner
		copy(output[outputBase:outputBase+leftAxis*inner], left.Data[leftBase:leftBase+leftAxis*inner])
		copy(
			output[outputBase+leftAxis*inner:outputBase+outputAxis*inner],
			right.Data[rightBase:rightBase+rightAxis*inner],
		)
	}
	return Value{Shape: shape, Data: output}, nil
}
