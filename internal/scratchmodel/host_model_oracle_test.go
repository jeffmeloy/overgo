package scratchmodel

import (
	"errors"
	"math"
	"math/rand"
	"slices"
	"sort"
	"strconv"
)

type hostState map[string][][]*value

type HostGroup struct {
	Name      string
	Rows      int
	Cols      int
	Weights   []float64
	Gradients []float64
}

type HostProbe struct {
	Loss   float64
	Logits [][][]float64
	Groups []HostGroup
}

func (c Construction) Probe(documents []string) (HostProbe, error) {
	if len(documents) == 0 {
		return HostProbe{}, errors.New("scratch model: probe documents absent")
	}
	state := c.hostState()
	model := hostModel{config: c.config, state: state}
	result := HostProbe{Logits: make([][][]float64, len(documents))}
	lossScale := 1 / float64(len(documents))
	for index, document := range documents {
		tokens, err := c.Tokens(document)
		if err != nil {
			return HostProbe{}, err
		}
		loss, logits := model.document(tokens, lossScale)
		result.Loss += loss
		result.Logits[index] = logits
	}
	result.Groups = flattenHostState(state)
	return result, nil
}

func (c Construction) hostState() hostState {
	flat := c.oracleWeights()
	state := make(hostState, len(c.parameters))
	for _, parameter := range c.parameters {
		binding, ok := c.bindings[parameter.Name]
		if !ok {
			panic("scratch oracle: parameter binding absent")
		}
		values := flat[binding.start:binding.end]
		matrix := make([][]*value, parameter.Rows)
		for row := range parameter.Rows {
			matrix[row] = make([]*value, parameter.Cols)
			for column := range parameter.Cols {
				matrix[row][column] = &value{data: values[row*parameter.Cols+column]}
			}
		}
		state[parameter.Name] = matrix
	}
	return state
}

func (c Construction) oracleWeights() []float64 {
	ordered := slices.Clone(c.parameters)
	sort.Slice(ordered, func(left, right int) bool {
		return c.bindings[ordered[left].Name].start < c.bindings[ordered[right].Name].start
	})
	rng := rand.New(rand.NewSource(c.seed))
	weights := make([]float64, len(c.weights))
	for _, parameter := range ordered {
		binding := c.bindings[parameter.Name]
		if parameter.Initializer != InitializerUniform {
			continue
		}
		for index := binding.start; index < binding.end; index++ {
			weights[index] = (rng.Float64()*2 - 1) * c.config.InitStd
		}
	}
	return weights
}

func flattenHostState(state hostState) []HostGroup {
	names := make([]string, 0, len(state))
	for name := range state {
		names = append(names, name)
	}
	sort.Strings(names)
	groups := make([]HostGroup, len(names))
	for index, name := range names {
		matrix := state[name]
		rows, columns := len(matrix), len(matrix[0])
		group := HostGroup{Name: name, Rows: rows, Cols: columns, Weights: make([]float64, rows*columns), Gradients: make([]float64, rows*columns)}
		for row := range rows {
			for column := range columns {
				item := matrix[row][column]
				group.Weights[row*columns+column] = item.data
				group.Gradients[row*columns+column] = item.grad
			}
		}
		groups[index] = group
	}
	return groups
}

type hostModel struct {
	config Config
	state  hostState
}

func (m hostModel) document(tokens []int, lossScale float64) (float64, [][]float64) {
	positions := min(m.config.BlockSize, len(tokens)-1)
	t := &tape{}
	hidden := make([][]*value, positions)
	for position := range positions {
		hidden[position] = m.embedding(t, tokens[position], position)
	}
	for layer := range m.config.LayerCount {
		qkv := m.qkvBatch(t, hidden, layer)
		attention := m.attentionBatch(t, qkv, layer)
		hidden = m.woBatch(t, attention, hidden, layer)
		hidden = m.mlpBatch(t, hidden, layer)
	}
	losses := make([]*value, 0, positions)
	logits := make([][]float64, 0, positions)
	for position := range positions {
		loss, row := m.outputLoss(t, hidden[position], tokens[position+1])
		losses = append(losses, loss)
		logits = append(logits, row)
	}
	var sum float64
	for _, loss := range losses {
		sum += loss.data
	}
	root := t.node(sum / float64(len(losses)))
	scale := lossScale / float64(len(losses))
	t.append(func() {
		if root.grad == 0 {
			return
		}
		for _, loss := range losses {
			loss.grad += scale * root.grad
		}
	})
	t.run(root)
	return root.data, logits
}

func (m hostModel) embedding(t *tape, token, position int) []*value {
	tokenWeights := m.state["wte"][token]
	positionWeights := m.state["wpe"][position]
	input := make([]*value, m.config.Embedding)
	for index := range input {
		input[index] = t.node(tokenWeights[index].data + positionWeights[index].data)
	}
	t.append(func() {
		for index := range input {
			tokenWeights[index].grad += input[index].grad
			positionWeights[index].grad += input[index].grad
		}
	})
	normalizedWeights, rawWeights := rankWeights(m.config.Embedding)
	return normalize(t, input, normalizedWeights, rawWeights, m.config.Epsilon)
}

type batchNormState struct {
	normed, a, b []float64
	c, scale     []float64
}

func (m hostModel) qkvBatch(t *tape, inputs [][]*value, layer int) [][]*value {
	positions, embedding := len(inputs), m.config.Embedding
	weights := m.state["l"+strconv.Itoa(layer)+".wqkv"]
	outputWidth := len(weights)
	normalizedWeights, rawWeights := rankWeights(embedding)
	state := batchNormState{
		normed: make([]float64, positions*embedding), a: make([]float64, positions*embedding),
		b: make([]float64, positions*embedding), c: make([]float64, positions), scale: make([]float64, positions),
	}
	for position := range positions {
		row := state.normed[position*embedding : (position+1)*embedding]
		for index := range embedding {
			row[index] = inputs[position][index].data
		}
		row, a, b, c, scale := madnormForward(row, normalizedWeights, rawWeights, m.config.Epsilon)
		copy(state.normed[position*embedding:], row)
		copy(state.a[position*embedding:], a)
		copy(state.b[position*embedding:], b)
		state.c[position], state.scale[position] = c, scale
	}
	weightData := flattenValueData(weights)
	weightTranspose := make([]float64, embedding*outputWidth)
	for row := range outputWidth {
		for column := range embedding {
			weightTranspose[column*outputWidth+row] = weightData[row*embedding+column]
		}
	}
	outputData := make([]float64, positions*outputWidth)
	matrixMultiply(outputData, state.normed, positions, embedding, weightTranspose, outputWidth)
	outputs := valueMatrix(t, outputData, positions, outputWidth)
	t.append(func() {
		outputGradient := flattenValueGrad(outputs)
		normalizedGradient := make([]float64, positions*embedding)
		weightGradient := make([]float64, outputWidth*embedding)
		matrixMultiply(normalizedGradient, outputGradient, positions, outputWidth, weightData, embedding)
		matrixMultiplyTransA(weightGradient, outputGradient, positions, outputWidth, state.normed, embedding)
		for position := range positions {
			start := position * embedding
			gradient := madnormBackward(
				normalizedGradient[start:start+embedding], state.normed[start:start+embedding],
				state.a[start:start+embedding], state.b[start:start+embedding], state.c[position], state.scale[position],
			)
			for index := range embedding {
				inputs[position][index].grad += gradient[index]
			}
		}
		accumulateValueGradient(weights, weightGradient)
	})
	return outputs
}

type attentionSaved struct {
	start, window int
	probabilities []float64
	data          []float64
	output        []*value
}

func (m hostModel) attentionBatch(t *tape, qkv [][]*value, layer int) [][]*value {
	positions, embedding := len(qkv), m.config.Embedding
	positionBias, temperature := m.state["pos_bias"][0], m.state["lt"][layer]
	inverseTemperature := make([]float64, m.config.HeadCount)
	for head := range m.config.HeadCount {
		inverseTemperature[head] = math.Exp(-temperature[head].data) / math.Sqrt(float64(m.config.HeadDim))
	}
	saved := make([]attentionSaved, positions)
	outputs := make([][]*value, positions)
	for position := range positions {
		vector := qkv[position][2*embedding:]
		if position == 0 {
			outputs[position] = vector
			saved[position] = attentionSaved{output: vector}
			continue
		}
		start := max(0, position+1-m.config.AttentionWindow)
		window := position + 1 - start
		probabilities := make([]float64, m.config.HeadCount*window)
		query := qkv[position][:embedding]
		for head := range m.config.HeadCount {
			headStart := head * m.config.HeadDim
			logits := probabilities[head*window : (head+1)*window]
			maximum := -1e308
			for offset := range window {
				key := qkv[start+offset][embedding : 2*embedding]
				var dot float64
				for column := range m.config.HeadDim {
					dot += query[headStart+column].data * key[headStart+column].data
				}
				lag := min(m.config.BlockSize-1, position-(start+offset))
				logits[offset] = dot*inverseTemperature[head] + positionBias[lag].data
				maximum = max(maximum, logits[offset])
			}
			var sum float64
			for offset := range window {
				logits[offset] = math.Exp(logits[offset] - maximum)
				sum += logits[offset]
			}
			for offset := range window {
				logits[offset] /= sum
			}
		}
		outputData := make([]float64, embedding)
		for offset := range window {
			valueVector := qkv[start+offset][2*embedding:]
			for head := range m.config.HeadCount {
				headStart := head * m.config.HeadDim
				probability := probabilities[head*window+offset]
				for column := range m.config.HeadDim {
					outputData[headStart+column] += probability * valueVector[headStart+column].data
				}
			}
		}
		outputs[position] = valueRow(t, outputData)
		saved[position] = attentionSaved{start: start, window: window, probabilities: probabilities, data: outputData, output: outputs[position]}
	}
	t.append(func() {
		temperatureGradient := make([]float64, m.config.HeadCount)
		for position := positions - 1; position >= 1; position-- {
			item := saved[position]
			query := qkv[position][:embedding]
			outputGradientDot := make([]float64, m.config.HeadCount)
			for head := range m.config.HeadCount {
				headStart := head * m.config.HeadDim
				for column := range m.config.HeadDim {
					index := headStart + column
					outputGradientDot[head] += item.data[index] * item.output[index].grad
				}
			}
			for offset := range item.window {
				key := qkv[item.start+offset][embedding : 2*embedding]
				vector := qkv[item.start+offset][2*embedding:]
				lag := min(m.config.BlockSize-1, position-(item.start+offset))
				for head := range m.config.HeadCount {
					headStart := head * m.config.HeadDim
					probability := item.probabilities[head*item.window+offset]
					var vectorGradientDot float64
					for column := range m.config.HeadDim {
						index := headStart + column
						vectorGradientDot += vector[index].data * item.output[index].grad
						vector[index].grad += probability * item.output[index].grad
					}
					logitGradient := probability * (vectorGradientDot - outputGradientDot[head])
					if logitGradient == 0 {
						continue
					}
					positionBias[lag].grad += logitGradient
					dotGradient := logitGradient * inverseTemperature[head]
					var dot float64
					for column := range m.config.HeadDim {
						index := headStart + column
						dot += query[index].data * key[index].data
						query[index].grad += key[index].data * dotGradient
						key[index].grad += query[index].data * dotGradient
					}
					temperatureGradient[head] += logitGradient * dot
				}
			}
		}
		for head := range m.config.HeadCount {
			temperature[head].grad += -inverseTemperature[head] * temperatureGradient[head]
		}
	})
	return outputs
}

func (m hostModel) woBatch(t *tape, attention, residual [][]*value, layer int) [][]*value {
	positions, embedding := len(attention), m.config.Embedding
	weights := m.state["l"+strconv.Itoa(layer)+".wo"]
	attentionData := flattenValueData(attention)
	weightData := flattenValueData(weights)
	weightTranspose := make([]float64, embedding*embedding)
	for row := range embedding {
		for column := range embedding {
			weightTranspose[column*embedding+row] = weightData[row*embedding+column]
		}
	}
	outputData := make([]float64, positions*embedding)
	matrixMultiply(outputData, attentionData, positions, embedding, weightTranspose, embedding)
	for position := range positions {
		for index := range embedding {
			outputData[position*embedding+index] += residual[position][index].data
		}
	}
	outputs := valueMatrix(t, outputData, positions, embedding)
	t.append(func() {
		outputGradient := flattenValueGrad(outputs)
		for position := range positions {
			for index := range embedding {
				residual[position][index].grad += outputGradient[position*embedding+index]
			}
		}
		attentionGradient := make([]float64, positions*embedding)
		weightGradient := make([]float64, embedding*embedding)
		matrixMultiply(attentionGradient, outputGradient, positions, embedding, weightData, embedding)
		matrixMultiplyTransA(weightGradient, outputGradient, positions, embedding, attentionData, embedding)
		accumulateValueGradient(attention, attentionGradient)
		accumulateValueGradient(weights, weightGradient)
	})
	return outputs
}

func (m hostModel) mlpBatch(t *tape, inputs [][]*value, layer int) [][]*value {
	positions, embedding, width := len(inputs), m.config.Embedding, m.config.MLPWidth
	w1 := m.state["l"+strconv.Itoa(layer)+".w1"]
	w2 := m.state["l"+strconv.Itoa(layer)+".w2"]
	inputData := flattenValueData(inputs)
	normalizedWeights, rawWeights := rankWeights(embedding)
	state := batchNormState{
		normed: make([]float64, positions*embedding), a: make([]float64, positions*embedding),
		b: make([]float64, positions*embedding), c: make([]float64, positions), scale: make([]float64, positions),
	}
	for position := range positions {
		start := position * embedding
		row, a, b, c, scale := madnormForward(inputData[start:start+embedding], normalizedWeights, rawWeights, m.config.Epsilon)
		copy(state.normed[start:], row)
		copy(state.a[start:], a)
		copy(state.b[start:], b)
		state.c[position], state.scale[position] = c, scale
	}
	w1Data, w2Data := flattenValueData(w1), flattenValueData(w2)
	hidden := make([]float64, positions*width)
	matrixMultiplyBT(hidden, state.normed, positions, embedding, w1Data, width)
	for index := range hidden {
		if hidden[index] <= 0 {
			hidden[index] = 0
		}
	}
	outputData := make([]float64, positions*embedding)
	matrixMultiplyBT(outputData, hidden, positions, width, w2Data, embedding)
	for index := range outputData {
		outputData[index] += inputData[index]
	}
	outputs := valueMatrix(t, outputData, positions, embedding)
	t.append(func() {
		outputGradient := flattenValueGrad(outputs)
		inputGradient := slices.Clone(outputGradient)
		hiddenGradient := make([]float64, positions*width)
		matrixMultiply(hiddenGradient, outputGradient, positions, embedding, w2Data, width)
		for index := range hiddenGradient {
			if hidden[index] <= 0 {
				hiddenGradient[index] = 0
			}
		}
		w2Gradient := make([]float64, embedding*width)
		w1Gradient := make([]float64, width*embedding)
		matrixMultiplyTransA(w2Gradient, outputGradient, positions, embedding, hidden, width)
		matrixMultiplyTransA(w1Gradient, hiddenGradient, positions, width, state.normed, embedding)
		normalizedGradient := make([]float64, positions*embedding)
		matrixMultiply(normalizedGradient, hiddenGradient, positions, width, w1Data, embedding)
		for position := range positions {
			start := position * embedding
			gradient := madnormBackward(
				normalizedGradient[start:start+embedding], state.normed[start:start+embedding],
				state.a[start:start+embedding], state.b[start:start+embedding], state.c[position], state.scale[position],
			)
			for index := range embedding {
				inputGradient[start+index] += gradient[index]
			}
		}
		accumulateValueGradient(inputs, inputGradient)
		accumulateValueGradient(w1, w1Gradient)
		accumulateValueGradient(w2, w2Gradient)
	})
	return outputs
}

func flattenValueData(matrix [][]*value) []float64 {
	columns := len(matrix[0])
	result := make([]float64, len(matrix)*columns)
	for row := range matrix {
		for column := range matrix[row] {
			result[row*columns+column] = matrix[row][column].data
		}
	}
	return result
}

func flattenValueGrad(matrix [][]*value) []float64 {
	columns := len(matrix[0])
	result := make([]float64, len(matrix)*columns)
	for row := range matrix {
		for column := range matrix[row] {
			result[row*columns+column] = matrix[row][column].grad
		}
	}
	return result
}

func valueRow(t *tape, data []float64) []*value {
	result := make([]*value, len(data))
	for index, item := range data {
		result[index] = t.node(item)
	}
	return result
}

func valueMatrix(t *tape, data []float64, rows, columns int) [][]*value {
	result := make([][]*value, rows)
	for row := range rows {
		result[row] = valueRow(t, data[row*columns:(row+1)*columns])
	}
	return result
}

func accumulateValueGradient(matrix [][]*value, gradient []float64) {
	columns := len(matrix[0])
	for row := range matrix {
		for column := range matrix[row] {
			matrix[row][column].grad += gradient[row*columns+column]
		}
	}
}

func (m hostModel) outputLoss(t *tape, input []*value, target int) (*value, []float64) {
	weights := m.state["lm_head"]
	logits := make([]float64, len(weights))
	for row, matrixRow := range weights {
		for column := range input {
			logits[row] += matrixRow[column].data * input[column].data
		}
	}
	probabilities := softmax(logits)
	floor := max(probabilities[target], m.config.Epsilon)
	loss := t.node(-math.Log(floor))
	t.append(func() {
		if loss.grad == 0 {
			return
		}
		inputGradient := make([]float64, len(input))
		for row, matrixRow := range weights {
			gradient := probabilities[row]
			if row == target {
				gradient--
			}
			gradient *= loss.grad
			if gradient == 0 {
				continue
			}
			for column := range input {
				matrixRow[column].grad += gradient * input[column].data
				inputGradient[column] += matrixRow[column].data * gradient
			}
		}
		for column := range input {
			input[column].grad += inputGradient[column]
		}
	})
	return loss, slices.Clone(logits)
}
