package reference

import (
	"errors"
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/tensor"
)

func moe(shape tensor.Shape, inputs []Value, attributes tensor.MoEAttributes) (Value, error) {
	wantInputs := 5
	if attributes.Gated && !attributes.FusedGateUp {
		wantInputs = 6
	}
	expectedInputs := wantInputs
	if attributes.HasSelectionBias {
		expectedInputs++
	}
	if attributes.HasExpertScale {
		expectedInputs++
	}
	if attributes.HasRouterBias {
		expectedInputs++
	}
	if attributes.HasExpertBiases {
		expectedInputs += 3
	}
	if attributes.HasSelectedExperts {
		expectedInputs++
	}
	if len(inputs) != expectedInputs {
		return Value{}, errors.New("MoE input count is invalid")
	}
	input, routerInput, router := inputs[0], inputs[1], inputs[2]
	var gate Value
	next := 3
	if attributes.Gated && !attributes.FusedGateUp {
		gate = inputs[next]
		next++
	}
	up, down := inputs[next], inputs[next+1]
	if attributes.FusedGateUp {
		gate = up
	}
	var selectionBias, expertScale, routerBias, gateBias, upBias, downBias, fixedExperts []float32
	optionalIndex := next + 2
	if attributes.HasSelectionBias {
		selectionBias = inputs[optionalIndex].Data
		optionalIndex++
	}
	if attributes.HasExpertScale {
		expertScale = inputs[optionalIndex].Data
		optionalIndex++
	}
	if attributes.HasRouterBias {
		routerBias = inputs[optionalIndex].Data
		optionalIndex++
	}
	if attributes.HasExpertBiases {
		gateBias = inputs[optionalIndex].Data
		upBias = inputs[optionalIndex+1].Data
		downBias = inputs[optionalIndex+2].Data
		optionalIndex += 3
	}
	if attributes.HasSelectedExperts {
		fixedExperts = inputs[optionalIndex].Data
	}
	hidden := int(input.Shape.Dims[0])
	routerHidden := int(routerInput.Shape.Dims[0])
	tokens := int(input.Shape.Dims[1])
	experts := int(attributes.Experts)
	expertIndexDivisor := int(attributes.ExpertIndexDivisor)
	if expertIndexDivisor == 0 {
		expertIndexDivisor = 1
	}
	topK := int(attributes.TopK)
	intermediate := int(up.Shape.Dims[1])
	if attributes.FusedGateUp {
		intermediate /= 2
	}
	bankExperts := int(up.Shape.Dims[2])
	if hidden <= 0 || tokens <= 0 || experts <= 0 || topK <= 0 || topK > experts ||
		experts%expertIndexDivisor != 0 || experts/expertIndexDivisor != bankExperts ||
		topK > bankExperts || int(down.Shape.Dims[2]) != bankExperts {
		return Value{}, errors.New("invalid MoE dimensions")
	}
	output := make([]float32, hidden*tokens)
	logits := make([]float64, experts)
	probabilities := make([]float64, experts)
	selectionScores := make([]float64, experts)
	selected := make([]int, topK)
	weights := make([]float64, topK)
	used := make([]bool, experts)
	for token := 0; token < tokens; token++ {
		x := input.Data[token*hidden : (token+1)*hidden]
		routerX := routerInput.Data[token*routerHidden : (token+1)*routerHidden]
		maximum := math.Inf(-1)
		for expert := 0; expert < experts; expert++ {
			var dot float64
			for channel := 0; channel < routerHidden; channel++ {
				dot += float64(routerX[channel]) * float64(router.Data[expert*routerHidden+channel])
			}
			if routerBias != nil {
				dot += float64(routerBias[expert])
			}
			logits[expert] = dot
			maximum = math.Max(maximum, dot)
		}
		var denominator float64
		if attributes.Routing == tensor.MoERoutingSoftmax {
			for _, logit := range logits {
				denominator += math.Exp(logit - maximum)
			}
		}
		for expert, logit := range logits {
			switch attributes.Routing {
			case tensor.MoERoutingSoftmax:
				probabilities[expert] = math.Exp(logit-maximum) / denominator
			case tensor.MoERoutingSigmoid:
				probabilities[expert] = 1 / (1 + math.Exp(-logit))
			case tensor.MoERoutingSelectedSoftmax:
				probabilities[expert] = logit
			case tensor.MoERoutingSqrtSoftplus:
				probabilities[expert] = math.Sqrt(math.Max(logit, 0) + math.Log1p(math.Exp(-math.Abs(logit))))
			default:
				return Value{}, errors.New("invalid MoE routing function")
			}
			selectionScores[expert] = probabilities[expert]
			if selectionBias != nil {
				selectionScores[expert] += float64(selectionBias[expert])
			}
		}
		clear(used)
		var selectedSum float64
		for slot := 0; slot < topK; slot++ {
			if fixedExperts != nil {
				expert := int(fixedExperts[token*topK+slot])
				if expert < 0 || expert >= experts || used[expert] {
					return Value{}, errors.New("invalid fixed MoE expert selection")
				}
				used[expert] = true
				selected[slot] = expert
				weights[slot] = probabilities[expert]
				selectedSum += weights[slot]
				continue
			}
			best := -1
			bestScore := math.Inf(-1)
			for expert, score := range selectionScores {
				if !used[expert] && (best < 0 || score > bestScore) {
					best, bestScore = expert, score
				}
			}
			used[best] = true
			selected[slot] = best
			weights[slot] = probabilities[best]
			selectedSum += weights[slot]
		}
		if attributes.Routing == tensor.MoERoutingSelectedSoftmax {
			selectedMaximum := math.Inf(-1)
			for _, weight := range weights {
				selectedMaximum = math.Max(selectedMaximum, weight)
			}
			selectedSum = 0
			for slot, weight := range weights {
				weights[slot] = math.Exp(weight - selectedMaximum)
				selectedSum += weights[slot]
			}
			for slot := range weights {
				weights[slot] /= selectedSum
			}
		}
		for slot := range topK {
			if attributes.NormalizeTopKProb {
				weights[slot] = weights[slot] / math.Max(selectedSum, 6.103515625e-5) * float64(attributes.Scale)
			} else {
				weights[slot] *= float64(attributes.Scale)
			}
		}
		for outputChannel := 0; outputChannel < hidden; outputChannel++ {
			var routed float64
			for slot, routedExpert := range selected {
				expert := routedExpert / expertIndexDivisor
				var expertOutput float64
				for inner := 0; inner < intermediate; inner++ {
					gateBase := (expert*intermediate + inner) * hidden
					upBase := gateBase
					if attributes.FusedGateUp {
						gateBase = (expert*2*intermediate + inner) * hidden
						upBase = gateBase + intermediate*hidden
					}
					var gateDot, upDot float64
					for channel := 0; channel < hidden; channel++ {
						if attributes.Gated {
							gateDot += float64(x[channel]) * float64(gate.Data[gateBase+channel])
						}
						upDot += float64(x[channel]) * float64(up.Data[upBase+channel])
					}
					if gateBias != nil {
						gateDot += float64(gateBias[expert*intermediate+inner])
						upDot += float64(upBias[expert*intermediate+inner])
					}
					var activation float64
					switch attributes.Activation {
					case tensor.MoEActivationSiLU:
						activation = upDot / (1 + math.Exp(-upDot))
						if attributes.Gated {
							if attributes.SwiGLUClamp > 0 {
								limit := float64(attributes.SwiGLUClamp)
								upDot = math.Max(-limit, math.Min(limit, upDot))
								if attributes.Routing == tensor.MoERoutingSqrtSoftplus {
									gateDot = math.Min(limit, gateDot)
								}
							}
							gateActivation := gateDot / (1 + math.Exp(-gateDot))
							if attributes.SwiGLUClamp > 0 && attributes.Routing != tensor.MoERoutingSqrtSoftplus {
								gateActivation = math.Min(float64(attributes.SwiGLUClamp), gateActivation)
							}
							activation = gateActivation * upDot
						}
					case tensor.MoEActivationReLU:
						activation = math.Max(upDot, 0)
						if attributes.Gated {
							activation = math.Max(gateDot, 0) * upDot
						}
					case tensor.MoEActivationReLUSquared:
						activation = math.Max(upDot, 0)
						activation *= activation
					case tensor.MoEActivationGELU:
						activation = moeGELU(upDot)
						if attributes.Gated {
							activation = moeGELU(gateDot) * upDot
						}
					case tensor.MoEActivationSwiGLUOAI:
						x := math.Min(gateDot, 7)
						y := math.Max(-7, math.Min(7, upDot))
						activation = hostmath.QuickGELU(x) * (y + 1)
					default:
						return Value{}, errors.New("invalid MoE activation")
					}
					downIndex := (expert*hidden+outputChannel)*intermediate + inner
					expertOutput += activation * float64(down.Data[downIndex])
				}
				if downBias != nil {
					expertOutput += float64(downBias[expert*hidden+outputChannel])
				}
				if expertScale != nil {
					expertOutput *= float64(expertScale[expert])
				}
				routed += weights[slot] * expertOutput
			}
			output[token*hidden+outputChannel] = float32(routed)
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func moeGELU(value float64) float64 {
	if value <= -10 {
		return 0
	}
	if value >= 10 {
		return value
	}
	x := float64(float16Round(float32(value)))
	result := float32(hostmath.GELUTanh(x))
	return float64(float16Round(result))
}
