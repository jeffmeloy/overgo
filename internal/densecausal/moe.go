// Routed mixture-of-experts feed-forward for the dense causal-LM capability:
// top-k routing over per-layer expert sets with an optional always-on shared
// expert, the DeepSeek-V2 family's FFN. Layer kind derives from the
// artifact's own tensors (mlp.experts.* present = routed layer) and the
// router policy from config.json declarations; nothing here asserts a model
// constant. Ported behavior from the reference MoE owner, written against
// overgo's hostmath primitives.
package densecausal

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// MoEScoring names the router's scoring activation.
type MoEScoring string

const (
	MoEScoringSoftmax MoEScoring = "softmax"
	MoEScoringSigmoid MoEScoring = "sigmoid"
)

// MoERouterPolicy: routing facts declared by the artifact's config.
type MoERouterPolicy struct {
	TopK              int
	Scoring           MoEScoring
	NormalizeTopKProb bool
	RoutedScaling     float32
	ExpertInter       int
}

// MoERouterMargin preserves whether an unselected expert existed for an
// observed row. Value is selected-boundary score minus best unselected score.
type MoERouterMargin struct {
	Observed bool
	Value    float32
}

// MoERouterObservation is the exact, policy-neutral route used by one layer.
// Selections and combine weights are row-major in router rank order.
type MoERouterObservation struct {
	Layer          int
	Rows           int
	Experts        int
	TopK           int
	Selections     []int
	CombineWeights []float32
	Accepted       []bool
	Margins        []MoERouterMargin
}

// moeRoute: fixed top-k selection with combine weights, per row.
type moeRoute struct {
	indices []int
	weights []float32
}

// moeExpert: one expert's SwiGLU weights.
type moeExpert struct {
	gate, up, down []float32
}

// moeWeights: one routed layer's mixture weights.
type moeWeights struct {
	router      []float32
	experts     []moeExpert
	shared      *moeExpert
	sharedInter int
	names       moeTensorNames
}

type moeTensorNames struct {
	router  string
	experts []moeExpertNames
	shared  *moeExpertNames
}

type moeExpertNames struct {
	gate, up, down string
}

// routeTopK computes activated router scores and the fixed top-k selection.
// The selection is piecewise-constant in the scores, so backward treats it
// as fixed and gradient flows only to the selected magnitudes.
func routeTopK(x, router []float32, rows, hidden, experts int, policy MoERouterPolicy) (moeRoute, []float32, error) {
	return routeTopKSelectionBias(x, router, rows, hidden, experts, policy, nil)
}

func routeTopKSelectionBias(x, router []float32, rows, hidden, experts int, policy MoERouterPolicy, selectionBias []float32) (moeRoute, []float32, error) {
	if policy.TopK <= 0 || policy.TopK > experts {
		return moeRoute{}, nil, fmt.Errorf("densecausal: moe top_k=%d out of range [1,%d]", policy.TopK, experts)
	}
	if policy.RoutedScaling == 0 {
		return moeRoute{}, nil, fmt.Errorf("densecausal: moe routed scaling factor must be non-zero")
	}
	if len(selectionBias) != 0 && len(selectionBias) != experts {
		return moeRoute{}, nil, errors.New("densecausal: moe selection bias differs from expert inventory")
	}
	for _, bias := range selectionBias {
		if math.IsNaN(float64(bias)) || math.IsInf(float64(bias), 0) {
			return moeRoute{}, nil, errors.New("densecausal: moe selection bias must be finite")
		}
	}
	scores := make([]float32, rows*experts)
	hostmath.Linear(scores, x, router, rows, hidden, experts)
	for r := range rows {
		row := scores[r*experts : (r+1)*experts]
		switch policy.Scoring {
		case MoEScoringSoftmax:
			hostmath.SoftmaxInPlace(row)
		case MoEScoringSigmoid:
			for i, v := range row {
				row[i] = float32(1 / (1 + math.Exp(-float64(v))))
			}
		default:
			return moeRoute{}, nil, fmt.Errorf("densecausal: unsupported moe scoring %q", policy.Scoring)
		}
	}
	route := moeRoute{indices: make([]int, rows*policy.TopK), weights: make([]float32, rows*policy.TopK)}
	used := make([]bool, experts)
	for r := range rows {
		row := scores[r*experts : (r+1)*experts]
		clear(used)
		var selectedSum float64
		for k := 0; k < policy.TopK; k++ {
			best, bestSelectionScore := -1, float32(math.Inf(-1))
			for e, score := range row {
				selectionScore := score
				if len(selectionBias) != 0 {
					selectionScore += selectionBias[e]
				}
				if !used[e] && (best < 0 || selectionScore > bestSelectionScore) {
					best, bestSelectionScore = e, selectionScore
				}
			}
			used[best] = true
			position := r*policy.TopK + k
			route.indices[position], route.weights[position] = best, row[best]
			selectedSum += float64(row[best])
		}
		if policy.NormalizeTopKProb && policy.TopK > 1 {
			if selectedSum == 0 {
				return moeRoute{}, nil, fmt.Errorf("densecausal: moe selected scores sum to zero at row %d", r)
			}
			for k := 0; k < policy.TopK; k++ {
				position := r*policy.TopK + k
				route.weights[position] = float32(float64(route.weights[position]) / selectedSum * float64(policy.RoutedScaling))
			}
			continue
		}
		for k := 0; k < policy.TopK; k++ {
			route.weights[r*policy.TopK+k] *= policy.RoutedScaling
		}
	}
	return route, scores, nil
}

// expertForward: SwiGLU expert on rows of x.
func expertForward(x []float32, expert moeExpert, rows, hidden, inter int) []float32 {
	gate := make([]float32, rows*inter)
	up := make([]float32, rows*inter)
	hostmath.Linear(gate, x, expert.gate, rows, hidden, inter)
	hostmath.Linear(up, x, expert.up, rows, hidden, inter)
	hostmath.SiLUInPlace(gate)
	for i := range gate {
		gate[i] *= up[i]
	}
	out := make([]float32, rows*hidden)
	hostmath.Linear(out, gate, expert.down, rows, inter, hidden)
	return out
}

// moeForward returns the routed mixture output, exact route, and activated
// scores already computed for that route.
func moeForward(x []float32, w moeWeights, rows, hidden int, policy MoERouterPolicy) ([]float32, moeRoute, []float32, error) {
	route, scores, err := routeTopK(x, w.router, rows, hidden, len(w.experts), policy)
	if err != nil {
		return nil, moeRoute{}, nil, err
	}
	out := make([]float32, rows*hidden)
	for r := range rows {
		row := x[r*hidden : (r+1)*hidden]
		for k := 0; k < policy.TopK; k++ {
			position := r*policy.TopK + k
			expertOut := expertForward(row, w.experts[route.indices[position]], 1, hidden, policy.ExpertInter)
			weight := route.weights[position]
			for i, v := range expertOut {
				out[r*hidden+i] += v * weight
			}
		}
	}
	if w.shared != nil {
		addInPlace(out, expertForward(x, *w.shared, rows, hidden, w.sharedInter))
	}
	return out, route, scores, nil
}

func newMoERouterObservation(layer int, route moeRoute, scores []float32, rows, experts, topK int) (MoERouterObservation, error) {
	if layer < 0 || rows <= 0 || experts <= 0 || topK <= 0 || topK > experts ||
		len(route.indices) != rows*topK || len(route.weights) != rows*topK || len(scores) != rows*experts {
		return MoERouterObservation{}, errors.New("densecausal: invalid moe router observation geometry")
	}
	observation := MoERouterObservation{
		Layer: layer, Rows: rows, Experts: experts, TopK: topK,
		Selections:     append([]int(nil), route.indices...),
		CombineWeights: append([]float32(nil), route.weights...),
		Accepted:       make([]bool, len(route.indices)), Margins: make([]MoERouterMargin, rows),
	}
	for index := range observation.Accepted {
		observation.Accepted[index] = true
	}
	if topK == experts {
		return observation, nil
	}
	selected := make([]bool, experts)
	for row := range rows {
		clear(selected)
		start := row * topK
		for _, expert := range route.indices[start : start+topK] {
			selected[expert] = true
		}
		bestUnselected := float32(math.Inf(-1))
		for expert, score := range scores[row*experts : (row+1)*experts] {
			if !selected[expert] && score > bestUnselected {
				bestUnselected = score
			}
		}
		observation.Margins[row] = MoERouterMargin{
			Observed: true,
			Value:    scores[row*experts+route.indices[start+topK-1]] - bestUnselected,
		}
	}
	return observation, nil
}
