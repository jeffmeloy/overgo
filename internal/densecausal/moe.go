// Routed mixture-of-experts feed-forward for the dense causal-LM capability:
// top-k routing over per-layer expert sets with an optional always-on shared
// expert, the DeepSeek-V2 family's FFN. Layer kind derives from the
// artifact's own tensors (mlp.experts.* present = routed layer) and the
// router policy from config.json declarations; nothing here asserts a model
// constant. Ported behavior from the reference MoE owner, written against
// overgo's hostmath primitives.
package densecausal

import (
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
	if policy.TopK <= 0 || policy.TopK > experts {
		return moeRoute{}, nil, fmt.Errorf("densecausal: moe top_k=%d out of range [1,%d]", policy.TopK, experts)
	}
	if policy.RoutedScaling == 0 {
		return moeRoute{}, nil, fmt.Errorf("densecausal: moe routed scaling factor must be non-zero")
	}
	scores := make([]float32, rows*experts)
	hostmath.Linear(scores, x, router, rows, hidden, experts)
	for r := 0; r < rows; r++ {
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
	for r := 0; r < rows; r++ {
		row := scores[r*experts : (r+1)*experts]
		used := make([]bool, experts)
		var selectedSum float64
		for k := 0; k < policy.TopK; k++ {
			best, bestScore := -1, float32(math.Inf(-1))
			for e, score := range row {
				if !used[e] && (best < 0 || score > bestScore) {
					best, bestScore = e, score
				}
			}
			used[best] = true
			position := r*policy.TopK + k
			route.indices[position], route.weights[position] = best, bestScore
			selectedSum += float64(bestScore)
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

// moeForward: routed mixture output for the whole sequence, returning the
// route for the backward.
func moeForward(x []float32, w moeWeights, rows, hidden int, policy MoERouterPolicy) ([]float32, moeRoute, error) {
	route, _, err := routeTopK(x, w.router, rows, hidden, len(w.experts), policy)
	if err != nil {
		return nil, moeRoute{}, err
	}
	out := make([]float32, rows*hidden)
	for r := 0; r < rows; r++ {
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
	return out, route, nil
}
