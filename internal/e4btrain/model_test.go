package e4btrain

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/hostmath"
)

// The tiny E4B-shaped decoder the tests train: the real model's layer kinds
// in miniature. A sliding layer and a full layer project their own keys and
// values; a sliding and a full layer after them attend those, as E4B's last
// layers attend the last two that project. The full layers rotate only the
// leading pairs of a wider head (the real model's frequency factors of 1e30
// stop the rest), and every layer carries a per-layer input adapter; all
// but the last scale their output.
const (
	tinyHidden       = 16
	tinyVocab        = 23
	tinyPerLayer     = 8
	tinyHeads        = 4
	tinyKVHeads      = 2
	tinySlidingWidth = 8
	tinyFullWidth    = 16
	tinyInter        = 32
	tinyWindow       = 3
	tinyTokens       = 8
	tinySoftcap      = 3
	tinyEpsilon      = 1e-6
	tinySlidingBase  = 1e4
	tinyFullBase     = 1e6
	tinyStoppedPair  = 1e30
	tinySlidingLayer = 0
	tinyFullLayer    = 1
)

func tinyGeometry() geometry {
	sliding := hostmath.RopeInvFreq(tinySlidingBase, tinySlidingWidth)
	full := hostmath.RopeInvFreq(tinyFullBase, tinyFullWidth)
	for pair := len(full) / 2; pair < len(full); pair++ {
		full[pair] /= tinyStoppedPair
	}
	layer := func(width, window, source int, invFreq []float64, scaled bool) layerGeometry {
		return layerGeometry{heads: tinyHeads, kvHeads: tinyKVHeads, headDim: width, inter: tinyInter,
			window: window, source: source, invFreq: invFreq, scaled: scaled}
	}
	return geometry{
		hidden: tinyHidden, vocab: tinyVocab, perLayer: tinyPerLayer, eps: tinyEpsilon,
		attentionScale: 1, embeddingScale: math.Sqrt(tinyHidden), logitScale: 1, softcap: tinySoftcap,
		perLayerScale: 1 / math.Sqrt2,
		layers: []layerGeometry{
			layer(tinySlidingWidth, tinyWindow, tinySlidingLayer, sliding, true),
			layer(tinyFullWidth, 0, tinyFullLayer, full, true),
			layer(tinySlidingWidth, tinyWindow, tinySlidingLayer, sliding, true),
			layer(tinyFullWidth, 0, tinyFullLayer, full, false),
		},
	}
}

// tinyModel is the tiny decoder with random weights from seed: matrices at
// unit fan-in variance, vectors (norm scales and output scales) near one.
func tinyModel(t *testing.T, seed int64) *Model {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	shape := tinyGeometry()
	random := func(n int, deviation float64) []float32 {
		values := make([]float32, n)
		for index := range values {
			values[index] = float32(rng.NormFloat64() * deviation)
		}
		return values
	}
	head := random(tinyVocab*tinyHidden, 0.5)
	perLayerRows := random(tinyTokens*len(shape.layers)*tinyPerLayer, 0.5)
	ids := make([]int, tinyTokens+1)
	for index := range ids {
		ids[index] = rng.Intn(tinyVocab)
	}
	m, err := newModel(shape, head, perLayerRows, ids)
	if err != nil {
		t.Fatal(err)
	}
	for _, group := range m.plan.Groups() {
		values := m.weights[group.Start:group.End]
		if group.Rows == 1 {
			for index := range values {
				values[index] = float32(1 + 0.2*rng.NormFloat64())
			}
			continue
		}
		copy(values, random(len(values), 1/math.Sqrt(float64(group.Cols))))
	}
	return m
}
