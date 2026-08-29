// Bit-packed catalog filter. Results remain advisory.
package composition

import (
	"container/heap"
	"fmt"
	"hash/fnv"
	"math"
	"math/bits"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/organ"
	"overgo/internal/tensorstats"
)

const (
	hdcWordBits = 64
	// Minimum signature width.
	hdcDefaultWordCount  = 8
	hdcDefaultDimensions = hdcDefaultWordCount * hdcWordBits
	// Measured resolvable margin.
	hdcResolvableScoreMargin = 0.074

	hdcBitsPerTerm  = 4
	hdcMixA         = 0x9E3779B97F4A7C15
	hdcMixC         = 0xBF58476D1CE4E5B9
	hdcMixB         = 0x6C62272E07BB0142
	hdcMixShift32   = 32
	hdcMixShift31   = 31
	hdcSignBitShift = hdcWordBits - 1
)

// HypervectorDimensionsFor: catalog-sized signature width.
func HypervectorDimensionsFor(n int) int {
	if n < 2 {
		return hdcDefaultDimensions
	}
	need := 2 * math.Log(float64(n)) / (hdcResolvableScoreMargin * hdcResolvableScoreMargin)
	words := int(math.Ceil(need / hdcWordBits))
	words = max(words, hdcDefaultWordCount)
	return words * hdcWordBits
}

// CatalogComponent is one indexed entry: identity facts plus the two signals.
type CatalogComponent struct {
	Model      artifact.ID
	Name       string
	Contract   organ.Contract
	Statistics *tensorstats.Characterization
}

// HypervectorHit is one ranked retrieval result.
type HypervectorHit struct {
	Component CatalogComponent
	Relevance float64
}

type rankedHypervectorHit struct {
	HypervectorHit
	order int
}

type hypervectorHitHeap []rankedHypervectorHit

func (h hypervectorHitHeap) Len() int { return len(h) }
func (h hypervectorHitHeap) Less(i, j int) bool {
	return betterHypervectorHit(h[j], h[i])
}
func (h hypervectorHitHeap) Swap(i, j int)   { h[i], h[j] = h[j], h[i] }
func (h *hypervectorHitHeap) Push(value any) { *h = append(*h, value.(rankedHypervectorHit)) }
func (h *hypervectorHitHeap) Pop() any {
	values := *h
	last := len(values) - 1
	value := values[last]
	values[last] = rankedHypervectorHit{}
	*h = values[:last]
	return value
}

// HypervectorIndex is the packed ternary index over catalog components.
type HypervectorIndex struct {
	dimensions int
	words      int
	sigWords   []uint64
	sigLit     []int
	components []CatalogComponent
}

// NewHypervectorIndex builds the index with catalog-derived signature width.
func NewHypervectorIndex(components []CatalogComponent) (*HypervectorIndex, error) {
	if len(components) == 0 {
		return nil, fmt.Errorf("composition: hypervector index requires components")
	}
	dimensions := HypervectorDimensionsFor(len(components))
	words := (dimensions + hdcWordBits - 1) / hdcWordBits
	index := &HypervectorIndex{
		dimensions: dimensions,
		words:      words,
		sigWords:   make([]uint64, len(components)*2*words),
		sigLit:     make([]int, len(components)),
		components: append([]CatalogComponent(nil), components...),
	}
	for componentIndex, component := range index.components {
		if component.Name == "" || !component.Model.Valid() {
			return nil, fmt.Errorf("composition: catalog component requires a model and a name")
		}
		offset := componentIndex * 2 * words
		index.sigLit[componentIndex] = encodeTermsInto(
			componentTerms(component), dimensions,
			index.sigWords[offset:offset+words],
			index.sigWords[offset+words:offset+2*words],
		)
	}
	return index, nil
}

// Len reports the indexed component count.
func (index *HypervectorIndex) Len() int { return len(index.sigLit) }

// Search: ranked lexical and distributional matches.
func (index *HypervectorIndex) Search(query CatalogComponent, limit int) ([]HypervectorHit, error) {
	if limit <= 0 {
		return nil, fmt.Errorf("composition: search limit must be positive")
	}
	queryWords := make([]uint64, 2*index.words)
	queryLit := encodeTermsInto(
		componentTerms(query), index.dimensions,
		queryWords[:index.words], queryWords[index.words:],
	)
	hits := make(hypervectorHitHeap, 0, min(limit, index.Len()))
	for i := range index.sigLit {
		offset := i * 2 * index.words
		relevance := signedSimilarity(
			queryWords[:index.words], queryWords[index.words:], queryLit,
			index.sigWords[offset:offset+index.words],
			index.sigWords[offset+index.words:offset+2*index.words],
			index.sigLit[i],
		)
		hit := rankedHypervectorHit{
			HypervectorHit: HypervectorHit{Component: index.components[i], Relevance: relevance},
			order:          i,
		}
		if len(hits) < limit {
			heap.Push(&hits, hit)
		} else if betterHypervectorHit(hit, hits[0]) {
			hits[0] = hit
			heap.Fix(&hits, 0)
		}
	}
	sort.Slice(hits, func(i, j int) bool { return betterHypervectorHit(hits[i], hits[j]) })
	results := make([]HypervectorHit, len(hits))
	for i := range hits {
		results[i] = hits[i].HypervectorHit
	}
	return results, nil
}

func betterHypervectorHit(left, right rankedHypervectorHit) bool {
	if left.Relevance != right.Relevance {
		return left.Relevance > right.Relevance
	}
	if left.Component.Name != right.Component.Name {
		return left.Component.Name < right.Component.Name
	}
	if left.Component.Model != right.Component.Model {
		return left.Component.Model.String() < right.Component.Model.String()
	}
	return left.order < right.order
}

// Candidates: cross-model bridge candidates.
func Candidates(hits []HypervectorHit, target artifact.ID) []BridgeCandidate {
	candidates := make([]BridgeCandidate, 0, len(hits))
	for _, hit := range hits {
		if hit.Component.Model == target {
			continue
		}
		distance := 1 - hit.Relevance
		if distance < 0 {
			distance = 0
		}
		candidates = append(candidates, BridgeCandidate{
			Donor: hit.Component.Model, Component: hit.Component.Name, Distance: distance,
		})
	}
	return candidates
}

// componentTerms: lexical and distributional signature terms.
func componentTerms(component CatalogComponent) []string {
	contract := component.Contract
	terms := []string{
		"modality:" + string(contract.Modality),
		"role:" + string(contract.Role),
		"space:" + string(contract.Space),
		"dtype:" + string(contract.DType),
		"layout:" + string(contract.Layout),
		"trainrole:" + string(contract.TrainingRole),
	}
	if component.Statistics != nil {
		statistics := component.Statistics
		terms = append(terms,
			fmt.Sprintf("iqr:%d", logBucket(statistics.InterquartileRange)),
			fmt.Sprintf("median:%d", logBucket(math.Abs(statistics.Median))),
			fmt.Sprintf("l2:%d", logBucket(statistics.LMoments.L2)),
			fmt.Sprintf("tau3:%d", ratioBucket(statistics.LMoments.Tau3)),
			fmt.Sprintf("tau4:%d", ratioBucket(statistics.LMoments.Tau4)),
		)
	}
	return terms
}

// logBucket: coarse magnitude decade.
func logBucket(value float64) int {
	if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
		return -100
	}
	bucket := int(math.Floor(math.Log10(value)))
	if bucket < -12 {
		bucket = -12
	}
	if bucket > 12 {
		bucket = 12
	}
	return bucket
}

// ratioBucket quantizes a bounded L-moment ratio into tenths.
func ratioBucket(value float64) int {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return -100
	}
	if value > 1 {
		value = 1
	}
	if value < -1 {
		value = -1
	}
	return int(math.Round(value * 10))
}

func termHash(term string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(term))
	return h.Sum64()
}

func mix(x uint64) uint64 {
	x *= hdcMixA
	x ^= x >> hdcMixShift32
	x *= hdcMixC
	x ^= x >> hdcMixShift31
	return x
}

func encodeTermsInto(terms []string, dimensions int, pos, neg []uint64) int {
	for _, term := range terms {
		base := termHash(term)
		for slot := range hdcBitsPerTerm {
			h := mix(base + uint64(slot)*hdcMixB)
			dim := int(h % uint64(dimensions))
			mask := uint64(1) << uint(dim%hdcWordBits)
			word := dim / hdcWordBits
			if ((h >> hdcSignBitShift) & 1) == 0 {
				pos[word] |= mask
				neg[word] &^= mask
			} else {
				neg[word] |= mask
				pos[word] &^= mask
			}
		}
	}
	lit := 0
	for i := range pos {
		lit += bits.OnesCount64(pos[i] | neg[i])
	}
	return lit
}

// signedSimilarity: normalized signed-Hamming agreement.
func signedSimilarity(aPos, aNeg []uint64, aLit int, bPos, bNeg []uint64, bLit int) float64 {
	n := min(len(aPos), len(bPos))
	if n == 0 || aLit == 0 || bLit == 0 {
		return 0
	}
	matches, conflicts := 0, 0
	for i := range n {
		matches += bits.OnesCount64(aPos[i]&bPos[i]) + bits.OnesCount64(aNeg[i]&bNeg[i])
		conflicts += bits.OnesCount64(aPos[i]&bNeg[i]) + bits.OnesCount64(aNeg[i]&bPos[i])
	}
	return float64(matches-conflicts) / math.Sqrt(float64(aLit*bLit))
}
