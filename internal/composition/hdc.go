// Bit-packed ternary hypervector retrieval over the component catalog,
// ported from adaptive_new go/extmodel/hdc.go as a verified capability: FNV
// term hashing, a splitmix-style bit mixer, signed ternary encoding, and
// signed-Hamming similarity normalized by sqrt(Lit_a*Lit_b). The index signs
// the LEXICAL organ contract (modality, role, space, dtype, layout, training
// role) together with the DISTRIBUTIONAL signal (quantized quartiles and
// L-moment ratios of the committed tensor measurements), so retrieval ranks
// components by both what they are and how their values behave. Advisory by
// construction: hits feed blocked bridge proposals, never promotions.
package composition

import (
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
	// hdcDefaultWordCount is the FLOOR width; real catalogs derive their
	// width per size through HypervectorDimensionsFor.
	hdcDefaultWordCount  = 8
	hdcDefaultDimensions = hdcDefaultWordCount * hdcWordBits
	// hdcResolvableScoreMargin is a MEASURED quantity carried from the
	// source port (adaptive_new ADV-49, live margin 0.0742 measured
	// 2026-08-02), not a knob.
	hdcResolvableScoreMargin = 0.074

	hdcBitsPerTerm  = 4
	hdcMixA         = 0x9E3779B97F4A7C15
	hdcMixC         = 0xBF58476D1CE4E5B9
	hdcMixB         = 0x6C62272E07BB0142
	hdcMixShift32   = 32
	hdcMixShift31   = 31
	hdcSignBitShift = hdcWordBits - 1
)

// HypervectorDimensionsFor derives the catalog-sized signature width from the
// extreme-value bound at the measured resolvable margin.
func HypervectorDimensionsFor(n int) int {
	if n < 2 {
		return hdcDefaultDimensions
	}
	need := 2 * math.Log(float64(n)) / (hdcResolvableScoreMargin * hdcResolvableScoreMargin)
	words := int(math.Ceil(need / hdcWordBits))
	if words < hdcDefaultWordCount {
		words = hdcDefaultWordCount
	}
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
	index := &HypervectorIndex{
		dimensions: dimensions,
		words:      (dimensions + hdcWordBits - 1) / hdcWordBits,
		components: append([]CatalogComponent(nil), components...),
	}
	for _, component := range index.components {
		if component.Name == "" || !component.Model.Valid() {
			return nil, fmt.Errorf("composition: catalog component requires a model and a name")
		}
		offset := len(index.sigWords)
		index.sigWords = append(index.sigWords, make([]uint64, 2*index.words)...)
		index.sigLit = append(index.sigLit, encodeTermsInto(
			componentTerms(component), dimensions,
			index.sigWords[offset:offset+index.words],
			index.sigWords[offset+index.words:offset+2*index.words],
		))
	}
	return index, nil
}

// Len reports the indexed component count.
func (index *HypervectorIndex) Len() int { return len(index.sigLit) }

// Search ranks the catalog against a query component by signed-Hamming
// similarity over the combined lexical and distributional signature.
func (index *HypervectorIndex) Search(query CatalogComponent, limit int) ([]HypervectorHit, error) {
	if limit <= 0 {
		return nil, fmt.Errorf("composition: search limit must be positive")
	}
	queryWords := make([]uint64, 2*index.words)
	queryLit := encodeTermsInto(
		componentTerms(query), index.dimensions,
		queryWords[:index.words], queryWords[index.words:],
	)
	hits := make([]HypervectorHit, 0, index.Len())
	for i := range index.sigLit {
		offset := i * 2 * index.words
		relevance := signedSimilarity(
			queryWords[:index.words], queryWords[index.words:], queryLit,
			index.sigWords[offset:offset+index.words],
			index.sigWords[offset+index.words:offset+2*index.words],
			index.sigLit[i],
		)
		hits = append(hits, HypervectorHit{Component: index.components[i], Relevance: relevance})
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].Relevance != hits[j].Relevance {
			return hits[i].Relevance > hits[j].Relevance
		}
		if hits[i].Component.Name != hits[j].Component.Name {
			return hits[i].Component.Name < hits[j].Component.Name
		}
		return hits[i].Component.Model.String() < hits[j].Component.Model.String()
	})
	if len(hits) > limit {
		hits = hits[:limit]
	}
	return hits, nil
}

// Candidates converts ranked hits into bridge candidates for a blocked
// proposal, excluding within-model hits.
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

// componentTerms combines the lexical organ signal with the distributional
// signal into the term bag one signature encodes.
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

// logBucket quantizes a positive magnitude into a coarse decade bucket so
// nearby distributions share terms without exact-value brittleness.
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
		for slot := 0; slot < hdcBitsPerTerm; slot++ {
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

// signedSimilarity is signed Hamming agreement over lit ternary positions,
// normalized by sqrt(Lit_a*Lit_b): a bounded [-1,1] similarity.
func signedSimilarity(aPos, aNeg []uint64, aLit int, bPos, bNeg []uint64, bLit int) float64 {
	n := min(len(aPos), len(bPos))
	if n == 0 || aLit == 0 || bLit == 0 {
		return 0
	}
	matches, conflicts := 0, 0
	for i := 0; i < n; i++ {
		matches += bits.OnesCount64(aPos[i]&bPos[i]) + bits.OnesCount64(aNeg[i]&bNeg[i])
		conflicts += bits.OnesCount64(aPos[i]&bNeg[i]) + bits.OnesCount64(aNeg[i]&bPos[i])
	}
	return float64(matches-conflicts) / math.Sqrt(float64(aLit*bLit))
}
