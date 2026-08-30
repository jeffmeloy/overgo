package composition

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	// descriptorUnmeasuredBucket marks a characterization field with no
	// measured value; it matches only other unmeasured fields.
	descriptorUnmeasuredBucket = -100
)

// CanonicalComponentDescriptor is the fixed-schema, unit-normalized
// characterization of one catalog component — the durable asset the
// composition driver and any future fingerprint reuse. The typed contract
// fields carry the lexical signal verbatim; the distributional signal is
// normalized by the component's own L-scale, so location and spread
// compare across models regardless of weight magnitude, while the scale
// bucket records the magnitude itself. Every field is a closed enumerated
// bucket: two descriptors are comparable by field equality alone.
type CanonicalComponentDescriptor struct {
	Model        artifact.ID `json:"model"`
	Name         string      `json:"name"`
	Modality     string      `json:"modality"`
	Role         string      `json:"role"`
	Space        string      `json:"space"`
	DType        string      `json:"dtype"`
	Layout       string      `json:"layout"`
	TrainingRole string      `json:"training_role"`
	Measured     bool        `json:"measured"`
	Scale        int         `json:"scale"`
	Location     int         `json:"location"`
	Spread       int         `json:"spread"`
	Tau3         int         `json:"tau3"`
	Tau4         int         `json:"tau4"`
}

// CanonicalDescriptor derives the fixed-schema descriptor for one catalog
// component. A component without measurements is still a valid descriptor
// with every characterization bucket at the unmeasured sentinel.
func CanonicalDescriptor(component CatalogComponent) (CanonicalComponentDescriptor, error) {
	if component.Name == "" || !component.Model.Valid() {
		return CanonicalComponentDescriptor{}, errors.New("composition: canonical descriptor requires a model and a name")
	}
	contract := component.Contract
	descriptor := CanonicalComponentDescriptor{
		Model: component.Model, Name: component.Name,
		Modality: string(contract.Modality), Role: string(contract.Role),
		Space: string(contract.Space), DType: string(contract.DType),
		Layout: string(contract.Layout), TrainingRole: string(contract.TrainingRole),
		Scale: descriptorUnmeasuredBucket, Location: descriptorUnmeasuredBucket,
		Spread: descriptorUnmeasuredBucket, Tau3: descriptorUnmeasuredBucket,
		Tau4: descriptorUnmeasuredBucket,
	}
	if component.Statistics == nil {
		return descriptor, nil
	}
	statistics := component.Statistics
	scale := statistics.LMoments.L2
	if !(scale > 0) || math.IsInf(scale, 0) {
		return CanonicalComponentDescriptor{}, fmt.Errorf(
			"composition: component %s measurements lack a positive L-scale", component.Name,
		)
	}
	descriptor.Measured = true
	descriptor.Scale = logBucket(scale)
	descriptor.Location = logBucket(math.Abs(statistics.Median) / scale)
	descriptor.Spread = logBucket(statistics.InterquartileRange / scale)
	descriptor.Tau3 = ratioBucket(statistics.LMoments.Tau3)
	descriptor.Tau4 = ratioBucket(statistics.LMoments.Tau4)
	return descriptor, nil
}

// terms enumerates the descriptor's exact retrieval vocabulary — one term
// per schema field, no free text and no width derivation.
func (descriptor CanonicalComponentDescriptor) terms() []string {
	terms := []string{
		"modality:" + descriptor.Modality,
		"role:" + descriptor.Role,
		"space:" + descriptor.Space,
		"dtype:" + descriptor.DType,
		"layout:" + descriptor.Layout,
		"trainrole:" + descriptor.TrainingRole,
	}
	if descriptor.Measured {
		terms = append(terms,
			fmt.Sprintf("scale:%d", descriptor.Scale),
			fmt.Sprintf("location:%d", descriptor.Location),
			fmt.Sprintf("spread:%d", descriptor.Spread),
			fmt.Sprintf("tau3:%d", descriptor.Tau3),
			fmt.Sprintf("tau4:%d", descriptor.Tau4),
		)
	}
	return terms
}

// ExactHit is one ranked exact-retrieval result.
type ExactHit struct {
	Descriptor CanonicalComponentDescriptor
	Overlap    int
	Relevance  float64
}

// ExactComponentIndex is the exact inverted index over canonical
// descriptors: every posting is a term of the fixed schema, retrieval
// touches only the query's own postings — never the whole catalog and
// never tensor bytes — and identical catalogs index identically no matter
// the input order.
type ExactComponentIndex struct {
	descriptors []CanonicalComponentDescriptor
	postings    map[string][]int
}

// NewExactComponentIndex builds the index from catalog components,
// canonicalizing order by model then name so retrieval is a pure function
// of catalog content.
func NewExactComponentIndex(components []CatalogComponent) (*ExactComponentIndex, error) {
	if len(components) == 0 {
		return nil, errors.New("composition: exact index requires components")
	}
	descriptors := make([]CanonicalComponentDescriptor, 0, len(components))
	for _, component := range components {
		descriptor, err := CanonicalDescriptor(component)
		if err != nil {
			return nil, err
		}
		descriptors = append(descriptors, descriptor)
	}
	slices.SortFunc(descriptors, func(a, b CanonicalComponentDescriptor) int {
		if by := strings.Compare(a.Model.String(), b.Model.String()); by != 0 {
			return by
		}
		return strings.Compare(a.Name, b.Name)
	})
	index := &ExactComponentIndex{
		descriptors: descriptors,
		postings:    make(map[string][]int),
	}
	for ordinal, descriptor := range descriptors {
		for _, term := range descriptor.terms() {
			index.postings[term] = append(index.postings[term], ordinal)
		}
	}
	return index, nil
}

// Len reports the indexed component count.
func (index *ExactComponentIndex) Len() int { return len(index.descriptors) }

// Search ranks components by exact term overlap with the query's canonical
// descriptor: overlap descending, then canonical order — fully
// deterministic, touching only the query's postings.
func (index *ExactComponentIndex) Search(query CatalogComponent, limit int) ([]ExactHit, error) {
	if limit <= 0 {
		return nil, errors.New("composition: search limit must be positive")
	}
	descriptor, err := CanonicalDescriptor(query)
	if err != nil {
		return nil, err
	}
	terms := descriptor.terms()
	overlaps := make(map[int]int)
	for _, term := range terms {
		for _, ordinal := range index.postings[term] {
			overlaps[ordinal]++
		}
	}
	ordinals := make([]int, 0, len(overlaps))
	for ordinal := range overlaps {
		ordinals = append(ordinals, ordinal)
	}
	slices.SortFunc(ordinals, func(a, b int) int {
		if overlaps[a] != overlaps[b] {
			return overlaps[b] - overlaps[a]
		}
		return a - b
	})
	if len(ordinals) > limit {
		ordinals = ordinals[:limit]
	}
	hits := make([]ExactHit, 0, len(ordinals))
	for _, ordinal := range ordinals {
		hits = append(hits, ExactHit{
			Descriptor: index.descriptors[ordinal],
			Overlap:    overlaps[ordinal],
			Relevance:  float64(overlaps[ordinal]) / float64(len(terms)),
		})
	}
	return hits, nil
}
