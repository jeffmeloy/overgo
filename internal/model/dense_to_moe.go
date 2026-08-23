package model

import (
	"cmp"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/tensor"
)

const (
	// DenseToMoERouterEvidenceVersion is the immutable calibration version.
	DenseToMoERouterEvidenceVersion = artifact.InitialDocumentVersion
	// DenseToMoERouterEvidenceMediaType identifies router calibration evidence.
	DenseToMoERouterEvidenceMediaType = "application/vnd.overgo.dense-to-moe-router-evidence+json"
	// DenseToMoERouterEvidenceSchema identifies router calibration evidence.
	DenseToMoERouterEvidenceSchema = "overgo/dense-to-moe-router-evidence/v1"
	// DenseToMoERouterPlanVersion is the immutable derived-router version.
	DenseToMoERouterPlanVersion = artifact.InitialDocumentVersion
	// DenseToMoERouterPlanMediaType identifies derived router plans.
	DenseToMoERouterPlanMediaType = "application/vnd.overgo.dense-to-moe-router-plan+json"
	// DenseToMoERouterPlanSchema identifies derived router plans.
	DenseToMoERouterPlanSchema = "overgo/dense-to-moe-router-plan/v1"
)

// DenseToMoERouterSample is one layer activation with exact relevant experts.
type DenseToMoERouterSample struct {
	Layer   uint32        `json:"layer"`
	Hidden  []float32     `json:"hidden"`
	Experts []artifact.ID `json:"experts"`
}

// DenseToMoERouterEvidence separates derivation and held-out screening data.
type DenseToMoERouterEvidence struct {
	Version         uint16                   `json:"version"`
	TrainingDataset artifact.ID              `json:"training_dataset"`
	HeldOutDataset  artifact.ID              `json:"held_out_dataset"`
	TrainingRun     artifact.ID              `json:"training_run"`
	HeldOutRun      artifact.ID              `json:"held_out_run"`
	Experts         []artifact.ID            `json:"experts"`
	Training        []DenseToMoERouterSample `json:"training"`
	HeldOut         []DenseToMoERouterSample `json:"held_out"`
	ID              artifact.ID              `json:"-"`
}

// DenseToMoERouterLayer contains expert-major, contiguous router rows.
type DenseToMoERouterLayer struct {
	Layer   uint32    `json:"layer"`
	Weights []float32 `json:"weights"`
}

// DenseToMoERouterPlan is the router admitted by held-out evidence.
type DenseToMoERouterPlan struct {
	Version  uint16                  `json:"version"`
	Evidence artifact.ID             `json:"evidence"`
	Experts  []artifact.ID           `json:"experts"`
	TopK     uint32                  `json:"top_k"`
	Width    uint32                  `json:"width"`
	Layers   []DenseToMoERouterLayer `json:"layers"`
	ID       artifact.ID             `json:"-"`
}

var denseToMoERouterEvidenceCodec = artifact.JSONDocumentCodec(
	"dense-to-MoE router evidence", artifact.KindEvidence,
	DenseToMoERouterEvidenceMediaType, DenseToMoERouterEvidenceSchema,
	canonicalizeDenseToMoERouterEvidence,
	func(value DenseToMoERouterEvidence) artifact.ID { return value.ID },
	func(value *DenseToMoERouterEvidence, id artifact.ID) { value.ID = id },
	cloneDenseToMoERouterEvidence,
)

var denseToMoERouterPlanCodec = artifact.JSONDocumentCodec(
	"dense-to-MoE router plan", artifact.KindProfile,
	DenseToMoERouterPlanMediaType, DenseToMoERouterPlanSchema,
	canonicalizeDenseToMoERouterPlan,
	func(value DenseToMoERouterPlan) artifact.ID { return value.ID },
	func(value *DenseToMoERouterPlan, id artifact.ID) { value.ID = id },
	cloneDenseToMoERouterPlan,
)

// NewDenseToMoERouterEvidence seals activation evidence and split lineage.
func NewDenseToMoERouterEvidence(value DenseToMoERouterEvidence) (DenseToMoERouterEvidence, error) {
	value.Version, value.ID = DenseToMoERouterEvidenceVersion, artifact.ID{}
	return denseToMoERouterEvidenceCodec.New(value)
}

// CompileDenseToMoERouter derives centered, normalized expert centroids and
// admits them only when every held-out label wins with a strict routing margin.
func CompileDenseToMoERouter(evidence DenseToMoERouterEvidence) (DenseToMoERouterPlan, error) {
	if err := evidence.ValidateIdentity(); err != nil {
		return DenseToMoERouterPlan{}, err
	}
	topK := uint32(len(evidence.Training[0].Experts))
	width := uint32(len(evidence.Training[0].Hidden))
	layers := denseToMoELayerCount(evidence.Training, evidence.HeldOut)
	plan := DenseToMoERouterPlan{
		Version: DenseToMoERouterPlanVersion, Evidence: evidence.ID,
		Experts: slices.Clone(evidence.Experts), TopK: topK, Width: width,
		Layers: make([]DenseToMoERouterLayer, layers),
	}
	for layer := range layers {
		weights, err := deriveDenseToMoELayer(
			uint32(layer), width, evidence.Experts, evidence.Training,
		)
		if err != nil {
			return DenseToMoERouterPlan{}, err
		}
		plan.Layers[layer] = DenseToMoERouterLayer{Layer: uint32(layer), Weights: weights}
	}
	if err := screenDenseToMoERouter(plan, evidence.HeldOut); err != nil {
		return DenseToMoERouterPlan{}, err
	}
	return denseToMoERouterPlanCodec.New(plan)
}

// ValidateIdentity verifies exact router evidence identity.
func (value DenseToMoERouterEvidence) ValidateIdentity() error {
	return denseToMoERouterEvidenceCodec.ValidateIdentity(value)
}

// Content returns exact router evidence content.
func (value DenseToMoERouterEvidence) Content() (artifact.Content, error) {
	return denseToMoERouterEvidenceCodec.Content(value)
}

// Lineage binds calibration to datasets, runs, and expert models.
func (value DenseToMoERouterEvidence) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.TrainingDataset, value.HeldOutDataset, value.TrainingRun, value.HeldOutRun}
	parents = append(parents, value.Experts...)
	return artifact.DependencyLineage(value.ID, parents...)
}

// Batch prepares publication of router evidence.
func (value DenseToMoERouterEvidence) Batch(key string) (artifact.Batch, error) {
	return denseToMoERouterEvidenceCodec.Batch(key, value, value.Lineage(), nil)
}

// ValidateIdentity verifies exact derived router identity.
func (value DenseToMoERouterPlan) ValidateIdentity() error {
	return denseToMoERouterPlanCodec.ValidateIdentity(value)
}

// Content returns exact derived router content.
func (value DenseToMoERouterPlan) Content() (artifact.Content, error) {
	return denseToMoERouterPlanCodec.Content(value)
}

// Lineage binds the router plan to its calibration evidence and experts.
func (value DenseToMoERouterPlan) Lineage() []artifact.Lineage {
	parents := append([]artifact.ID{value.Evidence}, value.Experts...)
	return artifact.DependencyLineage(value.ID, parents...)
}

// Batch prepares publication of a derived router plan.
func (value DenseToMoERouterPlan) Batch(key string) (artifact.Batch, error) {
	return denseToMoERouterPlanCodec.Batch(key, value, value.Lineage(), nil)
}

func canonicalizeDenseToMoERouterEvidence(value *DenseToMoERouterEvidence) error {
	if value == nil || value.Version != DenseToMoERouterPlanVersion ||
		value.TrainingDataset.Kind() != artifact.KindDataset || value.HeldOutDataset.Kind() != artifact.KindDataset ||
		value.TrainingDataset == value.HeldOutDataset || value.TrainingRun.Kind() != artifact.KindRun ||
		value.HeldOutRun.Kind() != artifact.KindRun || value.TrainingRun == value.HeldOutRun ||
		len(value.Experts) < tensor.PairedExtent || uint64(len(value.Experts)) > math.MaxUint32 ||
		len(value.Training) == 0 || len(value.HeldOut) == 0 {
		return errors.New("model: invalid dense-to-MoE router evidence")
	}
	expertOrder := make(map[artifact.ID]int, len(value.Experts))
	for index, expert := range value.Experts {
		if expert.Kind() != artifact.KindModel {
			return errors.New("model: dense-to-MoE expert identity is invalid")
		}
		if _, duplicate := expertOrder[expert]; duplicate {
			return errors.New("model: dense-to-MoE expert identity is duplicated")
		}
		expertOrder[expert] = index
	}
	width, topK := len(value.Training[0].Hidden), len(value.Training[0].Experts)
	if width == 0 || uint64(width) > math.MaxUint32 || topK == 0 || topK >= len(value.Experts) {
		return errors.New("model: dense-to-MoE router dimensions are invalid")
	}
	for _, samples := range [][]DenseToMoERouterSample{value.Training, value.HeldOut} {
		for index := range samples {
			sample := &samples[index]
			if sample.Layer == math.MaxUint32 || len(sample.Hidden) != width || len(sample.Experts) != topK {
				return errors.New("model: dense-to-MoE sample geometry differs")
			}
			for _, component := range sample.Hidden {
				if !checked.Finite32(component) {
					return errors.New("model: dense-to-MoE sample contains a non-finite activation")
				}
			}
			sort.Slice(sample.Experts, func(left, right int) bool {
				return expertOrder[sample.Experts[left]] < expertOrder[sample.Experts[right]]
			})
			for expertIndex, expert := range sample.Experts {
				if _, found := expertOrder[expert]; !found ||
					expertIndex > 0 && expert == sample.Experts[expertIndex-1] {
					return errors.New("model: dense-to-MoE sample expert label is invalid")
				}
			}
		}
		sortDenseToMoERouterSamples(samples, expertOrder)
	}
	layers := denseToMoELayerCount(value.Training, value.HeldOut)
	for layer := range layers {
		for _, samples := range [][]DenseToMoERouterSample{value.Training, value.HeldOut} {
			if !slices.ContainsFunc(samples, func(sample DenseToMoERouterSample) bool {
				return sample.Layer == uint32(layer)
			}) {
				return errors.New("model: dense-to-MoE calibration layer coverage differs")
			}
		}
		for _, expert := range value.Experts {
			if !slices.ContainsFunc(value.Training, func(sample DenseToMoERouterSample) bool {
				return sample.Layer == uint32(layer) && slices.Contains(sample.Experts, expert)
			}) {
				return errors.New("model: dense-to-MoE expert lacks training coverage")
			}
		}
	}
	return nil
}

func canonicalizeDenseToMoERouterPlan(value *DenseToMoERouterPlan) error {
	if value == nil || value.Version != DenseToMoERouterEvidenceVersion ||
		value.Evidence.Kind() != artifact.KindEvidence || len(value.Experts) < tensor.PairedExtent ||
		uint64(len(value.Experts)) > math.MaxUint32 ||
		value.TopK == 0 || value.TopK >= uint32(len(value.Experts)) || value.Width == 0 || len(value.Layers) == 0 {
		return errors.New("model: invalid dense-to-MoE router plan")
	}
	seenExperts := make(map[artifact.ID]bool, len(value.Experts))
	for _, expert := range value.Experts {
		if expert.Kind() != artifact.KindModel || seenExperts[expert] {
			return errors.New("model: dense-to-MoE router plan expert identity is invalid")
		}
		seenExperts[expert] = true
	}
	wantWeights, ok := checked.MulInt(len(value.Experts), int(value.Width))
	if !ok {
		return errors.New("model: dense-to-MoE router extent overflows")
	}
	for index, layer := range value.Layers {
		if layer.Layer != uint32(index) || len(layer.Weights) != wantWeights {
			return errors.New("model: dense-to-MoE router layer geometry differs")
		}
		for _, weight := range layer.Weights {
			if !checked.Finite32(weight) {
				return errors.New("model: dense-to-MoE router contains a non-finite weight")
			}
		}
	}
	return nil
}

func deriveDenseToMoELayer(
	layer, width uint32,
	experts []artifact.ID,
	samples []DenseToMoERouterSample,
) ([]float32, error) {
	widthInt := int(width)
	centroids := make([]float64, len(experts)*widthInt)
	counts := make([]uint64, len(experts))
	for _, sample := range samples {
		if sample.Layer != layer {
			continue
		}
		for _, relevant := range sample.Experts {
			expert := slices.Index(experts, relevant)
			counts[expert]++
			for dimension, component := range sample.Hidden {
				centroids[expert*widthInt+dimension] += float64(component)
			}
		}
	}
	mean := make([]float64, widthInt)
	for expert, count := range counts {
		if count == 0 {
			return nil, errors.New("model: dense-to-MoE router expert is unobserved")
		}
		for dimension := range widthInt {
			position := expert*widthInt + dimension
			centroids[position] /= float64(count)
			mean[dimension] += centroids[position] / float64(len(experts))
		}
	}
	weights := make([]float32, len(centroids))
	for expert := range experts {
		var squaredNorm float64
		for dimension := range widthInt {
			position := expert*widthInt + dimension
			centroids[position] -= mean[dimension]
			squaredNorm += centroids[position] * centroids[position]
		}
		if squaredNorm == 0 || math.IsNaN(squaredNorm) || math.IsInf(squaredNorm, 0) {
			return nil, errors.New("model: dense-to-MoE router expert centroid is degenerate")
		}
		norm := math.Sqrt(squaredNorm)
		for dimension := range widthInt {
			position := expert*widthInt + dimension
			weights[position] = float32(centroids[position] / norm)
		}
	}
	return weights, nil
}

func screenDenseToMoERouter(plan DenseToMoERouterPlan, samples []DenseToMoERouterSample) error {
	width := int(plan.Width)
	for _, sample := range samples {
		layer := plan.Layers[sample.Layer]
		type score struct {
			expert artifact.ID
			value  float64
		}
		scores := make([]score, len(plan.Experts))
		for expert, id := range plan.Experts {
			var value float64
			for dimension, hidden := range sample.Hidden {
				value += float64(layer.Weights[expert*width+dimension]) * float64(hidden)
			}
			scores[expert] = score{expert: id, value: value}
		}
		sort.Slice(scores, func(left, right int) bool {
			if scores[left].value != scores[right].value {
				return scores[left].value > scores[right].value
			}
			return scores[left].expert.String() < scores[right].expert.String()
		})
		boundary := int(plan.TopK)
		if scores[boundary-1].value == scores[boundary].value {
			return errors.New("model: dense-to-MoE held-out router margin is tied")
		}
		selected := make([]artifact.ID, boundary)
		for index := range boundary {
			selected[index] = scores[index].expert
		}
		slices.SortFunc(selected, func(left, right artifact.ID) int {
			return cmp.Compare(slices.Index(plan.Experts, left), slices.Index(plan.Experts, right))
		})
		if !slices.Equal(selected, sample.Experts) {
			return fmt.Errorf("model: dense-to-MoE held-out routing fails at layer %d", sample.Layer)
		}
	}
	return nil
}

func denseToMoELayerCount(groups ...[]DenseToMoERouterSample) int {
	var count uint32
	for _, samples := range groups {
		for _, sample := range samples {
			count = max(count, sample.Layer+uint32(tensor.SingletonExtent))
		}
	}
	return int(count)
}

func sortDenseToMoERouterSamples(samples []DenseToMoERouterSample, expertOrder map[artifact.ID]int) {
	slices.SortFunc(samples, func(left, right DenseToMoERouterSample) int {
		if order := cmp.Compare(left.Layer, right.Layer); order != 0 {
			return order
		}
		for index := range left.Experts {
			if order := cmp.Compare(expertOrder[left.Experts[index]], expertOrder[right.Experts[index]]); order != 0 {
				return order
			}
		}
		for index := range left.Hidden {
			if order := cmp.Compare(left.Hidden[index], right.Hidden[index]); order != 0 {
				return order
			}
		}
		return 0
	})
}

func cloneDenseToMoERouterEvidence(value DenseToMoERouterEvidence) DenseToMoERouterEvidence {
	value.Experts = slices.Clone(value.Experts)
	value.Training = cloneDenseToMoESamples(value.Training)
	value.HeldOut = cloneDenseToMoESamples(value.HeldOut)
	return value
}

func cloneDenseToMoESamples(samples []DenseToMoERouterSample) []DenseToMoERouterSample {
	result := make([]DenseToMoERouterSample, len(samples))
	for index, sample := range samples {
		result[index] = sample
		result[index].Hidden = slices.Clone(sample.Hidden)
		result[index].Experts = slices.Clone(sample.Experts)
	}
	return result
}

func cloneDenseToMoERouterPlan(value DenseToMoERouterPlan) DenseToMoERouterPlan {
	value.Experts = slices.Clone(value.Experts)
	value.Layers = slices.Clone(value.Layers)
	for index := range value.Layers {
		value.Layers[index].Weights = slices.Clone(value.Layers[index].Weights)
	}
	return value
}
