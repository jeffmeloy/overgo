package modelartifact

import (
	"errors"
	"math"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/tensorstats"
)

const (
	TensorMeasurementVersion   uint16 = 3
	TensorMeasurementMediaType        = "application/vnd.overgo.tensor-measurement+json"
	TensorMeasurementSchema           = "overgo/tensor-measurement/v3"
)

// Spectral status for a tensor's normalized effective rank. Empty means spectral
// analysis was not requested (policy SpectralMaxDim == 0).
const (
	SpectralComputed      = "computed"       // effective rank is present
	SpectralNotApplicable = "not-applicable" // not a 2-D matrix, or a zero matrix
	SpectralDeferred      = "deferred"       // 2-D but larger than the compute budget
)

var tensorMeasurementContract = artifact.DocumentContract{
	Kind: artifact.KindTensorInventory, MediaType: TensorMeasurementMediaType, Schema: TensorMeasurementSchema,
}

var tensorMeasurementCodec = artifact.JSONDocumentCodec(
	"model artifact tensor measurement", tensorMeasurementContract.Kind,
	tensorMeasurementContract.MediaType, tensorMeasurementContract.Schema,
	canonicalizeTensorMeasurements, func(value TensorMeasurementDocument) artifact.ID { return value.ID },
	func(value *TensorMeasurementDocument, id artifact.ID) { value.ID = id }, func(value TensorMeasurementDocument) TensorMeasurementDocument {
		value.Measurements = slices.Clone(value.Measurements)
		return value
	},
)

type MeasurementPolicy struct {
	MaxSamplesPerTensor uint64 `json:"max_samples_per_tensor"`
	MaxReadBytes        uint64 `json:"max_read_bytes"`
	// SpectralMaxDim enables normalized effective rank for 2-D tensors whose
	// dimensions are both within it (0 disables spectral analysis). The bound is
	// a compute budget — exact singular values are O(dim³) — not a metric
	// threshold; larger matrices are reported deferred.
	SpectralMaxDim uint64 `json:"spectral_max_dim,omitempty"`
}

// TensorMeasurement is one tensor's distribution-free characterization: robust
// L-moments, order statistics, and energy descriptors over a bounded sample.
// The statistics come from the shared tensorstats package; only Name is local.
type TensorMeasurement struct {
	Name string `json:"name"`
	tensorstats.Characterization
	// EffectiveRank is the normalized effective rank (spectral flatness) of a
	// 2-D tensor, present only when SpectralStatus is "computed".
	EffectiveRank  float64 `json:"effective_rank,omitempty"`
	SpectralStatus string  `json:"spectral_status,omitempty"`
}

// TensorMeasurementDocument: bounded sampled tensor evidence.
type TensorMeasurementDocument struct {
	Version      uint16              `json:"version"`
	Inventory    artifact.ID         `json:"inventory"`
	Policy       MeasurementPolicy   `json:"policy"`
	ReadBytes    uint64              `json:"read_bytes"`
	Measurements []TensorMeasurement `json:"measurements"`
	ID           artifact.ID         `json:"-"`
}

func newTensorMeasurementDocument(
	inventory artifact.ID,
	policy MeasurementPolicy,
	readBytes uint64,
	measurements []TensorMeasurement,
) (TensorMeasurementDocument, error) {
	document := TensorMeasurementDocument{
		Version: TensorMeasurementVersion, Inventory: inventory, Policy: policy,
		ReadBytes: readBytes, Measurements: slices.Clone(measurements),
	}
	return tensorMeasurementCodec.New(document)
}

// ParseTensorMeasurementDocument decodes one committed measurement document:
// the store-catalog read path for cross-model component retrieval.
func ParseTensorMeasurementDocument(content []byte) (TensorMeasurementDocument, error) {
	return tensorMeasurementCodec.Parse(content)
}

func (d TensorMeasurementDocument) ValidateIdentity() error {
	return tensorMeasurementCodec.ValidateIdentity(d)
}

func (d TensorMeasurementDocument) Content() (artifact.Content, error) {
	return tensorMeasurementCodec.Content(d)
}

func (d TensorMeasurementDocument) Lineage() artifact.Lineage {
	return artifact.Lineage{Child: d.ID, Parent: d.Inventory, Relation: artifact.RelationDerivedFrom}
}

func (d TensorMeasurementDocument) Batch(key string) (artifact.Batch, error) {
	return tensorMeasurementCodec.Batch(key, d, []artifact.Lineage{d.Lineage()}, nil)
}

func measurementFromSamples(name string, elements uint64, samples []float64) (TensorMeasurement, error) {
	characterization, ok := tensorstats.Characterize(elements, samples)
	if !ok {
		return TensorMeasurement{}, errors.New("model artifact: tensor sample has no finite values")
	}
	return TensorMeasurement{Name: name, Characterization: characterization}, nil
}

func canonicalizeTensorMeasurements(document *TensorMeasurementDocument) error {
	if document == nil || document.Version != TensorMeasurementVersion ||
		document.Inventory.Kind() != artifact.KindTensorInventory ||
		document.Policy.MaxSamplesPerTensor == 0 ||
		document.Policy.MaxReadBytes == 0 || document.ReadBytes == 0 ||
		document.ReadBytes > document.Policy.MaxReadBytes || len(document.Measurements) == 0 ||
		len(document.Measurements) > maxInventoryTensors {
		return errors.New("model artifact: invalid tensor measurement envelope")
	}
	sort.Slice(document.Measurements, func(i, j int) bool {
		return document.Measurements[i].Name < document.Measurements[j].Name
	})
	for index, measurement := range document.Measurements {
		c := measurement.Characterization
		if measurement.Name == "" || strings.TrimSpace(measurement.Name) != measurement.Name ||
			c.Elements == 0 || c.Samples == 0 || c.Samples > c.Elements ||
			c.Samples > document.Policy.MaxSamplesPerTensor ||
			c.FiniteSamples == 0 || c.FiniteSamples > c.Samples ||
			!finiteMeasurement(c.LowerQuartile) || !finiteMeasurement(c.Median) ||
			!finiteMeasurement(c.UpperQuartile) || c.InterquartileRange < 0 ||
			c.LowerQuartile > c.Median || c.Median > c.UpperQuartile ||
			!validSpectralMeasurement(measurement) ||
			index > 0 && document.Measurements[index-1].Name == measurement.Name {
			return errors.New("model artifact: invalid tensor measurement")
		}
		// A nonzero spectral policy demands an explicit verdict per tensor:
		// an empty status would let a skipped spectral pass masquerade as a
		// measured document (the 2026-08-16 audit's evidence-integrity P1).
		if document.Policy.SpectralMaxDim > 0 && measurement.SpectralStatus == "" {
			return errors.New("model artifact: spectral policy is set but a measurement carries no spectral status")
		}
	}
	return nil
}

// validSpectralMeasurement checks the optional effective-rank fields: it is
// present and in (0,1] exactly when the status is "computed", and absent
// otherwise.
func validSpectralMeasurement(m TensorMeasurement) bool {
	switch m.SpectralStatus {
	case "", SpectralNotApplicable, SpectralDeferred:
		return m.EffectiveRank == 0
	case SpectralComputed:
		return finiteMeasurement(m.EffectiveRank) && m.EffectiveRank > 0 && m.EffectiveRank <= 1
	default:
		return false
	}
}

func finiteMeasurement(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
