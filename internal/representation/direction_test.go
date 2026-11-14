package representation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestToolCallResidualDirection pins the residual-direction claim contract:
// the document binds every authority the extraction depends on, quantiles
// stay strictly ordered probabilities, and the alpha selector is predeclared
// and complete.
func TestToolCallResidualDirection(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	model := id(artifact.KindModel, "direction model")
	definition := id(artifact.KindModelDefinition, "direction definition")
	layer := uint32(7)
	contract := func(tap TapPoint, layerRef *uint32) Contract {
		value, contractErr := NewContract(Contract{
			Producer: Producer{Model: model, Definition: definition, Tap: tap, Layer: layerRef},
			Modality: ModalityText,
			Tensor: TensorContract{DataType: dtype.F32, Axes: []Axis{
				{Kind: AxisChannel, Bounds: AxisBounds{Extent: 8}},
				{Kind: AxisSequence, Bounds: AxisBounds{Minimum: 1, Maximum: 16}},
			}},
			Sequence: SequenceContract{
				Axis: AxisSequence, Mask: MaskPrefix, Padding: PaddingSuffix,
				Position: PositionSequential, PositionAxes: []AxisKind{AxisSequence},
			},
			Normalization: NormalizationContract{Kind: NormalizationNone, Magnitude: MagnitudeNative},
		})
		if contractErr != nil {
			t.Fatal(contractErr)
		}
		return value
	}
	layerInput := contract(TapLayerInput, &layer)
	selector := AlphaSelector{
		Objective:     "maximize realized inspection-call rate",
		Normalization: "direction unit norm at the contract layer",
		Target:        0.5,
		TieBreak:      "smallest alpha",
	}
	direction := ResidualDirection{
		Contract: layerInput.ID, Model: model, Definition: definition,
		Manuals:     id(artifact.KindProfile, "effect-classed manual catalog"),
		Readout:     id(artifact.KindEvidence, "shared tool-open readout"),
		Corpus:      id(artifact.KindDataset, "extraction corpus"),
		Split:       id(artifact.KindDatasetShard, "extraction split"),
		Quantiles:   []float64{0.25, 0.5, 0.75},
		Extraction:  id(artifact.KindEvidence, "extraction run"),
		Environment: id(artifact.KindEvidence, "extraction environment"),
		Vector:      id(artifact.KindTensorSet, "direction vector"),
		Selector:    selector,
	}
	published := testcheck.Must(NewResidualDirection(direction))(t)
	if !published.ID.Valid() || len(published.Lineage()) != 10 {
		t.Fatalf("direction identity or lineage differs: %s edges=%d", published.ID, len(published.Lineage()))
	}
	refusals := []struct {
		name   string
		mutate func(*ResidualDirection)
	}{
		{"unsorted quantiles", func(value *ResidualDirection) { value.Quantiles = []float64{0.5, 0.25} }},
		{"quantile at one", func(value *ResidualDirection) { value.Quantiles = []float64{1.0} }},
		{"empty quantiles", func(value *ResidualDirection) { value.Quantiles = nil }},
		{"missing selector objective", func(value *ResidualDirection) { value.Selector.Objective = " " }},
		{"negative selector target", func(value *ResidualDirection) { value.Selector.Target = -1 }},
		{"missing readout", func(value *ResidualDirection) { value.Readout = artifact.ID{} }},
		{"missing vector", func(value *ResidualDirection) { value.Vector = artifact.ID{} }},
		{"contract kind", func(value *ResidualDirection) { value.Contract = value.Readout }},
	}
	for _, refusal := range refusals {
		broken := direction
		broken.Quantiles = append([]float64(nil), direction.Quantiles...)
		refusal.mutate(&broken)
		if _, err := NewResidualDirection(broken); err == nil {
			t.Errorf("%s admitted", refusal.name)
		}
	}
}
