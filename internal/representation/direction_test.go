package representation

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testutil"
)

// TestToolCallResidualDirection pins the residual-direction claim contract:
// the document binds every authority the extraction depends on, quantiles
// stay strictly ordered probabilities, the alpha selector is predeclared and
// complete, and the loading path refuses any direction whose representation
// contract is not a layer-input tap on the direction's own model.
func TestToolCallResidualDirection(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
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
	embedding := contract(TapEmbeddingOutput, nil)
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
	published, err := NewResidualDirection(direction)
	if err != nil {
		t.Fatal(err)
	}
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

	commit := func(contracts ...Contract) {
		t.Helper()
		contents := make([]artifact.Content, 0, len(contracts)+1)
		for _, value := range contracts {
			content, contentErr := value.Content()
			if contentErr != nil {
				t.Fatal(contentErr)
			}
			contents = append(contents, content)
		}
		directionContent, contentErr := published.Content()
		if contentErr != nil {
			t.Fatal(contentErr)
		}
		contents = append(contents, directionContent)
		parents := []artifact.Descriptor{
			{ID: model}, {ID: definition}, {ID: published.Manuals}, {ID: published.Readout},
			{ID: published.Corpus}, {ID: published.Split}, {ID: published.Extraction},
			{ID: published.Environment}, {ID: published.Vector, Size: 32},
		}
		if _, err := store.Commit(ctx, artifact.Batch{
			Key: "fixture/residual-direction", Artifacts: parents,
			Contents: contents, Lineage: published.Lineage(),
		}); err != nil {
			t.Fatal(err)
		}
	}
	commit(layerInput, embedding)
	loaded, err := RequireResidualDirection(ctx, store, published.ID)
	if err != nil || loaded.ID != published.ID || loaded.Selector != selector {
		t.Fatalf("loaded direction = %+v, %v", loaded, err)
	}

	wrongTap := direction
	wrongTap.Contract = embedding.ID
	wrongPublished, err := NewResidualDirection(wrongTap)
	if err != nil {
		t.Fatal(err)
	}
	wrongContent, err := wrongPublished.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:      "fixture/residual-direction-wrong-tap",
		Contents: []artifact.Content{wrongContent}, Lineage: wrongPublished.Lineage(),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := RequireResidualDirection(ctx, store, wrongPublished.ID); err == nil ||
		!strings.Contains(err.Error(), "layer-input") {
		t.Fatalf("embedding-tap direction admitted: %v", err)
	}
}
