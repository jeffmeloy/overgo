package representation

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

const (
	// ResidualDirectionMediaType identifies one extracted residual direction.
	ResidualDirectionMediaType = "application/vnd.overgo.residual-direction+json"
	// ResidualDirectionSchema identifies the residual-direction contract.
	ResidualDirectionSchema = "overgo/residual-direction/v1"
)

// AlphaSelector is the predeclared objective-constrained rule that derives
// the intervention strength. Alpha is never a request knob: activation
// derives it from this selector over admitted evidence, with the
// normalization, target, and tie-break fixed before any measurement exists.
type AlphaSelector struct {
	Objective     string  `json:"objective"`
	Normalization string  `json:"normalization"`
	Target        float64 `json:"target"`
	TieBreak      string  `json:"tie_break"`
}

// Validate checks the selector is complete and finite.
func (selector AlphaSelector) Validate() error {
	if strings.TrimSpace(selector.Objective) == "" || strings.TrimSpace(selector.Normalization) == "" ||
		strings.TrimSpace(selector.TieBreak) == "" || !checked.Finite64(selector.Target) ||
		selector.Target < 0 {
		return errors.New("representation: invalid alpha selector")
	}
	return nil
}

// ResidualDirection is one content-addressed extracted direction in residual
// space: the training-free but distribution- and harness-dependent hypothesis
// that adding this vector at the contracted layer input changes tool-call
// propensity. It binds every authority the claim depends on -- the exact
// TapLayerInput contract, model and definition, the effect-classed manual
// catalog, the shared tool-open readout, corpus and split, quantiles, the
// extraction run, and the environment -- and owns no recipe or activation
// path.
type ResidualDirection struct {
	Version     uint16        `json:"version"`
	Contract    artifact.ID   `json:"contract"`
	Model       artifact.ID   `json:"model"`
	Definition  artifact.ID   `json:"definition"`
	Manuals     artifact.ID   `json:"manuals"`
	Readout     artifact.ID   `json:"readout"`
	Corpus      artifact.ID   `json:"corpus"`
	Split       artifact.ID   `json:"split"`
	Quantiles   []float64     `json:"quantiles"`
	Extraction  artifact.ID   `json:"extraction"`
	Environment artifact.ID   `json:"environment"`
	Vector      artifact.ID   `json:"vector"`
	Selector    AlphaSelector `json:"selector"`
	ID          artifact.ID   `json:"-"`
}

var residualDirectionCodec = artifact.JSONDocumentCodec(
	"residual direction", artifact.KindRecipe, ResidualDirectionMediaType, ResidualDirectionSchema,
	canonicalizeResidualDirection,
	func(value ResidualDirection) artifact.ID { return value.ID },
	func(value *ResidualDirection, id artifact.ID) { value.ID = id },
	func(value ResidualDirection) ResidualDirection {
		value.Quantiles = slices.Clone(value.Quantiles)
		return value
	},
)

// NewResidualDirection canonicalizes and identifies one direction claim.
func NewResidualDirection(value ResidualDirection) (ResidualDirection, error) {
	return residualDirectionCodec.NewInitial(value)
}

// ParseResidualDirection decodes one canonical direction document.
func ParseResidualDirection(data []byte) (ResidualDirection, error) {
	return residualDirectionCodec.Parse(data)
}

// Content returns the immutable direction document.
func (value ResidualDirection) Content() (artifact.Content, error) {
	return residualDirectionCodec.Content(value)
}

// Lineage binds the direction to every authority its claim depends on.
func (value ResidualDirection) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID,
		value.Contract, value.Model, value.Definition, value.Manuals, value.Readout,
		value.Corpus, value.Split, value.Extraction, value.Environment, value.Vector,
	)
}

// RequireResidualDirection loads the exact direction and refuses one whose
// representation contract is not a layer-input tap on the same model: a
// direction extracted at any other boundary cannot claim residual authority.
func RequireResidualDirection(ctx context.Context, reader artifact.Reader, id artifact.ID) (ResidualDirection, error) {
	direction, err := residualDirectionCodec.RequireExactLineage(ctx, reader, id, ResidualDirection.Lineage)
	if err != nil {
		return ResidualDirection{}, err
	}
	contract, err := LoadContract(ctx, reader, direction.Contract)
	if err != nil {
		return ResidualDirection{}, err
	}
	if contract.Producer.Tap != TapLayerInput || contract.Producer.Layer == nil ||
		contract.Producer.Model != direction.Model {
		return ResidualDirection{}, errors.New("representation: residual direction contract is not a layer-input tap on the direction's model")
	}
	return direction, nil
}

func canonicalizeResidualDirection(value *ResidualDirection) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Contract.Kind() != artifact.KindProfile ||
		value.Model.Kind() != artifact.KindModel ||
		value.Definition.Kind() != artifact.KindModelDefinition ||
		!value.Manuals.Valid() || !value.Readout.Valid() ||
		!value.Corpus.Valid() || !value.Split.Valid() ||
		!value.Extraction.Valid() || !value.Environment.Valid() || !value.Vector.Valid() {
		return errors.New("representation: invalid residual direction binding")
	}
	if err := value.Selector.Validate(); err != nil {
		return err
	}
	if len(value.Quantiles) == 0 {
		return errors.New("representation: residual direction requires extraction quantiles")
	}
	previous := 0.0
	for _, quantile := range value.Quantiles {
		if !checked.Finite64(quantile) || quantile <= previous || quantile >= 1 {
			return errors.New("representation: residual direction quantiles must ascend strictly within (0, 1)")
		}
		previous = quantile
	}
	value.Quantiles = slices.Clone(value.Quantiles)
	return nil
}
