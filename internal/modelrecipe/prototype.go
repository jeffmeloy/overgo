package modelrecipe

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/textcheck"
)

const (
	// ModelPrototypeMediaType identifies model prototype documents.
	ModelPrototypeMediaType = "application/vnd.overgo.model-prototype+json"
	// ModelPrototypeSchema identifies the model prototype contract.
	ModelPrototypeSchema = "overgo/model-prototype/v1"
	// prototypeTextBytes bounds every free-text hypothesis field.
	prototypeTextBytes = 2048
)

// PrototypeHypothesis states what the prototype is for and how it dies: the
// capability it targets, the predicted improvement, and the falsifiable
// check that can reject it.
type PrototypeHypothesis struct {
	Capability string `json:"capability"`
	Prediction string `json:"prediction"`
	Falsifier  string `json:"falsifier"`
}

// PrototypeBudget is the resource ceiling a trial may not exceed.
type PrototypeBudget struct {
	GPUMinutes uint64 `json:"gpu_minutes"`
	HostRAMGiB uint32 `json:"host_ram_gib"`
	VRAMGiB    uint32 `json:"vram_gib"`
	MaxWallNS  uint64 `json:"max_wall_ns"`
}

// ModelPrototype is a content-addressed hypothesis, not executable code,
// weights, or an active recipe. Its architecture name must resolve to a
// compiled Go registration, every policy and evidence dependency is an exact
// artifact identity, its promotion split is independent of its development
// split, and its code identity is evidence only — holding a prototype
// document authorizes nothing to execute.
type ModelPrototype struct {
	Version             uint16              `json:"version"`
	Architecture        string              `json:"architecture"`
	Parent              artifact.ID         `json:"parent,omitzero"`
	ArchitectureProfile artifact.ID         `json:"architecture_profile"`
	DerivationPolicy    artifact.ID         `json:"derivation_policy"`
	TrainingObjective   artifact.ID         `json:"training_objective"`
	DevelopmentSplit    artifact.ID         `json:"development_split"`
	PromotionSplit      artifact.ID         `json:"promotion_split"`
	Hypothesis          PrototypeHypothesis `json:"hypothesis"`
	Budget              PrototypeBudget     `json:"budget"`
	CodeEvidence        artifact.ID         `json:"code_evidence"`
	ID                  artifact.ID         `json:"-"`
}

var modelPrototypeCodec = artifact.JSONDocumentCodec(
	"model prototype", artifact.KindProfile, ModelPrototypeMediaType, ModelPrototypeSchema,
	canonicalizeModelPrototype,
	func(value ModelPrototype) artifact.ID { return value.ID },
	func(value *ModelPrototype, id artifact.ID) { value.ID = id }, nil,
)

func validPrototypeText(value string) bool {
	return value != "" && textcheck.Bounded(value, prototypeTextBytes, "\x00")
}

func canonicalizeModelPrototype(value *ModelPrototype) error {
	if value.Version != artifact.InitialDocumentVersion {
		return errors.New("model recipe: invalid prototype version")
	}
	if _, registered := model.LookupArchitecture(value.Architecture); !registered {
		return errors.New("model recipe: prototype architecture has no compiled Go registration")
	}
	if value.Parent.Valid() && value.Parent.Kind() != artifact.KindModel {
		return errors.New("model recipe: prototype parent must be a model identity")
	}
	if value.ArchitectureProfile.Kind() != artifact.KindProfile ||
		value.DerivationPolicy.Kind() != artifact.KindProfile {
		return errors.New("model recipe: prototype profile and derivation policy must be exact profile identities")
	}
	if value.TrainingObjective.Kind() != artifact.KindRecipe {
		return errors.New("model recipe: prototype training objective must be an exact recipe identity")
	}
	if value.DevelopmentSplit.Kind() != artifact.KindDataset || value.PromotionSplit.Kind() != artifact.KindDataset {
		return errors.New("model recipe: prototype splits must be exact dataset identities")
	}
	if value.DevelopmentSplit == value.PromotionSplit {
		return errors.New("model recipe: promotion split must be independent of the development split")
	}
	if !validPrototypeText(value.Hypothesis.Capability) || !validPrototypeText(value.Hypothesis.Prediction) ||
		!validPrototypeText(value.Hypothesis.Falsifier) {
		return errors.New("model recipe: prototype hypothesis requires capability, prediction, and falsifier")
	}
	if value.Budget.GPUMinutes == 0 || value.Budget.MaxWallNS == 0 {
		return errors.New("model recipe: prototype budget requires positive compute and wall ceilings")
	}
	if value.CodeEvidence.Kind() != artifact.KindEvidence {
		return errors.New("model recipe: prototype code identity must be exact evidence; it cannot authorize execution")
	}
	return nil
}

// NewModelPrototype canonicalizes and identifies one immutable prototype.
func NewModelPrototype(value ModelPrototype) (ModelPrototype, error) {
	return modelPrototypeCodec.NewInitial(value)
}

// ParseModelPrototype decodes one canonical prototype document and proves
// its content identity; the admission row extends this surface when it
// consumes stored prototypes.
func ParseModelPrototype(content []byte) (ModelPrototype, error) {
	return modelPrototypeCodec.Parse(content)
}

// Lineage binds the prototype to every exact dependency it cites.
func (value ModelPrototype) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.Parent, value.ArchitectureProfile, value.DerivationPolicy,
		value.TrainingObjective, value.DevelopmentSplit, value.PromotionSplit, value.CodeEvidence,
	}
	valid := parents[:0]
	for _, parent := range parents {
		if parent.Valid() {
			valid = append(valid, parent)
		}
	}
	return artifact.DependencyLineage(value.ID, valid...)
}

// Batch wraps the prototype as one committable store batch.
func (value ModelPrototype) Batch(key string) (artifact.Batch, error) {
	return modelPrototypeCodec.Batch(key, value, value.Lineage(), nil)
}
