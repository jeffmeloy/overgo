package modelrecipe

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/trainingprogram"
)

// PrototypeTrial is the closed-world compilation of one admitted prototype:
// the resolved compiled architecture with its derived tensor-shape draft,
// the exact scratch-construction authority, the objective that can reject
// the hypothesis, the bound promotion evaluation plan, and the recipe
// dependencies the trial rides. Nothing here is generated source — every
// module is a registered Go capability, and an unsupported objective,
// unregistered architecture, or incoherent identity refuses before any
// compute is spent.
type PrototypeTrial struct {
	Prototype        ModelPrototype
	Admission        artifact.ID
	Architecture     model.ArchitectureProfile
	Draft            model.DraftPlan
	Construction     trainingprogram.ScratchConstruction
	Objective        trainingprogram.ObjectiveKind
	EvaluationPlan   artifact.ID
	CandidateRecipes []artifact.ID
}

// CompilePrototypeTrial derives one executable-shaped trial from an admitted
// prototype without executing anything: scratchmodel construction authority,
// the registered architecture's tensor-shape draft, the training objective,
// and the evaluation plan bind through their existing owners.
func CompilePrototypeTrial(
	prototype ModelPrototype,
	admission artifact.ID,
	construction trainingprogram.ScratchSpec,
	objective trainingprogram.ObjectiveKind,
	evaluationPlan artifact.ID,
	heads uint32,
) (PrototypeTrial, error) {
	if err := canonicalizeModelPrototype(&prototype); err != nil {
		return PrototypeTrial{}, err
	}
	if admission.Kind() != artifact.KindEvidence {
		return PrototypeTrial{}, errors.New("model recipe: trial requires the exact admission evidence")
	}
	architecture, registered := model.LookupArchitecture(prototype.Architecture)
	if !registered {
		return PrototypeTrial{}, errors.New("model recipe: trial architecture has no compiled Go registration")
	}
	if !trainingprogram.SupportedObjectiveKind(objective) {
		return PrototypeTrial{}, errors.New("model recipe: trial objective is not a registered capability; refused before compute")
	}
	if heads == 0 {
		return PrototypeTrial{}, errors.New("model recipe: trial tensor draft requires a positive head count")
	}
	if construction.Dataset != prototype.DevelopmentSplit {
		return PrototypeTrial{}, errors.New("model recipe: trial construction must consume the prototype's development split")
	}
	if construction.DerivationProfile != prototype.DerivationPolicy {
		return PrototypeTrial{}, errors.New("model recipe: trial construction must ride the prototype's derivation policy")
	}
	authority, err := trainingprogram.CompileScratchConstruction(construction)
	if err != nil {
		return PrototypeTrial{}, err
	}
	if evaluationPlan.Kind() != artifact.KindEvidence {
		return PrototypeTrial{}, errors.New("model recipe: trial requires a bound promotion evaluation plan")
	}
	recipes := []artifact.ID{construction.Recipe, prototype.TrainingObjective}
	slices.SortFunc(recipes, artifact.CompareID)
	return PrototypeTrial{
		Prototype: prototype, Admission: admission,
		Architecture: architecture, Draft: architecture.DraftPlan(heads),
		Construction: authority, Objective: objective,
		EvaluationPlan: evaluationPlan, CandidateRecipes: slices.Compact(recipes),
	}, nil
}
