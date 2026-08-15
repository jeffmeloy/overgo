package trainingprogram

import (
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
)

// ImprovementKind: controller proposal class.
type ImprovementKind string

const (
	ImprovementCorpus               ImprovementKind = "corpus"
	ImprovementRecipe               ImprovementKind = "recipe"
	ImprovementEvaluator            ImprovementKind = "evaluator"
	ImprovementComponentComposition ImprovementKind = "component-composition"
)

type ImprovementSpec struct {
	Kind             ImprovementKind
	ParentModel      artifact.ID
	Candidate        artifact.ID
	Dataset          artifact.ID
	DevelopmentSplit artifact.ID
	Recipe           artifact.ID
	Code             artifact.ID
	Proposer         artifact.ID
	Components       []artifact.ID
}

// ImprovementProposal: immutable, non-authorizing descendant proposal.
type ImprovementProposal struct {
	id               artifact.ID
	kind             ImprovementKind
	parentModel      artifact.ID
	candidate        artifact.ID
	dataset          artifact.ID
	developmentSplit artifact.ID
	recipe           artifact.ID
	code             artifact.ID
	proposer         artifact.ID
	components       []artifact.ID
}

func CompileImprovementProposal(spec ImprovementSpec) (ImprovementProposal, error) {
	if spec.ParentModel.Kind() != artifact.KindModel || spec.Dataset.Kind() != artifact.KindDataset ||
		spec.DevelopmentSplit.Kind() != artifact.KindDatasetShard || spec.Recipe.Kind() != artifact.KindRecipe ||
		spec.Code.Kind() != artifact.KindEvidence || spec.Proposer.Kind() != artifact.KindEvidence ||
		spec.Code == spec.Proposer || spec.Candidate.Kind() != improvementCandidateKind(spec.Kind) ||
		spec.Kind == ImprovementCorpus && spec.Candidate == spec.Dataset ||
		spec.Kind == ImprovementRecipe && spec.Candidate == spec.Recipe {
		return ImprovementProposal{}, errors.New("training program: invalid improvement proposal")
	}
	components := slices.Clone(spec.Components)
	sort.Slice(components, func(i, j int) bool { return components[i].String() < components[j].String() })
	if err := validateImprovementComponents(spec.Kind, components); err != nil {
		return ImprovementProposal{}, err
	}
	body := struct {
		Kind             ImprovementKind `json:"kind"`
		ParentModel      artifact.ID     `json:"parent_model"`
		Candidate        artifact.ID     `json:"candidate"`
		Dataset          artifact.ID     `json:"dataset"`
		DevelopmentSplit artifact.ID     `json:"development_split"`
		Recipe           artifact.ID     `json:"recipe"`
		Code             artifact.ID     `json:"code"`
		Proposer         artifact.ID     `json:"proposer"`
		Components       []artifact.ID   `json:"components,omitempty"`
	}{
		Kind: spec.Kind, ParentModel: spec.ParentModel, Candidate: spec.Candidate,
		Dataset: spec.Dataset, DevelopmentSplit: spec.DevelopmentSplit, Recipe: spec.Recipe,
		Code: spec.Code, Proposer: spec.Proposer, Components: components,
	}
	id, err := identifyJSON(artifact.KindRecipe, body)
	if err != nil {
		return ImprovementProposal{}, err
	}
	return ImprovementProposal{
		id: id, kind: spec.Kind, parentModel: spec.ParentModel, candidate: spec.Candidate,
		dataset: spec.Dataset, developmentSplit: spec.DevelopmentSplit, recipe: spec.Recipe,
		code: spec.Code, proposer: spec.Proposer, components: components,
	}, nil
}

func (p ImprovementProposal) ID() artifact.ID               { return p.id }
func (p ImprovementProposal) ProposalKind() ImprovementKind { return p.kind }
func (p ImprovementProposal) ParentModel() artifact.ID      { return p.parentModel }
func (p ImprovementProposal) Candidate() artifact.ID        { return p.candidate }
func (p ImprovementProposal) Dataset() artifact.ID          { return p.dataset }
func (p ImprovementProposal) DevelopmentSplit() artifact.ID { return p.developmentSplit }
func (p ImprovementProposal) Recipe() artifact.ID           { return p.recipe }
func (p ImprovementProposal) Code() artifact.ID             { return p.code }
func (p ImprovementProposal) Proposer() artifact.ID         { return p.proposer }
func (p ImprovementProposal) Components() []artifact.ID     { return slices.Clone(p.components) }

func improvementCandidateKind(kind ImprovementKind) artifact.Kind {
	switch kind {
	case ImprovementCorpus:
		return artifact.KindDataset
	case ImprovementRecipe:
		return artifact.KindRecipe
	case ImprovementEvaluator:
		return artifact.KindEvidence
	case ImprovementComponentComposition:
		return artifact.KindModelDefinition
	default:
		return artifact.KindInvalid
	}
}

func validateImprovementComponents(kind ImprovementKind, components []artifact.ID) error {
	if kind != ImprovementComponentComposition {
		if len(components) != 0 {
			return errors.New("training program: non-composition proposal carries components")
		}
		return nil
	}
	if len(components) == 0 {
		return errors.New("training program: composition proposal lacks components")
	}
	for index, component := range components {
		if !component.Valid() || index > 0 && components[index-1] == component {
			return errors.New("training program: invalid composition components")
		}
	}
	return nil
}
