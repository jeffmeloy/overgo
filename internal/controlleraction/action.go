// Package controlleraction defines the allowlisted artifact-transformation
// action language the workflow controller emits. Every action is a typed
// document naming one transformation kind with typed parameters; deterministic
// Go executors compile actions through the existing recipe, composition and
// model-artifact authorities into content-addressed artifacts. The language
// contains no code, no shell, and no runtime orchestration -- an action a Go
// executor cannot compile is unrepresentable, and identical actions compile
// to identical artifacts.
package controlleraction

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/evaluation"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scratchmodel"
	"overgo/internal/strictjson"
	"overgo/internal/trainingprogram"
)

// ActionVersion is the action document version. Actions are controller
// emissions consumed by the executor, not store documents; the artifacts the
// executor produces carry their own registered media types.
const ActionVersion uint16 = 1

// Kind names one allowlisted transformation. The switch in Compile is the
// entire language: a kind without a deterministic executor cannot exist.
type Kind string

const (
	// KindChainRecipes compiles the Tier-0 two-model chain recipe pair.
	KindChainRecipes Kind = "chain-recipes"
	// KindBridgeProposal emits a promotion-blocked composition proposal.
	KindBridgeProposal Kind = "bridge-proposal"
	// KindComponentDecomposition classifies a model's tensors across the
	// organ axes.
	KindComponentDecomposition Kind = "component-decomposition"
	// KindComposeModel assembles constituents into one executable
	// content-addressed model artifact with full lineage.
	KindComposeModel       Kind = "compose-model"
	KindImprovementTrial   Kind = "improvement-trial"
	KindEvaluationBudget   Kind = "evaluation-budget"
	KindEvaluatorPromotion Kind = "evaluator-promotion"
)

// ChainAction: compose two whole validated models through typed ports.
type ChainAction struct {
	Scorer  artifact.ID `json:"scorer"`
	Drafter artifact.ID `json:"drafter"`
}

// ProposalAction: convert retrieval candidates into a blocked proposal.
type ProposalAction struct {
	Target           artifact.ID                   `json:"target"`
	Candidates       []composition.BridgeCandidate `json:"candidates"`
	RequiredVerifier string                        `json:"required_verifier"`
	Blocker          string                        `json:"blocker"`
}

// DecompositionAction: classify one model's tensor facts.
type DecompositionAction struct {
	Model    artifact.ID                `json:"model"`
	Family   string                     `json:"family,omitempty"`
	Modality string                     `json:"modality,omitempty"`
	Tensors  []modelartifact.TensorFact `json:"tensors"`
}

// ComposeAction: assemble a composed model artifact from its constituents.
type ComposeAction struct {
	Document composition.ComposedModelDocument `json:"document"`
}

type ImprovementAction struct {
	Proposal       trainingprogram.ImprovementSpec `json:"proposal"`
	Profile        *scratchmodel.DerivationProfile `json:"profile,omitempty"`
	PromotionSplit artifact.ID                     `json:"promotion_split"`
	Evaluator      artifact.ID                     `json:"evaluator"`
	Authority      artifact.ID                     `json:"authority"`
	Decision       *ImprovementDecisionAction      `json:"decision,omitempty"`
}

type ImprovementDecisionAction struct {
	Child      artifact.ID                        `json:"child"`
	Run        artifact.ID                        `json:"run"`
	Evaluation artifact.ID                        `json:"evaluation"`
	Decider    artifact.ID                        `json:"decider"`
	State      runrecord.ImprovementDecisionState `json:"state"`
}

type BudgetChargeAction struct {
	Amount   uint64      `json:"amount"`
	Consumer artifact.ID `json:"consumer"`
	Purpose  string      `json:"purpose"`
}

type EvaluationBudgetAction struct {
	Unit      string               `json:"unit"`
	Split     artifact.ID          `json:"split"`
	Issued    uint64               `json:"issued"`
	Authority artifact.ID          `json:"authority"`
	Charges   []BudgetChargeAction `json:"charges"`
}

type EvaluatorPromotionAction struct {
	Candidate  evaluation.Evaluator       `json:"candidate"`
	OutcomeSet artifact.ID                `json:"outcome_set"`
	Outcomes   []evaluation.KnownOutcome  `json:"outcomes"`
	Current    runrecord.AdmissionBinding `json:"current"`
	Prior      runrecord.AdmissionBinding `json:"prior"`
	Approval   recipe.Decision            `json:"approval"`
}

// Action is one controller emission: exactly the payload matching Kind is
// present. There is no field anywhere in the language that carries code or
// shell -- parameters are artifact identities, typed facts and enums; the one
// command-shaped string (RequiredVerifier) must validate as a failable go
// test declaration inside the proposal authority itself.
type Action struct {
	Version       uint16                    `json:"version"`
	Kind          Kind                      `json:"kind"`
	Chain         *ChainAction              `json:"chain,omitempty"`
	Proposal      *ProposalAction           `json:"proposal,omitempty"`
	Decomposition *DecompositionAction      `json:"decomposition,omitempty"`
	Compose       *ComposeAction            `json:"compose,omitempty"`
	Improvement   *ImprovementAction        `json:"improvement,omitempty"`
	Budget        *EvaluationBudgetAction   `json:"budget,omitempty"`
	Evaluator     *EvaluatorPromotionAction `json:"evaluator,omitempty"`
}

// ParseAction decodes one strict action document.
func ParseAction(data []byte) (Action, error) {
	var action Action
	if err := strictjson.DecodeBytes(data, &action); err != nil {
		return Action{}, fmt.Errorf("controller action: %w", err)
	}
	if err := action.validate(); err != nil {
		return Action{}, err
	}
	return action, nil
}

func (a Action) validate() error {
	if a.Version != ActionVersion {
		return errors.New("controller action: unsupported version")
	}
	payloads := 0
	for _, present := range []bool{a.Chain != nil, a.Proposal != nil, a.Decomposition != nil, a.Compose != nil, a.Improvement != nil, a.Budget != nil, a.Evaluator != nil} {
		if present {
			payloads++
		}
	}
	if payloads != 1 {
		return fmt.Errorf("controller action: exactly one payload required, have %d", payloads)
	}
	switch a.Kind {
	case KindChainRecipes:
		if a.Chain == nil {
			return errors.New("controller action: chain-recipes requires the chain payload")
		}
	case KindBridgeProposal:
		if a.Proposal == nil {
			return errors.New("controller action: bridge-proposal requires the proposal payload")
		}
	case KindComponentDecomposition:
		if a.Decomposition == nil {
			return errors.New("controller action: component-decomposition requires the decomposition payload")
		}
	case KindComposeModel:
		if a.Compose == nil {
			return errors.New("controller action: compose-model requires the compose payload")
		}
	case KindImprovementTrial:
		if a.Improvement == nil || !validImprovementDecisionFields(*a.Improvement) {
			return errors.New("controller action: improvement-trial requires coherent authority and decision fields")
		}
	case KindEvaluationBudget:
		if a.Budget == nil || len(a.Budget.Charges) == 0 {
			return errors.New("controller action: evaluation-budget requires a grant and charges")
		}
	case KindEvaluatorPromotion:
		if a.Evaluator == nil {
			return errors.New("controller action: evaluator-promotion requires evidence")
		}
	default:
		return fmt.Errorf("controller action: kind %q is not in the allowlist", a.Kind)
	}
	return nil
}

// Compile executes one action deterministically through the shared
// authorities and returns the produced content-addressed artifacts. This
// switch is the whole execution semantics of the language.
func CompileTransaction(ctx context.Context, reader artifact.Reader, action Action) (artifact.Batch, error) {
	if err := action.validate(); err != nil {
		return artifact.Batch{}, err
	}
	identity, err := artifact.JSONID(artifact.KindProfile, action)
	if err != nil {
		return artifact.Batch{}, err
	}
	contents, lineage, err := compileAction(ctx, reader, action)
	if err != nil {
		return artifact.Batch{}, err
	}
	return artifact.NewDocumentBatch("controller-action/"+string(action.Kind)+"/"+identity.String(), contents, lineage, nil)
}

func compileAction(ctx context.Context, reader artifact.Reader, action Action) ([]artifact.Content, []artifact.Lineage, error) {
	switch action.Kind {
	case KindChainRecipes:
		chainProgram, baseProgram, err := composition.ChainPrograms(action.Chain.Scorer, action.Chain.Drafter)
		if err != nil {
			return nil, nil, err
		}
		chainContent, err := chainProgram.Definition().ArtifactContent()
		if err != nil {
			return nil, nil, err
		}
		baseContent, err := baseProgram.Definition().ArtifactContent()
		if err != nil {
			return nil, nil, err
		}
		return []artifact.Content{chainContent, baseContent}, nil, nil
	case KindBridgeProposal:
		proposal, err := composition.NewBridgeProposal(
			action.Proposal.Target, slices.Clone(action.Proposal.Candidates),
			action.Proposal.RequiredVerifier, action.Proposal.Blocker,
		)
		if err != nil {
			return nil, nil, err
		}
		content, err := proposal.Content()
		if err != nil {
			return nil, nil, err
		}
		return []artifact.Content{content}, proposal.Lineage(), nil
	case KindComposeModel:
		composed, err := composition.NewComposedModel(action.Compose.Document)
		if err != nil {
			return nil, nil, err
		}
		content, err := composed.Content()
		if err != nil {
			return nil, nil, err
		}
		return []artifact.Content{content}, composed.Lineage(), nil
	case KindComponentDecomposition:
		decomposition, err := modelartifact.NewComponentDecomposition(
			action.Decomposition.Model, action.Decomposition.Family,
			action.Decomposition.Modality, action.Decomposition.Tensors,
		)
		if err != nil {
			return nil, nil, err
		}
		content, err := decomposition.Content()
		if err != nil {
			return nil, nil, err
		}
		return []artifact.Content{content}, []artifact.Lineage{decomposition.Lineage()}, nil
	case KindImprovementTrial:
		return compileImprovementTrial(ctx, reader, *action.Improvement)
	case KindEvaluationBudget:
		return compileEvaluationBudget(*action.Budget)
	case KindEvaluatorPromotion:
		return compileEvaluatorPromotion(*action.Evaluator)
	default:
		return nil, nil, fmt.Errorf("controller action: kind %q has no executor", action.Kind)
	}
}

func compileEvaluatorPromotion(action EvaluatorPromotionAction) ([]artifact.Content, []artifact.Lineage, error) {
	if err := evaluation.ValidateEvaluatorPromotion(action.Candidate, action.OutcomeSet, action.Outcomes, action.Current, action.Prior, action.Approval); err != nil {
		return nil, nil, err
	}
	candidate, err := action.Candidate.Content()
	if err != nil {
		return nil, nil, err
	}
	approval, err := action.Approval.Content()
	if err != nil {
		return nil, nil, err
	}
	return []artifact.Content{candidate, approval}, append(action.Candidate.Lineage(), action.Approval.Lineage()...), nil
}

func compileEvaluationBudget(action EvaluationBudgetAction) ([]artifact.Content, []artifact.Lineage, error) {
	budget, err := runrecord.NewBudget(action.Unit, action.Split, action.Issued, action.Authority)
	if err != nil {
		return nil, nil, err
	}
	content, err := budget.Content()
	if err != nil {
		return nil, nil, err
	}
	contents := make([]artifact.Content, 1, 1+len(action.Charges))
	contents[0] = content
	lineage := budget.Lineage()
	for _, declaration := range action.Charges {
		charge, err := runrecord.NewBudgetCharge(budget.ID, declaration.Amount, declaration.Consumer, declaration.Purpose)
		if err != nil {
			return nil, nil, err
		}
		content, err := charge.Content()
		if err != nil {
			return nil, nil, err
		}
		contents = append(contents, content)
		lineage = append(lineage, charge.Lineage()...)
	}
	return contents, lineage, nil
}

func compileImprovementTrial(
	ctx context.Context,
	reader artifact.Reader,
	action ImprovementAction,
) ([]artifact.Content, []artifact.Lineage, error) {
	var profileContent []artifact.Content
	if action.Profile != nil {
		profile, err := scratchmodel.NewDerivationProfile(*action.Profile)
		if err != nil {
			return nil, nil, err
		}
		if action.Proposal.Candidate.Valid() && action.Proposal.Candidate != profile.ID {
			return nil, nil, errors.New("controller action: derivation profile identity differs")
		}
		action.Proposal.Candidate = profile.ID
		content, err := profile.Content()
		if err != nil {
			return nil, nil, err
		}
		profileContent = []artifact.Content{content}
	}
	proposal, err := trainingprogram.CompileImprovementProposal(action.Proposal)
	if err != nil {
		return nil, nil, err
	}
	admission, err := runrecord.AdmitImprovement(proposal, action.Authority, action.Evaluator, action.PromotionSplit)
	if err != nil {
		return nil, nil, err
	}
	proposalContent, err := proposal.Content()
	if err != nil {
		return nil, nil, err
	}
	admissionContent, err := admission.Content()
	if err != nil {
		return nil, nil, err
	}
	contents := append(profileContent, proposalContent, admissionContent)
	lineage := append(proposal.Lineage(), admission.Lineage()...)
	if action.Decision == nil {
		return contents, lineage, nil
	}
	if ctx == nil || reader == nil {
		return nil, nil, errors.New("controller action: improvement decision store is absent")
	}
	run, err := loadRun(ctx, reader, action.Decision.Run)
	if err != nil {
		return nil, nil, err
	}
	evaluation, err := loadEvaluation(ctx, reader, action.Decision.Evaluation)
	if err != nil {
		return nil, nil, err
	}
	decision, err := runrecord.DecideImprovement(
		admission, action.Decision.Child, run, evaluation, action.Decision.Decider, action.Decision.State,
	)
	if err != nil {
		return nil, nil, err
	}
	decisionContent, err := decision.Content()
	if err != nil {
		return nil, nil, err
	}
	return append(contents, decisionContent), append(lineage, decision.Lineage()...), nil
}

func validImprovementDecisionFields(action ImprovementAction) bool {
	if action.PromotionSplit.Kind() != artifact.KindDatasetShard || action.Evaluator.Kind() != artifact.KindEvidence ||
		action.Authority.Kind() != artifact.KindEvidence {
		return false
	}
	if (action.Proposal.Kind == trainingprogram.ImprovementDerivationProfile) != (action.Profile != nil) {
		return false
	}
	if action.Decision == nil {
		return true
	}
	return action.Decision.Child.Kind() == artifact.KindModel && action.Decision.Run.Kind() == artifact.KindRun &&
		action.Decision.Evaluation.Kind() == artifact.KindEvaluation && action.Decision.Decider.Kind() == artifact.KindEvidence &&
		(action.Decision.State == runrecord.ImprovementPromote || action.Decision.State == runrecord.ImprovementRefuse)
}

func loadRun(ctx context.Context, reader artifact.Reader, id artifact.ID) (runrecord.Run, error) {
	content, found, err := reader.Content(ctx, id)
	if err != nil || !found {
		return runrecord.Run{}, errors.Join(err, errors.New("controller action: improvement run is absent"))
	}
	return runrecord.ParseRun(content.Data)
}

func loadEvaluation(ctx context.Context, reader artifact.Reader, id artifact.ID) (runrecord.Evaluation, error) {
	content, found, err := reader.Content(ctx, id)
	if err != nil || !found {
		return runrecord.Evaluation{}, errors.Join(err, errors.New("controller action: improvement evaluation is absent"))
	}
	return runrecord.ParseEvaluation(content.Data)
}
