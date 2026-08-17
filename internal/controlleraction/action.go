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
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/modelartifact"
	"overgo/internal/strictjson"
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

// Action is one controller emission: exactly the payload matching Kind is
// present. There is no field anywhere in the language that carries code or
// shell -- parameters are artifact identities, typed facts and enums; the one
// command-shaped string (RequiredVerifier) must validate as a failable go
// test declaration inside the proposal authority itself.
type Action struct {
	Version       uint16               `json:"version"`
	Kind          Kind                 `json:"kind"`
	Chain         *ChainAction         `json:"chain,omitempty"`
	Proposal      *ProposalAction      `json:"proposal,omitempty"`
	Decomposition *DecompositionAction `json:"decomposition,omitempty"`
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
	for _, present := range []bool{a.Chain != nil, a.Proposal != nil, a.Decomposition != nil} {
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
	default:
		return fmt.Errorf("controller action: kind %q is not in the allowlist", a.Kind)
	}
	return nil
}

// Compile executes one action deterministically through the shared
// authorities and returns the produced content-addressed artifacts. This
// switch is the whole execution semantics of the language.
func Compile(action Action) ([]artifact.Content, error) {
	if err := action.validate(); err != nil {
		return nil, err
	}
	switch action.Kind {
	case KindChainRecipes:
		chainProgram, baseProgram, err := composition.ChainPrograms(action.Chain.Scorer, action.Chain.Drafter)
		if err != nil {
			return nil, err
		}
		chainContent, err := chainProgram.Definition().ArtifactContent()
		if err != nil {
			return nil, err
		}
		baseContent, err := baseProgram.Definition().ArtifactContent()
		if err != nil {
			return nil, err
		}
		return []artifact.Content{chainContent, baseContent}, nil
	case KindBridgeProposal:
		proposal, err := composition.NewBridgeProposal(
			action.Proposal.Target, slices.Clone(action.Proposal.Candidates),
			action.Proposal.RequiredVerifier, action.Proposal.Blocker,
		)
		if err != nil {
			return nil, err
		}
		content, err := proposal.Content()
		if err != nil {
			return nil, err
		}
		return []artifact.Content{content}, nil
	case KindComponentDecomposition:
		decomposition, err := modelartifact.NewComponentDecomposition(
			action.Decomposition.Model, action.Decomposition.Family,
			action.Decomposition.Modality, action.Decomposition.Tensors,
		)
		if err != nil {
			return nil, err
		}
		content, err := decomposition.Content()
		if err != nil {
			return nil, err
		}
		return []artifact.Content{content}, nil
	default:
		return nil, fmt.Errorf("controller action: kind %q has no executor", action.Kind)
	}
}
