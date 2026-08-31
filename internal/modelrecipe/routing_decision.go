package modelrecipe

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/recipe"
	"overgo/internal/textcheck"
)

const (
	// RecipeRoutingDecisionMediaType identifies recipe-routing decision documents.
	RecipeRoutingDecisionMediaType = "application/vnd.overgo.recipe-routing-decision+json"
	// RecipeRoutingDecisionSchema identifies the recipe-routing decision contract.
	RecipeRoutingDecisionSchema = "overgo/recipe-routing-decision/v1"
	// RoutingDerivationCheapestEligible is the live registered derivation
	// rule: among candidates whose published evidence meets the signal's
	// admitted threshold, the cheapest by measured resource is selected.
	RoutingDerivationCheapestEligible = "overgo/routing-derivation/cheapest-eligible/v1"
	// RoutingDerivationBestQualityEligible is the registered counterfactual
	// candidate rule: among eligible candidates, the highest measured
	// quality is selected, with ties broken by cheaper resource then recipe
	// identity.
	RoutingDerivationBestQualityEligible = "overgo/routing-derivation/best-quality-eligible/v1"
	// routingTextBytes bounds every free-text signal field.
	routingTextBytes = 512
)

// RoutingSignal names the task demand one routing decision answers: the
// task, the capability it must serve, and the evidence-derived quality
// threshold a candidate must meet. The threshold is an admitted derived
// bound, never a tunable weight.
type RoutingSignal struct {
	Task              recipe.Task `json:"task"`
	Capability        string      `json:"capability"`
	QualityThreshold  float64     `json:"quality_threshold"`
	ThresholdEvidence artifact.ID `json:"threshold_evidence"`
}

// RoutingCandidate is one considered serving option: the exact recipe and
// model identities, the published evidence it was judged on, and the
// measured quality and resource read from that evidence.
type RoutingCandidate struct {
	Recipe        artifact.ID   `json:"recipe"`
	Model         artifact.ID   `json:"model"`
	Evidence      []artifact.ID `json:"evidence"`
	Quality       float64       `json:"quality"`
	ResourceBytes uint64        `json:"resource_bytes"`
}

// RecipeRoutingDecision is the canonical recipe-routing fact: which verified recipe
// serves a task signal, derived from the named candidates' published
// evidence by exactly one registered derivation rule. The record binds the
// signal, every candidate considered, the evidence each was judged on, and
// the derived selection, so the decision is replayable and auditable like
// any other promotion.
type RecipeRoutingDecision struct {
	Version    uint16             `json:"version"`
	Signal     RoutingSignal      `json:"signal"`
	Candidates []RoutingCandidate `json:"candidates"`
	Selected   artifact.ID        `json:"selected"`
	Derivation string             `json:"derivation"`
	ID         artifact.ID        `json:"-"`
}

var recipeRoutingDecisionCodec = artifact.JSONDocumentCodec(
	"recipe routing decision", artifact.KindEvidence, RecipeRoutingDecisionMediaType, RecipeRoutingDecisionSchema,
	canonicalizeRecipeRoutingDecision,
	func(value RecipeRoutingDecision) artifact.ID { return value.ID },
	func(value *RecipeRoutingDecision, id artifact.ID) { value.ID = id },
	func(value RecipeRoutingDecision) RecipeRoutingDecision {
		value.Candidates = slices.Clone(value.Candidates)
		for index, candidate := range value.Candidates {
			value.Candidates[index].Evidence = slices.Clone(candidate.Evidence)
		}
		return value
	},
)

func canonicalizeRecipeRoutingDecision(value *RecipeRoutingDecision) error {
	if value.Version != artifact.InitialDocumentVersion {
		return errors.New("model recipe: invalid routing decision version")
	}
	if !value.Signal.Task.Valid() {
		return fmt.Errorf("model recipe: invalid routing task %q", value.Signal.Task)
	}
	if value.Signal.Capability == "" || !textcheck.Bounded(value.Signal.Capability, routingTextBytes, "\x00") {
		return errors.New("model recipe: routing signal requires a bounded capability")
	}
	if !checked.Finite64(value.Signal.QualityThreshold) || value.Signal.ThresholdEvidence.Kind() != artifact.KindEvidence {
		return errors.New("model recipe: routing threshold must be a finite bound derived from exact evidence")
	}
	if len(value.Candidates) == 0 {
		return errors.New("model recipe: routing decision requires at least one candidate")
	}
	slices.SortFunc(value.Candidates, func(a, b RoutingCandidate) int {
		return strings.Compare(a.Recipe.String(), b.Recipe.String())
	})
	for index, candidate := range value.Candidates {
		if candidate.Recipe.Kind() != artifact.KindRecipe || candidate.Model.Kind() != artifact.KindModel {
			return errors.New("model recipe: routing candidate requires exact recipe and model identities")
		}
		if index > 0 && candidate.Recipe == value.Candidates[index-1].Recipe {
			return fmt.Errorf("model recipe: duplicate routing candidate %s", candidate.Recipe)
		}
		if len(candidate.Evidence) == 0 {
			return fmt.Errorf("model recipe: candidate %s was judged on no evidence", candidate.Recipe)
		}
		for _, id := range candidate.Evidence {
			if !id.Valid() {
				return fmt.Errorf("model recipe: candidate %s cites invalid evidence", candidate.Recipe)
			}
		}
		slices.SortFunc(candidate.Evidence, func(a, b artifact.ID) int {
			return strings.Compare(a.String(), b.String())
		})
		value.Candidates[index].Evidence = slices.Compact(candidate.Evidence)
		if !checked.Finite64(candidate.Quality) || candidate.ResourceBytes == 0 {
			return fmt.Errorf("model recipe: candidate %s lacks finite measured quality and resource", candidate.Recipe)
		}
	}
	if !slices.ContainsFunc(value.Candidates, func(candidate RoutingCandidate) bool {
		return candidate.Recipe == value.Selected
	}) {
		return errors.New("model recipe: routing selection must name a considered candidate")
	}
	if value.Derivation != RoutingDerivationCheapestEligible && value.Derivation != RoutingDerivationBestQualityEligible {
		return fmt.Errorf("model recipe: unregistered routing derivation %q", value.Derivation)
	}
	return nil
}

// selectByDerivation applies one registered derivation rule to the eligible
// candidate set; both rules share one eligibility test — the signal's
// evidence-derived threshold — and differ only in the preference among
// eligibles, so a corpus recorded under either replays under the other.
func selectByDerivation(
	derivation string, signal RoutingSignal, candidates []RoutingCandidate,
) (*RoutingCandidate, error) {
	prefer := func(candidate, incumbent *RoutingCandidate) bool {
		if candidate.ResourceBytes != incumbent.ResourceBytes {
			return candidate.ResourceBytes < incumbent.ResourceBytes
		}
		return strings.Compare(candidate.Recipe.String(), incumbent.Recipe.String()) < 0
	}
	if derivation == RoutingDerivationBestQualityEligible {
		cheapest := prefer
		prefer = func(candidate, incumbent *RoutingCandidate) bool {
			if candidate.Quality != incumbent.Quality {
				return candidate.Quality > incumbent.Quality
			}
			return cheapest(candidate, incumbent)
		}
	} else if derivation != RoutingDerivationCheapestEligible {
		return nil, fmt.Errorf("model recipe: unregistered routing derivation %q", derivation)
	}
	var selected *RoutingCandidate
	for index := range candidates {
		candidate := &candidates[index]
		if candidate.Quality < signal.QualityThreshold {
			continue
		}
		if selected == nil || prefer(candidate, selected) {
			selected = candidate
		}
	}
	if selected == nil {
		return nil, errors.New("model recipe: no candidate's published evidence meets the routing threshold")
	}
	return selected, nil
}

// DeriveRecipeRoutingDecision applies the one registered derivation rule to a
// task signal and its judged candidates: among candidates whose measured
// quality meets the signal's evidence-derived threshold, the cheapest by
// measured resource is selected, with ties broken by recipe identity so the
// derivation is deterministic. No candidate meeting the threshold is a
// refusal, not a fallback.
func DeriveRecipeRoutingDecision(
	signal RoutingSignal, candidates []RoutingCandidate,
) (RecipeRoutingDecision, error) {
	selected, err := selectByDerivation(RoutingDerivationCheapestEligible, signal, candidates)
	if err != nil {
		return RecipeRoutingDecision{}, err
	}
	return NewRecipeRoutingDecision(RecipeRoutingDecision{
		Signal: signal, Candidates: candidates, Selected: selected.Recipe,
		Derivation: RoutingDerivationCheapestEligible,
	})
}

// NewRecipeRoutingDecision canonicalizes and identifies one immutable recipe-routing
// decision record.
func NewRecipeRoutingDecision(value RecipeRoutingDecision) (RecipeRoutingDecision, error) {
	return recipeRoutingDecisionCodec.NewInitial(value)
}

// ParseRecipeRoutingDecision decodes one canonical recipe-routing decision document and
// proves its content identity.
func ParseRecipeRoutingDecision(content []byte) (RecipeRoutingDecision, error) {
	return recipeRoutingDecisionCodec.Parse(content)
}

// Lineage binds the decision to its threshold evidence, every candidate's
// identities and judged evidence, and the selected recipe.
func (value RecipeRoutingDecision) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Signal.ThresholdEvidence}
	for _, candidate := range value.Candidates {
		parents = append(parents, candidate.Recipe, candidate.Model)
		parents = append(parents, candidate.Evidence...)
	}
	return artifact.UniqueDependencyLineage(value.ID, parents...)
}

// Batch wraps the decision as one committable store batch.
func (value RecipeRoutingDecision) Batch(key string) (artifact.Batch, error) {
	return recipeRoutingDecisionCodec.Batch(key, value, value.Lineage(), nil)
}
