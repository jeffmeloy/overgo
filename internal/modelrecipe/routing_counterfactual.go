package modelrecipe

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

const (
	// RoutingCounterfactualMediaType identifies counterfactual replay evidence.
	RoutingCounterfactualMediaType = "application/vnd.overgo.routing-counterfactual+json"
	// RoutingCounterfactualSchema identifies the counterfactual contract.
	RoutingCounterfactualSchema = "overgo/routing-counterfactual/v1"
)

// RoutingCounterfactual is the published measurement of one candidate
// selection policy replayed over the recorded decision corpus: identical
// inputs, both policies, exact deltas with complete coverage. It is the
// admission evidence a routing-policy rollout must cite, so policy
// improvement is measured before it is trusted with live work.
type RoutingCounterfactual struct {
	Version       uint16        `json:"version"`
	Policy        string        `json:"policy"`
	Decisions     []artifact.ID `json:"decisions"`
	Changed       uint64        `json:"changed"`
	QualityDelta  float64       `json:"quality_delta"`
	ResourceDelta int64         `json:"resource_delta"`
	Coverage      uint64        `json:"coverage"`
	ID            artifact.ID   `json:"-"`
}

var routingCounterfactualCodec = artifact.JSONDocumentCodec(
	"routing counterfactual", artifact.KindEvidence,
	RoutingCounterfactualMediaType, RoutingCounterfactualSchema,
	canonicalizeRoutingCounterfactual,
	func(value RoutingCounterfactual) artifact.ID { return value.ID },
	func(value *RoutingCounterfactual, id artifact.ID) { value.ID = id },
	func(value RoutingCounterfactual) RoutingCounterfactual {
		value.Decisions = slices.Clone(value.Decisions)
		return value
	},
)

func canonicalizeRoutingCounterfactual(value *RoutingCounterfactual) error {
	if value.Version != artifact.InitialDocumentVersion {
		return errors.New("model recipe: invalid counterfactual version")
	}
	if value.Policy != RoutingDerivationCheapestEligible && value.Policy != RoutingDerivationBestQualityEligible {
		return fmt.Errorf("model recipe: unregistered routing derivation %q", value.Policy)
	}
	if len(value.Decisions) == 0 {
		return errors.New("model recipe: counterfactual evidence requires a recorded decision corpus")
	}
	slices.SortFunc(value.Decisions, func(a, b artifact.ID) int {
		return strings.Compare(a.String(), b.String())
	})
	for index, id := range value.Decisions {
		if id.Kind() != artifact.KindEvidence {
			return errors.New("model recipe: counterfactual corpus entries must be recorded decision evidence")
		}
		if index > 0 && id == value.Decisions[index-1] {
			return fmt.Errorf("model recipe: duplicate corpus decision %s", id)
		}
	}
	if value.Coverage != uint64(len(value.Decisions)) {
		return errors.New("model recipe: counterfactual coverage must equal the replayed corpus")
	}
	if value.Changed > value.Coverage || !checked.Finite64(value.QualityDelta) {
		return errors.New("model recipe: counterfactual deltas are not measured")
	}
	return nil
}

// Lineage binds the counterfactual to every replayed decision record.
func (value RoutingCounterfactual) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Decisions...)
}

// ReplayRoutingDecisions evaluates one candidate selection policy against
// the recorded decision corpus at near-zero cost: each recorded decision is
// loaded with its identity proved, its exact signal and candidate set are
// replayed under the candidate policy, and the quality and resource deltas
// between the counterfactual and recorded selections accumulate into one
// published evidence record citing every replayed decision. Coverage is
// complete by construction — a corpus entry that fails to load, parse, or
// replay refuses the whole measurement rather than shrinking it.
func ReplayRoutingDecisions(
	ctx context.Context,
	store artifact.Repository,
	policy string,
	decisions []artifact.ID,
) (RoutingCounterfactual, error) {
	if len(decisions) == 0 {
		return RoutingCounterfactual{}, errors.New("model recipe: counterfactual evidence requires a recorded decision corpus")
	}
	report := RoutingCounterfactual{Policy: policy, Decisions: decisions}
	for _, id := range decisions {
		content, found, err := artifact.ReadContent(ctx, store, id)
		if err != nil {
			return RoutingCounterfactual{}, err
		}
		if !found {
			return RoutingCounterfactual{}, fmt.Errorf("model recipe: recorded decision %s is absent", id)
		}
		recorded, err := ParseRecipeRoutingDecision(content.Data)
		if err != nil {
			return RoutingCounterfactual{}, err
		}
		counterfactual, err := selectByDerivation(policy, recorded.Signal, recorded.Candidates)
		if err != nil {
			return RoutingCounterfactual{}, fmt.Errorf("model recipe: decision %s does not replay: %w", id, err)
		}
		var live *RoutingCandidate
		for index := range recorded.Candidates {
			if recorded.Candidates[index].Recipe == recorded.Selected {
				live = &recorded.Candidates[index]
				break
			}
		}
		if live == nil {
			return RoutingCounterfactual{}, fmt.Errorf("model recipe: decision %s lost its recorded selection", id)
		}
		if counterfactual.Recipe != live.Recipe {
			report.Changed++
		}
		report.QualityDelta += counterfactual.Quality - live.Quality
		report.ResourceDelta += int64(counterfactual.ResourceBytes) - int64(live.ResourceBytes)
	}
	report.Coverage = uint64(len(decisions))
	report.Version = artifact.InitialDocumentVersion
	report, err := routingCounterfactualCodec.New(report)
	if err != nil {
		return RoutingCounterfactual{}, err
	}
	batch, err := routingCounterfactualCodec.Batch(
		"routing/counterfactual/"+report.ID.String(), report, report.Lineage(), nil,
	)
	if err != nil {
		return RoutingCounterfactual{}, err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return RoutingCounterfactual{}, err
	}
	return report, nil
}
