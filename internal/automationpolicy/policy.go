// Package automationpolicy owns the evidence-gated lifecycle of
// automation policies: a policy names the strategy that should drive a
// slot of automation, and it moves declared -> experimental ->
// verified -> active exactly as model recipes promote -- every
// promotion demands measured evidence from the store, activation
// retains the incumbent as its rollback target, and no stage is ever
// skipped or asserted.
package automationpolicy

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// Stage is one lifecycle position; the order is the promotion path.
type Stage string

const (
	// StageDeclared is a named intention with no evidence yet.
	StageDeclared Stage = "declared"
	// StageExperimental means at least one recorded experiment ran it.
	StageExperimental Stage = "experimental"
	// StageVerified means attempt history shows repeated measured wins.
	StageVerified Stage = "verified"
	// StageActive means the policy governs its slot; the incumbent it
	// displaced is retained as the rollback target.
	StageActive Stage = "active"
)

// promotionOrder walks the single sanctioned path.
var promotionOrder = []Stage{StageDeclared, StageExperimental, StageVerified, StageActive}

const (
	policyMediaType   = "application/vnd.overgo.automation-policy+json"
	policySchema      = "overgo/automation-policy/v1"
	policyAliasPrefix = "automation-policy/"
	// candidateAliasSuffix tracks a challenger's lifecycle off the
	// governing slot: the slot alias always names the last ACTIVATED
	// policy, so trials never leave the slot ungoverned.
	candidateAliasSuffix = "/candidate"
)

// Verification thresholds: a strategy verifies on repeated measured
// success, never a single lucky run. Three attempts with a majority
// succeeding is the smallest sample that is repetition rather than an
// anecdote.
const (
	verifiedMinAttempts  = 3
	verifiedMinSuccesses = 2
)

// Policy is one slot's governing document at one lifecycle stage.
type Policy struct {
	ID       artifact.ID   `json:"-"`
	Version  uint16        `json:"version"`
	Name     string        `json:"name"`
	Strategy string        `json:"strategy"`
	Stage    Stage         `json:"stage"`
	Evidence []artifact.ID `json:"evidence,omitempty"`
	// RollbackTo is the incumbent active policy this one displaced;
	// activation over an incumbent must retain it.
	RollbackTo artifact.ID `json:"rollback_to,omitzero"`
}

var policyCodec = artifact.JSONDocumentCodec(
	"automation policy", artifact.KindRecipe, policyMediaType, policySchema,
	func(value *Policy) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion {
			return errors.New("automation policy: invalid version")
		}
		if strings.TrimSpace(value.Name) == "" || strings.ContainsAny(value.Name, " /\x00\r\n\t") ||
			strings.TrimSpace(value.Strategy) == "" || strings.ContainsAny(value.Strategy, " /\x00\r\n\t") {
			return errors.New("automation policy: name and strategy must be bounded tokens")
		}
		switch value.Stage {
		case StageDeclared, StageExperimental, StageVerified, StageActive:
		default:
			return errors.New("automation policy: invalid stage")
		}
		if value.Stage != StageDeclared && len(value.Evidence) == 0 {
			return errors.New("automation policy: a promoted stage requires its evidence")
		}
		for _, id := range value.Evidence {
			if !id.Valid() {
				return errors.New("automation policy: invalid evidence identity")
			}
		}
		return nil
	},
	func(value Policy) artifact.ID { return value.ID },
	func(value *Policy, id artifact.ID) { value.ID = id },
	func(value Policy) Policy {
		value.Evidence = append([]artifact.ID(nil), value.Evidence...)
		return value
	},
)

// Load reads one slot's governing policy: the last activated document.
// Absent means the slot has never activated a policy.
func Load(ctx context.Context, store *overgodb.Store, name string) (Policy, bool, error) {
	return loadAlias(ctx, store, policyAliasPrefix+name)
}

// LoadCandidate reads the slot's challenger mid-lifecycle.
func LoadCandidate(ctx context.Context, store *overgodb.Store, name string) (Policy, bool, error) {
	return loadAlias(ctx, store, policyAliasPrefix+name+candidateAliasSuffix)
}

func loadAlias(ctx context.Context, store *overgodb.Store, alias string) (Policy, bool, error) {
	id, bound, err := artifact.ResolveAlias(ctx, store, alias)
	if err != nil || !bound {
		return Policy{}, false, err
	}
	return policyCodec.Read(ctx, store, id)
}

// Declare opens a challenger's lifecycle at declared. One challenger
// runs per slot at a time: a mid-lifecycle candidate refuses a second,
// while an activated one clears the way for the next.
func Declare(ctx context.Context, store *overgodb.Store, name, strategy string) (Policy, error) {
	candidate, standing, err := LoadCandidate(ctx, store, name)
	if err != nil {
		return Policy{}, err
	}
	if standing && candidate.Stage != StageActive {
		return Policy{}, fmt.Errorf("automation policy: slot %q already has a candidate mid-lifecycle", name)
	}
	var previous *artifact.ID
	if standing {
		previous = &candidate.ID
	}
	return publish(ctx, store, Policy{Name: name, Strategy: strategy, Stage: StageDeclared}, previous)
}

// Promote advances a slot exactly one stage, gated on measured store
// evidence: an experiment report for experimental, repeated attempt
// wins for verified, and a measured non-regression over the incumbent
// with a retained rollback target for active.
func Promote(ctx context.Context, store *overgodb.Store, name string, evidence []artifact.ID) (Policy, error) {
	current, standing, err := LoadCandidate(ctx, store, name)
	if err != nil {
		return Policy{}, err
	}
	if !standing {
		return Policy{}, fmt.Errorf("automation policy: slot %q has no declared candidate", name)
	}
	target, err := nextStage(current.Stage)
	if err != nil {
		return Policy{}, err
	}
	promoted := Policy{
		Name: current.Name, Strategy: current.Strategy, Stage: target,
		Evidence: append(append([]artifact.ID(nil), current.Evidence...), evidence...),
	}
	switch target {
	case StageExperimental:
		if err := requireStoredEvidence(ctx, store, evidence); err != nil {
			return Policy{}, err
		}
	case StageVerified:
		if err := requireMeasuredWins(ctx, store, current.Strategy); err != nil {
			return Policy{}, err
		}
	case StageActive:
		incumbent, governed, err := Load(ctx, store, name)
		if err != nil {
			return Policy{}, err
		}
		if governed {
			if err := requireNonRegression(ctx, store, current.Strategy, incumbent.Strategy); err != nil {
				return Policy{}, err
			}
			promoted.RollbackTo = incumbent.ID
		}
	}
	return publish(ctx, store, promoted, &current.ID)
}

// Rollback reactivates the displaced incumbent an active policy
// retained; a policy without one has nothing to roll back to.
func Rollback(ctx context.Context, store *overgodb.Store, name string) (Policy, error) {
	current, standing, err := Load(ctx, store, name)
	if err != nil {
		return Policy{}, err
	}
	if !standing || current.Stage != StageActive || !current.RollbackTo.Valid() {
		return Policy{}, fmt.Errorf("automation policy: slot %q holds no active policy with a rollback target", name)
	}
	previous, found, err := policyCodec.Read(ctx, store, current.RollbackTo)
	if err != nil {
		return Policy{}, err
	}
	if !found {
		return Policy{}, errors.New("automation policy: rollback target content is absent")
	}
	alias := artifact.AliasBinding{Name: policyAliasPrefix + name, Target: previous.ID, Previous: &current.ID}
	_, err = store.Commit(ctx, artifact.Batch{
		Key: "automation-policy/rollback/" + current.ID.String(), Aliases: []artifact.AliasBinding{alias},
	})
	if err != nil {
		return Policy{}, err
	}
	return previous, nil
}

func nextStage(current Stage) (Stage, error) {
	for index, stage := range promotionOrder {
		if stage == current {
			if index+1 == len(promotionOrder) {
				return "", errors.New("automation policy: the policy is already active")
			}
			return promotionOrder[index+1], nil
		}
	}
	return "", errors.New("automation policy: invalid current stage")
}

// requireStoredEvidence demands every cited experiment exists in the
// store: a promotion cannot cite evidence nobody recorded.
func requireStoredEvidence(ctx context.Context, store *overgodb.Store, evidence []artifact.ID) error {
	if len(evidence) == 0 {
		return errors.New("automation policy: experimental promotion requires recorded experiment evidence")
	}
	for _, id := range evidence {
		_, _, found, err := store.OpenContent(ctx, id)
		if err != nil {
			return err
		}
		if !found {
			return fmt.Errorf("automation policy: evidence %s is not in the store", id)
		}
	}
	return nil
}

// requireMeasuredWins demands the strategy's own attempt history shows
// repeated success before verification.
func requireMeasuredWins(ctx context.Context, store *overgodb.Store, strategy string) error {
	attempts, successes, err := strategyRecord(ctx, store, strategy)
	if err != nil {
		return err
	}
	if attempts < verifiedMinAttempts || successes < verifiedMinSuccesses {
		return fmt.Errorf(
			"automation policy: verification requires at least %d attempts with %d successes; history shows %d/%d",
			verifiedMinAttempts, verifiedMinSuccesses, successes, attempts,
		)
	}
	return nil
}

// requireNonRegression demands the candidate's measured success rate
// is not below the incumbent strategy's: activation is a measured win,
// never a preference.
func requireNonRegression(ctx context.Context, store *overgodb.Store, candidate, incumbent string) error {
	candidateAttempts, candidateSuccesses, err := strategyRecord(ctx, store, candidate)
	if err != nil {
		return err
	}
	incumbentAttempts, incumbentSuccesses, err := strategyRecord(ctx, store, incumbent)
	if err != nil {
		return err
	}
	if candidateAttempts == 0 || incumbentAttempts == 0 {
		return errors.New("automation policy: activation requires measured history for both strategies")
	}
	// Cross-multiplied rate comparison avoids floating error entirely.
	if candidateSuccesses*incumbentAttempts < incumbentSuccesses*candidateAttempts {
		return fmt.Errorf(
			"automation policy: candidate %s measures below incumbent %s (%d/%d vs %d/%d)",
			candidate, incumbent, candidateSuccesses, candidateAttempts, incumbentSuccesses, incumbentAttempts,
		)
	}
	return nil
}

func strategyRecord(ctx context.Context, store *overgodb.Store, strategy string) (attempts, successes int, err error) {
	history, err := runrecord.LoadAttemptHistory(ctx, store, runrecord.AttemptFilter{Strategy: strategy})
	if err != nil {
		return 0, 0, err
	}
	for _, step := range history.Steps {
		attempts += step.Attempts
		successes += step.Succeeded
	}
	return attempts, successes, nil
}

// publish commits the document and binds the candidate alias; an
// activation additionally rebinds the governing slot alias with the
// displaced incumbent as its recorded predecessor.
func publish(ctx context.Context, store *overgodb.Store, policy Policy, previousCandidate *artifact.ID) (Policy, error) {
	policy.Version = artifact.InitialDocumentVersion
	published, err := policyCodec.New(policy)
	if err != nil {
		return Policy{}, err
	}
	content, err := policyCodec.Content(published)
	if err != nil {
		return Policy{}, err
	}
	aliases := []artifact.AliasBinding{{
		Name:   policyAliasPrefix + published.Name + candidateAliasSuffix,
		Target: published.ID, Previous: previousCandidate,
	}}
	if published.Stage == StageActive {
		slot := artifact.AliasBinding{Name: policyAliasPrefix + published.Name, Target: published.ID}
		if published.RollbackTo.Valid() {
			slot.Previous = &published.RollbackTo
		}
		aliases = append(aliases, slot)
	}
	_, err = store.Commit(ctx, artifact.Batch{
		Key:       "automation-policy/" + published.ID.String(),
		Artifacts: []artifact.Descriptor{content.Descriptor},
		Contents:  []artifact.Content{content},
		Aliases:   aliases,
	})
	if err != nil {
		return Policy{}, err
	}
	return published, nil
}
