package gate

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
)

// batchEvidence retains the current requirements and immutable prior revisions.
// A changed input replaces its requirement explicitly; removal cannot erase debt.
type batchEvidence struct {
	Version   uint16                        `json:"version"`
	ID        artifact.ID                   `json:"-"`
	Task      artifact.ID                   `json:"task"`
	Base      artifact.ID                   `json:"base"`
	Candidate artifact.ID                   `json:"candidate"`
	Scope     []string                      `json:"scope"`
	Previous  artifact.ID                   `json:"previous,omitzero"`
	Checks    map[string]batchCheckEvidence `json:"checks"`
}

type batchCheckEvidence struct {
	Contract   plan.VerificationCheckpoint `json:"contract"`
	Obligation artifact.ID                 `json:"obligation"`
	Slot       artifact.ID                 `json:"slot"`
	Input      artifact.ID                 `json:"input"`
	Result     automationcheck.CacheEntry  `json:"result,omitzero"`
	Resolution artifact.ID                 `json:"resolution,omitzero"`
}

var batchEvidenceCodec = artifact.JSONDocumentCodec(
	"gate batch evidence", artifact.KindEvidence,
	"application/vnd.overgo.gate-batch-evidence+json", "overgo/gate-batch-evidence/v1",
	func(value *batchEvidence) error {
		if value.Version != artifact.InitialDocumentVersion || value.Task.Kind() != artifact.KindRecipe || !value.Base.Valid() || !value.Candidate.Valid() ||
			value.Previous.Valid() && value.Previous.Kind() != artifact.KindEvidence || len(value.Checks) == 0 {
			return errors.New("gate batch evidence: invalid authority")
		}
		for name, check := range value.Checks {
			if name != check.Contract.GateCheckName() || check.Contract.Verify == "" || check.Obligation.Kind() != artifact.KindEvidence || check.Slot.Kind() != artifact.KindRecipe || !check.Input.Valid() {
				return errors.New("gate batch evidence: invalid check")
			}
			if check.Result != (automationcheck.CacheEntry{}) && (check.Result.Invocation != check.Slot || check.Result.Input != check.Input ||
				check.Result.Evidence.Kind() != artifact.KindEvidence || check.Result.Outcome != runrecord.LanePassed) {
				return errors.New("gate batch evidence: invalid terminal result")
			}
			if check.Result.Evidence.Valid() != check.Resolution.Valid() || check.Resolution.Valid() && check.Resolution.Kind() != artifact.KindEvidence {
				return errors.New("gate batch evidence: terminal result requires an exact resolution")
			}
		}
		return nil
	}, func(value batchEvidence) artifact.ID { return value.ID },
	func(value *batchEvidence, id artifact.ID) { value.ID = id },
	func(value batchEvidence) batchEvidence {
		value.Scope = slices.Clone(value.Scope)
		value.Checks = maps.Clone(value.Checks)
		for name, check := range value.Checks {
			check.Contract.DependsOn = slices.Clone(check.Contract.DependsOn)
			value.Checks[name] = check
		}
		return value
	},
)

type batchEvidenceLedger struct {
	mutex sync.Mutex
	store *overgodb.Store
	state batchEvidence
	alias string
}

// openBatchEvidence publishes requirements before running any batch member.
// The retry file is a projection; exact persisted results rebuild its slots.
func (g *gateContext) openBatchEvidence(checks []automationcheck.Invocation, inputs map[artifact.ID]artifact.ID, cache *automationcheck.EvidenceCache) (*batchEvidenceLedger, error) {
	if g.verificationBatch == nil && g.planRef == "" {
		return nil, nil
	}
	if g.verificationBatch != nil && g.manifestPlan == nil {
		return nil, errors.New("gate batch evidence: source-bound manifest is required")
	}
	owner, err := g.openPackageEvidence()
	if err != nil {
		return nil, err
	}
	ledger := &batchEvidenceLedger{store: owner.store}
	if err := ledger.prepare(context.Background(), g, checks, inputs, cache); err != nil {
		return nil, errors.Join(err, ledger.store.Close())
	}
	if g.verificationBatch == nil {
		return nil, ledger.store.Close()
	}
	return ledger, nil
}

func (ledger *batchEvidenceLedger) prepare(ctx context.Context, g *gateContext, checks []automationcheck.Invocation, inputs map[artifact.ID]artifact.ID, cache *automationcheck.EvidenceCache) error {
	task, err := artifact.JSONID(artifact.KindRecipe, struct{ Policy, Reference string }{"gate-batch/v1", g.planRef})
	if err != nil {
		return err
	}
	ledger.alias = "gate/batch/" + task.String()
	previous, found, err := batchEvidenceCodec.Resolve(ctx, ledger.store, ledger.alias)
	if err != nil {
		return err
	}
	if found && previous.Task != task {
		return errors.New("gate batch evidence: task mismatch")
	}
	if g.verificationBatch == nil {
		if found {
			return errors.New("gate batch evidence: replan removes the registered batch before parent promotion")
		}
		return nil
	}
	next := batchEvidence{Task: task, Base: g.manifestPlan.BaseManifest, Candidate: g.manifestPlan.CandidateManifest, Scope: g.verificationBatch.Scope, Previous: previous.ID, Checks: map[string]batchCheckEvidence{}}
	publication := artifact.Batch{}
	identities := map[artifact.ID]bool{task: true, g.environment.ID: true, next.Base: true, next.Candidate: true}
	for _, check := range checks {
		index := slices.IndexFunc(g.verificationBatch.Checkpoints, func(member plan.VerificationCheckpoint) bool { return member.GateCheckName() == check.Check.Name })
		if index < 0 {
			continue
		}
		slot, input, _ := g.checkCacheKey(check, inputs)
		contract, err := artifact.JSONID(artifact.KindProfile, struct {
			Checkpoint plan.VerificationCheckpoint
			Scope      []string
		}{g.verificationBatch.Checkpoints[index], g.verificationBatch.Scope})
		if err != nil {
			return err
		}
		obligation, err := runrecord.NewAgentObligation(runrecord.AgentObligation{
			Task: task, Name: "verification-checkpoint", Scope: check.Check.Name,
			Sources: []artifact.ID{input, contract, g.environment.ID},
		})
		if err != nil {
			return err
		}
		content, err := obligation.Content()
		if err != nil {
			return err
		}
		publication.Contents = append(publication.Contents, content)
		publication.Lineage = append(publication.Lineage, obligation.Lineage()...)
		identities[input], identities[contract] = true, true
		member := batchCheckEvidence{Contract: g.verificationBatch.Checkpoints[index], Obligation: obligation.ID, Slot: slot.ID, Input: input}
		old := previous.Checks[check.Check.Name]
		if old.Obligation == member.Obligation && old.Slot == member.Slot && old.Input == member.Input {
			member.Result = old.Result
			member.Resolution = old.Resolution
			if member.Resolution.Valid() {
				resolution, err := runrecord.RequireAgentObligationResolution(ctx, ledger.store, member.Resolution)
				if err != nil {
					return err
				}
				if !runrecord.AgentObligationSatisfied(obligation, []runrecord.AgentObligationResolution{resolution}) || !slices.Equal(resolution.Evidence, []artifact.ID{member.Result.Evidence}) {
					return errors.New("gate batch evidence: mismatched obligation resolution")
				}
			}
		}
		next.Checks[check.Check.Name] = member
	}
	if len(next.Checks) != len(g.verificationBatch.Checkpoints) {
		return errors.New("gate batch evidence: required checkpoint absent from execution")
	}
	for name := range previous.Checks {
		if _, retained := next.Checks[name]; !retained {
			return fmt.Errorf("gate batch evidence: replan drops outstanding checkpoint %s", name)
		}
	}
	for id := range identities {
		descriptor, found, err := ledger.store.Artifact(ctx, id)
		if err != nil {
			return err
		}
		if !found {
			descriptor = artifact.Descriptor{ID: id}
		}
		publication.Artifacts = append(publication.Artifacts, descriptor)
	}
	if err := ledger.publish(ctx, next, publication); err != nil {
		return err
	}
	for _, member := range next.Checks {
		delete(cache.Entries, member.Slot.String())
		if member.Result.Evidence.Valid() {
			cache.Entries[member.Slot.String()] = member.Result
		}
	}
	return nil
}

func (ledger *batchEvidenceLedger) publish(ctx context.Context, next batchEvidence, publication artifact.Batch) error {
	record, err := batchEvidenceCodec.NewInitial(next)
	if err != nil {
		return err
	}
	content, err := batchEvidenceCodec.Content(record)
	if err != nil {
		return err
	}
	publication.Contents = append(publication.Contents, content)
	parents := []artifact.ID{record.Task, record.Base, record.Candidate}
	if record.Previous.Valid() {
		parents = append(parents, record.Previous)
	}
	for _, check := range record.Checks {
		parents = append(parents, check.Obligation)
		if check.Result.Evidence.Valid() {
			parents = append(parents, check.Resolution)
		}
	}
	publication.Lineage = append(publication.Lineage, artifact.DependencyLineage(record.ID, parents...)...)
	alias := artifact.AliasBinding{Name: ledger.alias, Target: record.ID}
	if record.Previous.Valid() {
		alias.Previous = &record.Previous
	}
	publication.Aliases = append(publication.Aliases, alias)
	publication.Key = ledger.alias + "/" + record.ID.String()
	if _, err := artifact.CommitBatch(ctx, ledger.store, publication); err != nil {
		return err
	}
	ledger.state = record
	return nil
}

// record keeps a failed member open and preserves unrelated terminal successes.
// Publication must succeed before a retry projection receives evidence credit.
func (ledger *batchEvidenceLedger) record(ctx context.Context, name string, evidence automationcheck.Evidence) error {
	ledger.mutex.Lock()
	defer ledger.mutex.Unlock()
	check, found := ledger.state.Checks[name]
	if !found {
		return nil
	}
	check.Result = automationcheck.CacheEntry{}
	check.Resolution = artifact.ID{}
	publication := artifact.Batch{}
	if evidence.Outcome == runrecord.LanePassed && !evidence.Inapplicable {
		if !evidence.ID.Valid() || evidence.Name != name {
			return fmt.Errorf("gate batch evidence: %s has no evidence identity", name)
		}
		check.Result = automationcheck.CacheEntry{Invocation: check.Slot, Input: check.Input, Evidence: evidence.ID, Outcome: evidence.Outcome}
		obligation, err := runrecord.RequireAgentObligation(ctx, ledger.store, check.Obligation)
		if err != nil {
			return err
		}
		resolution, err := runrecord.NewAgentObligationResolution(runrecord.AgentObligationResolution{
			Obligation: obligation.ID, Scope: obligation.Scope, MutationEpoch: obligation.MutationEpoch, Evidence: []artifact.ID{evidence.ID},
		})
		if err != nil {
			return err
		}
		content, err := resolution.Content()
		if err != nil {
			return err
		}
		publication.Contents = append(publication.Contents, content)
		publication.Lineage = append(publication.Lineage, resolution.Lineage()...)
		check.Resolution = resolution.ID
	}
	next := ledger.state
	next.Previous = next.ID
	next.Checks = maps.Clone(next.Checks)
	next.Checks[name] = check
	if check.Result.Evidence.Valid() {
		publication.Artifacts = append(publication.Artifacts, artifact.Descriptor{ID: check.Result.Evidence})
	}
	return ledger.publish(ctx, next, publication)
}
