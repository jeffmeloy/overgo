package gate

import (
	"context"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
)

// Operation classes name what a landing is. They are derived from state --
// the ship set and the store's own chain -- never declared, so the precheck
// a class owns cannot be skipped by forgetting to say what the work was.
const (
	classPlanEdit      = "plan-edit"
	classCodeChange    = "code-change"
	classStoreMutation = "store-mutation"
)

// operationClasses reads the ship set for plan edit or code change, and the
// chain for a store mutation: content released after the last landing.
func operationClasses(paths []string, released, landed uint64) []string {
	classes := []string{classCodeChange}
	if len(paths) == 1 && paths[0] == plan.Path {
		classes[0] = classPlanEdit
	}
	if released > landed {
		classes = append(classes, classStoreMutation)
	}
	return classes
}

// requireOperationPrechecks derives the landing's classes and holds each to
// the precheck it owns. Plan edits and code changes are the gate's own
// pipeline; a store mutation must carry the proof its candidate earned,
// which the store admits only from the command that ran the checks. It reads
// the lifecycle head before this gate publishes its preparation over it.
func (g *gateContext) requireOperationPrechecks(ctx context.Context, store *overgodb.Store) error {
	var landed uint64
	if current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias); err != nil {
		return err
	} else if found {
		introduction, _, err := store.ArtifactIntroduction(ctx, current)
		if err != nil {
			return err
		}
		landed = introduction.Sequence
	}
	g.advise(noteClass, "operation class: "+strings.Join(operationClasses(g.paths, store.LastRelease(), landed), "+"))
	return store.RequireProvenSince(ctx, landed)
}
