package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/authoritylock"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

// Called only inside withPlanMutation's process-owned critical section.
func savePlanMutation(root string, document plan.Plan) error {
	path := filepath.Join(root, filepath.FromSlash(plan.Path))
	before, err := plan.Load(path)
	if err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		return err
	}
	defer store.Close()
	if err := plan.ValidateClaimedPlan(context.Background(), store, before, document); err != nil {
		return err
	}
	return plan.Save(path, document)
}

func releaseDispatchClaim(root, rawID, worker, reason string, output io.Writer) (err error) {
	if reason != "cancelled" && reason != "handoff" {
		return errors.New("plan: -release-claim requires -release-reason cancelled or handoff; completion belongs to the gate")
	}
	id, err := artifact.ParseID(rawID)
	if err != nil {
		return err
	}
	lock, err := authoritylock.Acquire(root)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, lock.Close()) }()
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, store.Close()) }()
	lease, found, err := plan.ReadWorkLease(context.Background(), store, id)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("plan: claim %s is unavailable", id)
	}
	worktree, err := filepath.Abs(root)
	if err != nil {
		return err
	}
	if !strings.EqualFold(filepath.ToSlash(worktree), lease.Worktree) {
		return errors.New("plan: release from the claimed worktree so its gate and plan mutation lock protects the handoff")
	}
	batch, err := lease.ReleaseBatch(worker, reason)
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(context.Background(), store, batch); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "claim=%s task=%s state=released reason=%s; evidence and outstanding checks retained\n", id, lease.Task, reason)
	return err
}
