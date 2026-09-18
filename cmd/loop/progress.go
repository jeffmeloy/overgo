package main

import (
	"context"
	"errors"
	"strings"

	"overgo/internal/authoritylock"
	"overgo/internal/loop"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/worklease"
)

// ProgressCheckpoint pins the revision before the worker starts.
func (w *execWorld) ProgressCheckpoint() (string, error) {
	out, err := runTool("git", "rev-parse", "HEAD")
	if err != nil {
		return "", err
	}
	revision := strings.TrimSpace(out)
	if !worklease.ValidCommit(revision) {
		return "", errors.New("loop: progress checkpoint is not an exact commit")
	}
	return revision, nil
}

// AcceptedProgress checks validated local completion evidence since checkpoint.
func (w *execWorld) AcceptedProgress(step loop.Step, checkpoint string) (bool, error) {
	if w.stopStore == nil {
		return false, errors.New("loop: accepted progress requires the completion store")
	}
	lock, err := authoritylock.Acquire(loopWorktree)
	if err != nil {
		return false, err
	}
	defer lock.Close()
	ctx := context.Background()
	if err := w.stopStore.Refresh(ctx); err != nil {
		return false, err
	}
	obligation, found, err := runrecord.CurrentGateLaneObligation(ctx, w.stopStore)
	if err != nil {
		return false, err
	}
	// Current normally waits or recovers validation first. Check the durable
	// owner again here; pending and failed validation cannot earn credit.
	if found && !obligation.Resolved() {
		return false, nil
	}
	document, err := plan.Load(plan.Path)
	if err != nil {
		return false, err
	}
	authority, err := plan.ResolveCompletionAuthority(ctx, loopWorktree, "HEAD", document, w.stopStore)
	if err != nil {
		return false, err
	}
	return authority.AcceptedProgressSince(ctx, document, step.Item, step.ID, checkpoint)
}
