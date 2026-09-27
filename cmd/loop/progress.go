package main

import (
	"context"
	"errors"
	"strings"

	"overgo/internal/loop"
	"overgo/internal/plan"
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

// AcceptedProgress reports a landing since checkpoint that moved the row
// forward, read from git.
func (w *execWorld) AcceptedProgress(step loop.Step, checkpoint string) (bool, error) {
	document, err := plan.Load(plan.Path)
	if err != nil {
		return false, err
	}
	return plan.AcceptedProgressSince(context.Background(), loopWorktree, document, step.Item, step.ID, checkpoint)
}
