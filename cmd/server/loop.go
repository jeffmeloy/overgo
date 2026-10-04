package main

import (
	"context"
	"os"
	"path/filepath"

	"overgo/internal/gitauthority"
	"overgo/internal/plan"
	llamaserver "overgo/internal/server"
)

// checkoutLoop reads and stops the unattended loop of one checkout.
type checkoutLoop struct {
	root string
}

// LoopStatus reads the checkout's loop view.
func (loop checkoutLoop) LoopStatus(ctx context.Context) (llamaserver.LoopStatus, error) {
	view, err := plan.ReadLoopView(ctx, loop.root)
	if err != nil {
		return llamaserver.LoopStatus{}, err
	}
	return llamaserver.LoopStatus{
		Running: view.Running, Stopped: view.Stopped, StopReason: view.StopReason, Complete: view.Complete, Row: view.Row, RowTitle: view.RowTitle,
		Landing: llamaserver.LoopLanding{Commit: view.Landing.Commit, Row: view.Landing.Row, Subject: view.Landing.Subject, At: view.Landing.At},
	}, nil
}

// StopLoop records a user stop on the checkout.
func (loop checkoutLoop) StopLoop(ctx context.Context, detail string) error {
	return plan.StopLoop(ctx, loop.root, detail)
}

// serverLoop is the loop of the checkout the server starts in, when that
// checkout holds a plan; elsewhere there is none.
func serverLoop(ctx context.Context) llamaserver.LoopControl {
	root, err := gitauthority.RepositoryRoot(ctx, ".")
	if err != nil {
		return nil
	}
	if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(plan.Path))); err != nil {
		return nil
	}
	return checkoutLoop{root: root}
}
