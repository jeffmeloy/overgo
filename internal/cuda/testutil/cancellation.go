package testutil

import (
	"context"

	"overgo/internal/cuda/driver"
)

// CancelAfterDeviceWork cancels at the first context check after this library
// submits a kernel or graph. It exercises a real execution boundary without a
// timer, additional device work, or assumptions about a model's layer count.
func CancelAfterDeviceWork(parent context.Context, library *driver.Library) (context.Context, context.CancelCauseFunc) {
	ctx, cancel := context.WithCancelCause(parent)
	return workCancellation{Context: ctx, library: library, before: library.ExecutionStats(), cancel: cancel}, cancel
}

type workCancellation struct {
	context.Context
	library *driver.Library
	before  driver.ExecutionStats
	cancel  context.CancelCauseFunc
}

// Err cancels after observed device work and returns the underlying context error.
func (ctx workCancellation) Err() error {
	after := ctx.library.ExecutionStats()
	if after.KernelLaunches+after.GraphLaunches > ctx.before.KernelLaunches+ctx.before.GraphLaunches {
		ctx.cancel(context.Canceled)
	}
	return ctx.Context.Err()
}
