package clioptions

import (
	"context"
	"time"
)

// Limited is ctx under a limit the user set, which ends it with cause, and ctx as it stands where the user set none
// (a limit of zero): inside the machine a clock never ends work by itself, only a limit a person asked for
// (no-clocks/own-budgets).
func Limited(ctx context.Context, limit time.Duration, cause error) (context.Context, context.CancelFunc) {
	if limit <= 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeoutCause(ctx, limit, cause)
}
