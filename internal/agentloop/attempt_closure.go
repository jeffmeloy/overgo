package agentloop

import (
	"context"

	"overgo/internal/runrecord"
)

// CloseAttempt publishes the terminal receipt for one attempt through
// the coordinator's store, so session closure and receipt publication
// share one authority.
func (c *Coordinator) CloseAttempt(ctx context.Context, receipt runrecord.TerminalAttemptReceipt) (runrecord.TerminalAttemptReceipt, error) {
	return runrecord.PublishTerminalAttemptReceipt(ctx, c.store, receipt)
}
