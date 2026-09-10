package processcontrol

import (
	"context"
	"errors"
	"time"
)

// Poll admission at most twenty times per second; no work starts before success.
// The caller's context bounds admission and preserves its cancellation cause.
const resourceAdmissionPoll = time.Second / 20

// AwaitResource retries typed contention until admission or context cancellation.
// Other failures return immediately; callers retain and release their own claim.
func AwaitResource(ctx context.Context, claim func() error) error {
	var busy error
	for {
		if err := context.Cause(ctx); err != nil {
			return errors.Join(busy, err)
		}
		err := claim()
		if !errors.Is(err, ErrResourceBusy) {
			return err
		}
		busy = err
		timer := time.NewTimer(resourceAdmissionPoll)
		select {
		case <-ctx.Done():
			timer.Stop()
			return errors.Join(err, context.Cause(ctx))
		case <-timer.C:
		}
	}
}
