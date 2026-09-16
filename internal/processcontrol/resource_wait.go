package processcontrol

import (
	"context"
	"errors"
)

// AwaitResource makes the claim, and while another process holds the
// resource the claim names waits on that holder: the wait ends when a
// holder exits or releases the resource, or when ctx ends, and the claim
// is then made again. Failures other than contention return at once;
// callers retain and release their own claim.
func AwaitResource(ctx context.Context, claim func() error) error {
	var waiter *releaseWaiter
	defer func() {
		if waiter != nil {
			_ = waiter.close()
		}
	}()
	var busy error
	for {
		if err := context.Cause(ctx); err != nil {
			return errors.Join(busy, err)
		}
		// Armed before the claim: a release between the claim and the wait
		// is then observed by the wait rather than lost.
		if waiter != nil {
			if err := waiter.arm(); err != nil {
				return errors.Join(busy, err)
			}
		}
		err := claim()
		if !errors.Is(err, ErrResourceBusy) {
			return err
		}
		busy = err
		if waiter == nil {
			contention, named := errors.AsType[*ResourceBusyError](err)
			if !named || !validResourceName(contention.Name) {
				return errors.Join(err, errors.New("processcontrol: the contention names no resource to wait on"))
			}
			waiter, err = openReleaseWaiter(contention.Name)
			if err != nil {
				return errors.Join(busy, err)
			}
			// The first refusal was made before the waiter existed; arm
			// and claim once more so no release goes unobserved.
			continue
		}
		if err := waiter.wait(ctx); err != nil {
			return errors.Join(busy, err)
		}
	}
}
