package processcontrol

import (
	"context"
	"errors"
)

// AwaitResource claims the named physical resource, and while another
// process holds it waits on that holder: the wait ends when a holder exits
// or releases the resource, or when ctx ends, and the claim is then made
// again. Failures other than contention return at once; callers retain and
// release their own claim.
func AwaitResource(ctx context.Context, name string, claim func() error) error {
	if !validResourceName(name) {
		return errors.New("processcontrol: invalid physical resource name")
	}
	waiter, err := openReleaseWaiter(name)
	if err != nil {
		return err
	}
	defer waiter.close()
	var busy error
	for {
		if err := context.Cause(ctx); err != nil {
			return errors.Join(busy, err)
		}
		// Armed before the claim: a release between the claim and the wait
		// is then observed by the wait rather than lost.
		if err := waiter.arm(); err != nil {
			return errors.Join(busy, err)
		}
		err := claim()
		if !errors.Is(err, ErrResourceBusy) {
			return err
		}
		busy = err
		if err := waiter.wait(ctx); err != nil {
			return errors.Join(busy, err)
		}
	}
}
