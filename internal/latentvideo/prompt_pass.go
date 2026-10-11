package latentvideo

import (
	"context"

	"overgo/internal/processmeasure"
)

// promptRowsAwaited names the part of a request spent waiting for its
// prompts' rows while another line of work encodes them.
const promptRowsAwaited = "prompt rows awaited"

// promptPass is the encoding of a request's prompts begun beside the loading
// of the model that serves it. The load that began it holds it until a
// runtime does, and neither lets go of it while it runs: it is stopped and
// waited for, or what it found is taken.
type promptPass struct {
	stop context.CancelFunc
	done chan struct{}
	err  error
	// stopped says the pass ended under a cancelled context: it was given up, and what it returned is no finding.
	stopped bool
}

// beginPromptPass starts encode beside its caller, under the caller's context
// and a stop of its own, timing nothing into the request's parts.
func beginPromptPass(ctx context.Context, encode func(context.Context) error) *promptPass {
	ctx, stop := context.WithCancel(processmeasure.Unstaged(ctx))
	pass := &promptPass{stop: stop, done: make(chan struct{})}
	go func() {
		defer close(pass.done)
		pass.err = encode(ctx)
		pass.stopped = ctx.Err() != nil
	}()
	return pass
}

// end stops the pass and returns when it has ended. No pass has ended.
func (pass *promptPass) end() {
	if pass == nil {
		return
	}
	pass.stop()
	<-pass.done
}

// taken waits for the pass and gives what it failed with, so that a request
// is told of a failed encoding and does not run it a second time unsaid. A
// request cancelled while it waits stops the pass, waits for it, and has its
// cancellation. A pass its own request's going stopped did not fail: the
// request that takes it encodes for itself.
func (pass *promptPass) taken(ctx context.Context) error {
	if pass == nil {
		return nil
	}
	defer processmeasure.Stage(ctx, promptRowsAwaited)()
	select {
	case <-pass.done:
	case <-ctx.Done():
	}
	pass.end()
	if err := ctx.Err(); err != nil || pass.stopped {
		return err
	}
	return pass.err
}

// loadedBeside gives what load loads while the pass encodes beside it. A load
// that fails, or whose caller has gone, stops the pass and returns when it
// has ended, with what the load itself gave: nothing is left encoding, or
// keeping rows in the store, for a model that was not loaded.
func loadedBeside[Runtime any](ctx context.Context, pass *promptPass, load func() (Runtime, error)) (Runtime, error) {
	loaded, err := load()
	if err != nil || ctx.Err() != nil {
		pass.end()
	}
	return loaded, err
}
