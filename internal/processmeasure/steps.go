package processmeasure

import "context"

type stepsKey struct{}

// WithSteps gives a request an observer of its sampling steps: a sampler that runs under the context returned tells
// observe each step it finishes, with how many it takes in all, so a page shows a run's progress as it goes
// (moss-page/result).
func WithSteps(ctx context.Context, observe func(done, total int)) context.Context {
	return context.WithValue(ctx, stepsKey{}, observe)
}

// Stepped tells the observer the context carries that done of a sampler's total steps are finished. Under a context
// that carries none it tells nothing.
func Stepped(ctx context.Context, done, total int) {
	if ctx == nil {
		return
	}
	if observe, _ := ctx.Value(stepsKey{}).(func(done, total int)); observe != nil {
		observe(done, total)
	}
}
