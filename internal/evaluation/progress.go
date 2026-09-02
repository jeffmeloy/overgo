package evaluation

import (
	"context"
	"time"
)

// Progress is one observed step of a suite's case loop: how many cases
// are scored, how many the suite holds, the running score over the
// scored cases, and the wall time spent so far. A sink derives rate
// and remaining time from these; the loop never guesses a cadence.
type Progress struct {
	Suite   string
	Done    int
	Total   int
	Score   float64
	Scored  bool
	Elapsed time.Duration
}

// ProgressSink receives every scored case. It owns the cadence of any
// output it produces: the suite loops report each case and stay silent
// themselves, so a 5761-case suite on a slow model is observable at
// whatever granularity the operator's sink chooses.
type ProgressSink func(Progress)

type progressKey struct{}

// WithProgress binds a sink to the context every suite loop reads. A
// context without one runs the loops unobserved, exactly as before.
func WithProgress(ctx context.Context, sink ProgressSink) context.Context {
	if ctx == nil || sink == nil {
		return ctx
	}
	return context.WithValue(ctx, progressKey{}, sink)
}

// progressTracker counts one suite's loop against a bound sink.
type progressTracker struct {
	sink    ProgressSink
	suite   string
	total   int
	done    int
	sum     float64
	started time.Time
}

func trackProgress(ctx context.Context, suite string, total int) *progressTracker {
	if ctx == nil {
		return nil
	}
	sink, _ := ctx.Value(progressKey{}).(ProgressSink)
	if sink == nil {
		return nil
	}
	return &progressTracker{sink: sink, suite: suite, total: total, started: time.Now()}
}

// observe records one case whose score contributes to a mean (1 or 0
// for a correct/incorrect choice, a probability mass, a rule rate).
func (tracker *progressTracker) observe(score float64) {
	if tracker == nil {
		return
	}
	tracker.done++
	tracker.sum += score
	tracker.sink(Progress{
		Suite: tracker.suite, Done: tracker.done, Total: tracker.total,
		Score: tracker.sum / float64(tracker.done), Scored: true,
		Elapsed: time.Since(tracker.started),
	})
}

// hit records one case with a binary verdict: the running score is the
// accuracy over the cases scored so far.
func (tracker *progressTracker) hit(correct bool) {
	var score float64
	if correct {
		score = 1
	}
	tracker.observe(score)
}

// advance records one case without a mean-forming score (a sequence
// likelihood whose aggregate is a perplexity over tokens, not cases).
func (tracker *progressTracker) advance() {
	if tracker == nil {
		return
	}
	tracker.done++
	tracker.sink(Progress{
		Suite: tracker.suite, Done: tracker.done, Total: tracker.total,
		Elapsed: time.Since(tracker.started),
	})
}
