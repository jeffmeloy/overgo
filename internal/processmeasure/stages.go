package processmeasure

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"
)

// part is one named part of a request: the wall that was its own, over the
// number of times the request went through it.
type part struct {
	name  string
	wall  time.Duration
	times int
}

// Stages is where one request's time went: the wall of each named part of it.
// A part's wall is its own. What a part begun inside it took belongs to that
// part and not also to the outer one, so no moment is counted twice, and the
// parts' sum is the time the request spent inside a named part: what is left
// of the request's wall is time nothing names.
type Stages struct {
	mu    sync.Mutex
	open  []openPart
	parts []part
	err   error
}

type openPart struct {
	started, inner time.Duration
}

type stagesKey struct{}

// WithStages gives a request a ledger of its parts: every Stage begun under
// the context returned is timed into it.
func WithStages(ctx context.Context) (context.Context, *Stages) {
	stages := &Stages{}
	return context.WithValue(ctx, stagesKey{}, stages), stages
}

// Unstaged gives work done beside a request a context that times nothing into
// the request's ledger: a ledger's parts nest in one line of work, and a part
// begun beside them would end outside the part it began in.
func Unstaged(ctx context.Context) context.Context {
	return context.WithValue(ctx, stagesKey{}, (*Stages)(nil))
}

// Stage begins one named part of the request whose ledger the context
// carries, and the function it returns ends it. Under a context that carries
// no ledger it times nothing. Parts nest: one begun inside another ends
// before it.
func Stage(ctx context.Context, name string) func() {
	if ctx == nil {
		return func() {}
	}
	stages, _ := ctx.Value(stagesKey{}).(*Stages)
	if stages == nil {
		return func() {}
	}
	started, err := Counter()
	stages.mu.Lock()
	defer stages.mu.Unlock()
	stages.fail(err)
	depth := len(stages.open)
	stages.open = append(stages.open, openPart{started: started})
	return func() {
		ended, err := Counter()
		stages.mu.Lock()
		defer stages.mu.Unlock()
		stages.fail(err)
		if len(stages.open) != depth+1 {
			stages.fail(errors.New("measurement: part " + name + " did not end inside the part it began in"))
			return
		}
		opened := stages.open[depth]
		stages.open = stages.open[:depth]
		wall := ended - opened.started
		if depth > 0 {
			stages.open[depth-1].inner += wall
		}
		for index := range stages.parts {
			if stages.parts[index].name == name {
				stages.parts[index].wall += wall - opened.inner
				stages.parts[index].times++
				return
			}
		}
		stages.parts = append(stages.parts, part{name: name, wall: wall - opened.inner, times: 1})
	}
}

// fail keeps the first fault; the caller holds the lock.
func (s *Stages) fail(err error) {
	if s.err == nil {
		s.err = err
	}
}

// Report writes where a request of the given wall went: a line for each part
// that has ended, in the order the parts first ended, and one for the time no
// part names, which it returns. It reports a fault when a clock could not be
// read or parts overlapped: their walls are then no division of the request.
func (s *Stages) Report(w io.Writer, wall time.Duration) (time.Duration, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return 0, s.err
	}
	unnamed := wall
	for _, part := range s.parts {
		unnamed -= part.wall
		fmt.Fprintf(w, "  %-32s %7.2f s %5.1f%%  x%d\n", part.name, part.wall.Seconds(), share(part.wall, wall), part.times)
	}
	fmt.Fprintf(w, "  %-32s %7.2f s %5.1f%%\n", "(no part names)", unnamed.Seconds(), share(unnamed, wall))
	return unnamed, nil
}

// hundredths is how many of a share make the whole.
const hundredths = 100

// share is a part of a wall in hundredths.
func share(part, wall time.Duration) float64 {
	if wall <= 0 {
		return 0
	}
	return hundredths * part.Seconds() / wall.Seconds()
}
