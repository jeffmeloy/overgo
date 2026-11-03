package processmeasure

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"sync"
	"time"
)

// Part is one named part of a request: the wall that was its own, over the
// number of times the request went through it, and, where its ledger reads the
// device, the most device memory the process held while it was open, what was
// begun inside it included.
type Part struct {
	Name            string        `json:"name"`
	Wall            time.Duration `json:"wall_ns"`
	Times           int           `json:"times"`
	PeakDeviceBytes uint64        `json:"peak_device_bytes,omitzero"`
}

// Stages is where one request's time went: the wall of each named part of it.
// A part's wall is its own. What a part begun inside it took belongs to that
// part and not also to the outer one, so no moment is counted twice, and the
// parts' sum is the time the request spent inside a named part: what is left
// of the request's wall is time nothing names.
type Stages struct {
	mu    sync.Mutex
	open  []openPart
	parts []Part
	err   error
	// peak and reset read and lower a high-water mark of the device memory the
	// process holds, where the ledger reads the device (WithDeviceStages).
	peak  func() uint64
	reset func()
}

type openPart struct {
	started, inner time.Duration
	peak           uint64
}

type stagesKey struct{}

// WithStages gives a request a ledger of its parts: every Stage begun under
// the context returned is timed into it.
func WithStages(ctx context.Context) (context.Context, *Stages) {
	stages := &Stages{}
	return context.WithValue(ctx, stagesKey{}, stages), stages
}

// WithDeviceStages gives a request a ledger of its parts that also reads the
// device memory each held: peak reads the most the process has held since reset
// last lowered it, and a part's peak is read and lowered as each part begins and
// ends.
func WithDeviceStages(ctx context.Context, peak func() uint64, reset func()) (context.Context, *Stages) {
	ctx, stages := WithStages(ctx)
	stages.peak, stages.reset = peak, reset
	return ctx, stages
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
	stages.held()
	depth := len(stages.open)
	stages.open = append(stages.open, openPart{started: started})
	return func() {
		ended, err := Counter()
		stages.mu.Lock()
		defer stages.mu.Unlock()
		stages.fail(err)
		stages.held()
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
		if depth > 0 {
			stages.open[depth-1].peak = max(stages.open[depth-1].peak, opened.peak)
		}
		for index := range stages.parts {
			if stages.parts[index].Name == name {
				stages.parts[index].Wall += wall - opened.inner
				stages.parts[index].Times++
				stages.parts[index].PeakDeviceBytes = max(stages.parts[index].PeakDeviceBytes, opened.peak)
				return
			}
		}
		stages.parts = append(stages.parts, Part{Name: name, Wall: wall - opened.inner, Times: 1, PeakDeviceBytes: opened.peak})
	}
}

// held gives the most device memory held since a part last began or ended to
// the innermost open part, whose peak its outer parts take when it ends, and
// lowers the mark; the caller holds the lock.
func (s *Stages) held() {
	if s.peak == nil {
		return
	}
	if depth := len(s.open); depth > 0 {
		s.open[depth-1].peak = max(s.open[depth-1].peak, s.peak())
	}
	s.reset()
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
	parts, err := s.Parts()
	if err != nil {
		return 0, err
	}
	unnamed := wall
	for _, part := range parts {
		unnamed -= part.Wall
		fmt.Fprintf(w, "  %-32s %7.2f s %5.1f%%  x%d", part.Name, part.Wall.Seconds(), share(part.Wall, wall), part.Times)
		if s.peak != nil {
			fmt.Fprintf(w, "  %6.2f GiB at most", float64(part.PeakDeviceBytes)/bytesPerGiB)
		}
		fmt.Fprintln(w)
	}
	fmt.Fprintf(w, "  %-32s %7.2f s %5.1f%%\n", "(no part names)", unnamed.Seconds(), share(unnamed, wall))
	return unnamed, nil
}

// Parts are the parts that have ended, in the order they first ended, or the
// fault that makes them no division of the request: a clock that could not be
// read, or parts that overlapped.
func (s *Stages) Parts() ([]Part, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.parts), s.err
}

// bytesPerGiB is a gibibyte's bytes, the unit a part's peak is reported in.
const bytesPerGiB = 1 << 30

// hundredths is how many of a share make the whole.
const hundredths = 100

// share is a part of a wall in hundredths.
func share(part, wall time.Duration) float64 {
	if wall <= 0 {
		return 0
	}
	return hundredths * part.Seconds() / wall.Seconds()
}
