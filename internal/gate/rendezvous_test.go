package gate

import "sync"

// rendezvous holds each expected arrival until every expected name has
// arrived: checks that must run in one wave prove it by meeting, and a
// scheduler that ran them one after another never completes the meeting,
// which the test binary then reports.
type rendezvous struct {
	mu       sync.Mutex
	expected map[string]bool
	arrived  int
	open     chan struct{}
}

func newRendezvous(names ...string) *rendezvous {
	expected := make(map[string]bool, len(names))
	for _, name := range names {
		expected[name] = true
	}
	return &rendezvous{expected: expected, open: make(chan struct{})}
}

// meet blocks an expected name until the whole party has arrived; other
// names pass through.
func (r *rendezvous) meet(name string) {
	r.mu.Lock()
	if !r.expected[name] {
		r.mu.Unlock()
		return
	}
	r.arrived++
	if r.arrived == len(r.expected) {
		close(r.open)
	}
	r.mu.Unlock()
	<-r.open
}
