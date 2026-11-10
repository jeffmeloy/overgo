package capabilityruntime

import (
	"context"
	"errors"
	"sync"
)

// idleReleaser is a session director that can give back the memory its idle sessions hold.
type idleReleaser interface {
	releaseIdle(context.Context) (int, error)
}

// idleSessions are the session directors of this process, made and not yet closed: a media model's sessions stay
// resident for the life of its director, which is the process's, so another model the process opens finds the card
// held by them unless they give it back.
var idleSessions = struct {
	sync.Mutex
	directors map[idleReleaser]struct{}
}{directors: map[idleReleaser]struct{}{}}

// holdsSessions records director among the process's session directors, and forgets records it again.
func holdsSessions(director idleReleaser, holds bool) {
	idleSessions.Lock()
	defer idleSessions.Unlock()
	if holds {
		idleSessions.directors[director] = struct{}{}
		return
	}
	delete(idleSessions.directors, director)
}

// ReleaseIdleSessions closes every session no lease holds, of every session director of this process, as the
// director's own eviction closes one, and says how many it closed. A model about to open calls it so it is not refused
// room the idle sessions hold (judge-speed/media-residency: beside two clips the Wan student had drawn, a judge's
// projector found 10 MB of the card free for a 72 MB encode); each session loads again on its next request, and a
// session in use is left as it is.
func ReleaseIdleSessions(ctx context.Context) (int, error) {
	idleSessions.Lock()
	directors := make([]idleReleaser, 0, len(idleSessions.directors))
	for director := range idleSessions.directors {
		directors = append(directors, director)
	}
	idleSessions.Unlock()
	released := 0
	var result error
	for _, director := range directors {
		closed, err := director.releaseIdle(ctx)
		released += closed
		result = errors.Join(result, err)
	}
	return released, result
}

// releaseIdle closes every session no lease holds, as eviction closes one: it is taken out of the director at once and
// counted as retiring until its Close returns, so a request for it loads it again beside nothing.
func (c *ModelSessionDirector[Input, Model, Output]) releaseIdle(ctx context.Context) (int, error) {
	c.mu.Lock()
	var idle []*sessionEntry[Model]
	for key, entry := range c.entries {
		select {
		case <-entry.ready:
		default:
			continue
		}
		if entry.borrowers != 0 || entry.dead {
			continue
		}
		entry.dead, entry.closing, entry.retiring = true, true, true
		delete(c.entries, key)
		c.evictions++
		c.retiring++
		idle = append(idle, entry)
	}
	if len(idle) != 0 {
		c.notify()
	}
	c.mu.Unlock()
	var result error
	for _, entry := range idle {
		result = errors.Join(result, c.finishRetirement(context.WithoutCancel(ctx), entry))
	}
	return len(idle), result
}
