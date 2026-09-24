package server

import (
	"errors"
	"sync"

	"overgo/internal/operation"
)

// hubResync asks a subscriber that fell a queue behind to take fresh
// snapshots in place of the events it missed.
const hubResync = "resync"

var errEventHubUnavailable = errors.New("operation: event subscription unavailable")

// hubEvent is one named event as every runtime stream sends it.
type hubEvent struct {
	name  string
	value any
}

// eventHub fans the workbench's events out to every runtime stream: each
// operation transition, serving publication and download change is published
// once, and each stream reads one bounded queue. A stream a queue behind is
// sent a resync, which it answers with fresh snapshots, so a slow reader
// never loses a transition (a terminal operation state reaches it in the
// snapshot). Its lock is the last any publisher takes, so publishing under
// another owner's lock cannot deadlock.
type eventHub struct {
	mu          sync.Mutex
	subscribers map[chan hubEvent]struct{}
	closed      bool
	// sessionsChanged coalesces operation transitions into one sessions
	// snapshot for every stream, taken outside the operation manager's lock.
	sessionsChanged chan struct{}
	stop            chan struct{}
	pumping         sync.WaitGroup
}

func newEventHub() *eventHub {
	return &eventHub{subscribers: map[chan hubEvent]struct{}{}, sessionsChanged: make(chan struct{}, 1), stop: make(chan struct{})}
}

// subscribe admits one stream under limit, which also sizes its queue.
func (hub *eventHub) subscribe(limit int) (<-chan hubEvent, func(), error) {
	hub.mu.Lock()
	defer hub.mu.Unlock()
	if hub.closed || len(hub.subscribers) >= limit {
		return nil, nil, errEventHubUnavailable
	}
	events := make(chan hubEvent, limit)
	hub.subscribers[events] = struct{}{}
	return events, func() {
		hub.mu.Lock()
		defer hub.mu.Unlock()
		if _, found := hub.subscribers[events]; found {
			delete(hub.subscribers, events)
			close(events)
		}
	}, nil
}

func (hub *eventHub) publish(name string, value any) {
	if hub == nil {
		return
	}
	hub.mu.Lock()
	defer hub.mu.Unlock()
	event := hubEvent{name, value}
	for events := range hub.subscribers {
		select {
		case events <- event:
		default:
			// Only this locked publisher writes a queue; its reader drains
			// concurrently, so the clear uses nonblocking receives.
		clear:
			for {
				select {
				case <-events:
				default:
					events <- hubEvent{name: hubResync}
					break clear
				}
			}
		}
	}
}

// observeOperation publishes a transition in the manager's order (it runs
// under the manager's lock) and asks for the sessions it changed.
func (hub *eventHub) observeOperation(event operation.Event) {
	hub.publish("operation", event)
	select {
	case hub.sessionsChanged <- struct{}{}:
	default:
	}
}

// pump publishes the sessions snapshot once per coalesced run of operation
// transitions, until the hub closes.
func (hub *eventHub) pump(sessions func() runtimeSessionsResponse) {
	hub.pumping.Go(func() {
		for {
			select {
			case <-hub.stop:
				return
			case <-hub.sessionsChanged:
				hub.publish("runtime.sessions", sessions())
			}
		}
	})
}

// close ends every stream's queue and the pump.
func (hub *eventHub) close() {
	if hub == nil {
		return
	}
	hub.mu.Lock()
	if hub.closed {
		hub.mu.Unlock()
		return
	}
	hub.closed = true
	for events := range hub.subscribers {
		delete(hub.subscribers, events)
		close(events)
	}
	hub.mu.Unlock()
	close(hub.stop)
	hub.pumping.Wait()
}
