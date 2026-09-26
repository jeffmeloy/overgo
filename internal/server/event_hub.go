package server

import (
	"errors"
	"net/http"
	"slices"
	"strings"
	"sync"

	"overgo/internal/operation"
)

// hubResync asks a subscriber that fell a queue behind to take fresh
// snapshots in place of the events it missed.
const hubResync = "resync"

// hubRecordsChanged tells every stream the store gained a record its
// activity snapshot projects (a stage receipt, a decision, an interaction),
// which serving events do not carry; each stream answers with a fresh
// activity snapshot.
const hubRecordsChanged = "runtime.records"

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
	// A workflow has written its stage receipts by the time it settles.
	if slices.Contains(settledStates, event.Status.State) {
		hub.publish(hubRecordsChanged, nil)
	}
	select {
	case hub.sessionsChanged <- struct{}{}:
	default:
	}
}

// settledStates are the operation states that stop its work: blocked on a
// decision, or ended.
var settledStates = []operation.State{operation.StateBlocked, operation.StateCompleted, operation.StateFailed, operation.StateCancelled}

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

// workspaceChange names the inventory a call changed ("/peers").
type workspaceChange struct {
	Inventory string `json:"inventory"`
}

// statusRecorder keeps the status a route answered.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

// WriteHeader keeps the status on its way out.
func (recorder *statusRecorder) WriteHeader(status int) {
	recorder.status = status
	recorder.ResponseWriter.WriteHeader(status)
}

// Unwrap hands http.ResponseController the writer underneath.
func (recorder *statusRecorder) Unwrap() http.ResponseWriter { return recorder.ResponseWriter }

// serveInventoryChange serves an inventory route and, when a call other than
// a GET succeeds, tells every workspace which inventory it changed.
func (h *Handler) serveInventoryChange(route routeDescriptor, response http.ResponseWriter, request *http.Request) {
	recorder := &statusRecorder{ResponseWriter: response, status: http.StatusOK}
	route.Handler(h, recorder, request)
	if request.Method != http.MethodGet && recorder.status < http.StatusMultipleChoices {
		segment, _, _ := strings.Cut(strings.TrimPrefix(route.Path, "/"), "/")
		h.events.publish("workspace.changed", workspaceChange{Inventory: "/" + segment})
	}
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
