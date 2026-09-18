package server

import (
	"sync"

	"overgo/internal/runrecord"
)

// A cursor names the publication boundary of this handler. Reconnection takes
// a fresh durable snapshot; artifact IDs retain identity across handler restarts.
type runtimeServingEvent struct {
	Cursor      uint64           `json:"cursor,string"`
	PublishFail uint64           `json:"publish_failures"`
	Activity    *servingActivity `json:"activity,omitempty"`
}

// servingEvents is the notification state of the existing runtime SSE route.
// Its mutex spans durable publication and snapshot capture so a snapshot cursor
// cannot skip a record or reinsert an older record from a buffered event.
type servingEvents struct {
	mu      sync.Mutex
	cursor  uint64
	watches map[chan runtimeServingEvent]struct{}
}

// Runtime streams acquire their bounded operation subscription first. Its
// admission limit also bounds the number of these paired serving subscribers.
func (events *servingEvents) subscribe(limit int) (<-chan runtimeServingEvent, func()) {
	events.mu.Lock()
	defer events.mu.Unlock()
	if events.watches == nil {
		events.watches = make(map[chan runtimeServingEvent]struct{})
	}
	channel := make(chan runtimeServingEvent, limit)
	events.watches[channel] = struct{}{}
	return channel, func() {
		events.mu.Lock()
		defer events.mu.Unlock()
		if _, exists := events.watches[channel]; exists {
			delete(events.watches, channel)
			close(channel)
		}
	}
}

// Called under mu only after durable publication. The record codec owns its
// slices, so all subscribers can retain the same immutable observation.
func (events *servingEvents) publishLocked(observation runrecord.ServingObservation, publishFailures uint64) {
	events.cursor++
	if len(events.watches) == 0 {
		return
	}
	event := runtimeServingEvent{
		Cursor:      events.cursor,
		PublishFail: publishFailures,
		Activity:    &servingActivity{ID: observation.ID, ServingObservation: observation},
	}
	for channel := range events.watches {
		select {
		case channel <- event:
		default:
			// An empty activity explicitly requests a new durable snapshot.
			// Only this locked publisher writes the buffer; readers can drain
			// concurrently, so the clear uses nonblocking receives.
		clear:
			for {
				select {
				case <-channel:
				default:
					channel <- runtimeServingEvent{Cursor: events.cursor}
					break clear
				}
			}
		}
	}
}
