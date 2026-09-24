package server

import (
	"sync"
)

// A cursor names the publication boundary of this handler. Reconnection takes
// a fresh durable snapshot; artifact IDs retain identity across handler restarts.
type runtimeServingEvent struct {
	Cursor      uint64           `json:"cursor,string"`
	PublishFail uint64           `json:"publish_failures"`
	Activity    *servingActivity `json:"activity,omitempty"`
}

// servingEvents orders serving publication: its mutex spans the durable
// write and the cursor that names it, so a snapshot that reads the cursor
// first holds every record at or before it. The event hub carries each
// publication to the streams.
type servingEvents struct {
	mu     sync.Mutex
	cursor uint64
}
