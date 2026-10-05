package server

import (
	"cmp"
	"context"
	"net/http"
	"slices"
	"sync"
	"time"
)

// requestMonitorCapacity bounds the finished requests the monitor holds, the newest. A request
// still being answered is held until it ends, so the count in flight is never cut short.
const requestMonitorCapacity = 256

// monitoredRequest is one API request from the route that took it to the status it answered.
type monitoredRequest struct {
	id               uint64
	method, route    string
	started          time.Time
	duration         time.Duration
	status           int
	promptTokens     uint64
	completionTokens uint64
}

// requestMonitor keeps what the server answered, for the Requests tab: the requests in flight and
// the last finished ones, with the tokens a generation made under each. Its zero value is ready.
type requestMonitor struct {
	mu       sync.Mutex
	total    uint64
	active   map[uint64]*monitoredRequest
	finished []*monitoredRequest
}

// monitoredKey carries a request's record on its context, so generate can add the tokens it made.
type monitoredKey struct{}

// begin records a request in flight and answers its context and the function that ends it, which
// reads the status the route answered from the recorder.
func (m *requestMonitor) begin(request *http.Request, route string) (*http.Request, func(*statusRecorder)) {
	m.mu.Lock()
	m.total++
	record := &monitoredRequest{id: m.total, method: request.Method, route: route, started: time.Now()}
	if m.active == nil {
		m.active = map[uint64]*monitoredRequest{}
	}
	m.active[record.id] = record
	m.mu.Unlock()
	return request.WithContext(context.WithValue(request.Context(), monitoredKey{}, record)), func(recorder *statusRecorder) {
		m.mu.Lock()
		defer m.mu.Unlock()
		record.duration = time.Since(record.started)
		record.status = recorder.status
		delete(m.active, record.id)
		m.finished = append(m.finished, record)
		if len(m.finished) > requestMonitorCapacity {
			m.finished = slices.Delete(m.finished, 0, len(m.finished)-requestMonitorCapacity)
		}
	}
}

// addTokens adds the tokens one generation made to the request its context belongs to.
func (m *requestMonitor) addTokens(ctx context.Context, prompt, completion uint64) {
	record, monitored := ctx.Value(monitoredKey{}).(*monitoredRequest)
	if !monitored {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	record.promptTokens += prompt
	record.completionTokens += completion
}

// requestEntry is a request as the Requests tab reads it: a request in flight has no status, and
// its duration is the time it has run so far.
type requestEntry struct {
	ID               uint64  `json:"id"`
	Method           string  `json:"method"`
	Route            string  `json:"route"`
	Status           int     `json:"status,omitzero"`
	StartedUnixMS    int64   `json:"started_ms"`
	DurationMS       float64 `json:"duration_ms"`
	InFlight         bool    `json:"in_flight,omitzero"`
	PromptTokens     uint64  `json:"prompt_tokens,omitzero"`
	CompletionTokens uint64  `json:"completion_tokens,omitzero"`
}

type requestLogResponse struct {
	// Total counts every request recorded since the server started; Limit is how many finished ones are held.
	Total    uint64         `json:"total"`
	Limit    int            `json:"limit"`
	Requests []requestEntry `json:"requests"`
}

// snapshot answers the requests in flight and the finished ones held, newest first.
func (m *requestMonitor) snapshot() requestLogResponse {
	m.mu.Lock()
	defer m.mu.Unlock()
	answer := requestLogResponse{Total: m.total, Limit: requestMonitorCapacity, Requests: make([]requestEntry, 0, len(m.active)+len(m.finished))}
	for _, record := range m.finished {
		answer.Requests = append(answer.Requests, entryOf(record, false))
	}
	for _, record := range m.active {
		answer.Requests = append(answer.Requests, entryOf(record, true))
	}
	slices.SortFunc(answer.Requests, func(a, b requestEntry) int { return cmp.Compare(b.ID, a.ID) })
	return answer
}

func entryOf(record *monitoredRequest, inFlight bool) requestEntry {
	entry := requestEntry{
		ID: record.id, Method: record.method, Route: record.route, StartedUnixMS: record.started.UnixMilli(),
		DurationMS: float64(record.duration) / float64(time.Millisecond), PromptTokens: record.promptTokens, CompletionTokens: record.completionTokens,
	}
	if inFlight {
		entry.InFlight = true
		entry.DurationMS = float64(time.Since(record.started)) / float64(time.Millisecond)
	} else {
		entry.Status = record.status
	}
	return entry
}

// requestLog answers GET /requests with the monitor's ring. The route is left out of the ring, so
// the tab's own polling never fills it.
func (h *Handler) requestLog(response http.ResponseWriter, request *http.Request) {
	writeJSON(response, http.StatusOK, h.requests.snapshot())
}
