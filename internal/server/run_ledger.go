package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"sync"

	"overgo/internal/artifact"
)

const (
	// ledgerCapacity is how many numbered submissions the server remembers: a
	// retry comes seconds after the request it repeats, so the recent ones are
	// the ones asked for.
	ledgerCapacity = 256
	// maxRequestIDLength bounds the id a client numbers a submission with.
	maxRequestIDLength = 64
)

// errRequestIDReused refuses a request id the server already holds for a
// different submission.
var errRequestIDReused = errors.New("server: the request id belongs to another submission")

// runLedger remembers the submissions a client numbered, so a repeat of one, a
// retry after a response was lost, finds the operation the first started and
// starts none of its own.
type runLedger struct {
	mu      sync.Mutex
	entries map[string]*ledgerEntry
	order   []string
}

// ledgerEntry is one numbered submission: what it was, and, once it is
// submitted, the operation it started or why it did not.
type ledgerEntry struct {
	fingerprint [sha256.Size]byte
	done        chan struct{}
	operation   artifact.ID
	err         error
}

// validRequestID reports whether id is a token a client may number a submission
// with: letters, digits, hyphen and underscore, at most maxRequestIDLength.
func validRequestID(id string) bool {
	if id == "" || len(id) > maxRequestIDLength {
		return false
	}
	for _, character := range id {
		if !(character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' || character >= '0' && character <= '9' || character == '-' || character == '_') {
			return false
		}
	}
	return true
}

// submissionFingerprint identifies what a submission asks for, apart from the id
// it is numbered with.
func submissionFingerprint(body workflowRequest) ([sha256.Size]byte, error) {
	body.Request = ""
	var compact bytes.Buffer
	if err := json.Compact(&compact, body.Input); err == nil {
		body.Input = compact.Bytes()
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		return [sha256.Size]byte{}, err
	}
	return sha256.Sum256(encoded), nil
}

// begin claims the key for a submission with the fingerprint. The caller that
// finds no entry owns the submission and must finish it; one that finds an entry
// for the same submission gets it to wait on, and one that finds the key held for
// a different submission is refused.
func (ledger *runLedger) begin(key string, fingerprint [sha256.Size]byte) (entry *ledgerEntry, owner bool, err error) {
	ledger.mu.Lock()
	defer ledger.mu.Unlock()
	if ledger.entries == nil {
		ledger.entries = map[string]*ledgerEntry{}
	}
	if held, found := ledger.entries[key]; found {
		if held.fingerprint != fingerprint {
			return nil, false, errRequestIDReused
		}
		return held, false, nil
	}
	entry = &ledgerEntry{fingerprint: fingerprint, done: make(chan struct{})}
	ledger.entries[key] = entry
	ledger.order = append(ledger.order, key)
	if len(ledger.order) > ledgerCapacity {
		delete(ledger.entries, ledger.order[0])
		ledger.order = ledger.order[1:]
	}
	return entry, true, nil
}

// finish records what the owner's submission came to and releases the callers
// waiting on it. A submission that failed is forgotten, so the retry the failure
// invites is a submission of its own.
func (ledger *runLedger) finish(key string, entry *ledgerEntry, operation artifact.ID, err error) {
	entry.operation, entry.err = operation, err
	if err != nil {
		ledger.mu.Lock()
		if ledger.entries[key] == entry {
			delete(ledger.entries, key)
		}
		ledger.mu.Unlock()
	}
	close(entry.done)
}

// wait is the operation a numbered submission started, once it has.
func (entry *ledgerEntry) wait(ctx context.Context) (artifact.ID, error) {
	select {
	case <-entry.done:
		return entry.operation, entry.err
	case <-ctx.Done():
		return artifact.ID{}, context.Cause(ctx)
	}
}
