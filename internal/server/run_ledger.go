package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"slices"
	"sync"

	"overgo/internal/artifact"
)

const (
	// ledgerCapacity is how many finished numbered submissions the server
	// remembers: a retry comes seconds after the request it repeats, so the
	// recent ones are the ones asked for. A submission that has not finished
	// is remembered until it has.
	ledgerCapacity = 256
	// maxRequestIDLength bounds the id a client numbers a submission with.
	maxRequestIDLength = 64
)

// errRequestIDReused refuses a request id the server already holds for a
// different submission.
var errRequestIDReused = errors.New("server: the request id belongs to another submission")

// errSubmissionAbandoned is what the callers waiting on a numbered submission
// are told when its owner never finished it, because submitting panicked.
var errSubmissionAbandoned = errors.New("server: the submission did not complete")

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
	// Over its capacity the ledger forgets the oldest submissions that have
	// finished. One still being submitted keeps its key however old it is: a
	// repeat of it must find it and wait, where a forgotten key would start
	// the submission a second time beside the first. Unfinished submissions
	// are as many as are in flight, so the ledger is over its capacity by no
	// more than those.
	for at := 0; len(ledger.order) > ledgerCapacity && at < len(ledger.order); {
		select {
		case <-ledger.entries[ledger.order[at]].done:
			delete(ledger.entries, ledger.order[at])
			ledger.order = slices.Delete(ledger.order, at, at+1)
		default:
			at++
		}
	}
	return entry, true, nil
}

// finish records what the owner's submission came to and releases the callers
// waiting on it. A submission that failed is forgotten, its place in the order
// with it, so the retry the failure invites is a submission of its own and
// neither takes a slot for nothing nor is evicted early by the failed one's
// stale place.
func (ledger *runLedger) finish(key string, entry *ledgerEntry, operation artifact.ID, err error) {
	entry.operation, entry.err = operation, err
	if err != nil {
		ledger.mu.Lock()
		if ledger.entries[key] == entry {
			delete(ledger.entries, key)
			ledger.order = slices.DeleteFunc(ledger.order, func(held string) bool { return held == key })
		}
		ledger.mu.Unlock()
	}
	close(entry.done)
}

// run submits as the owner of entry and finishes it however submit returns: a
// panic finishes it as abandoned before it goes on, so the callers waiting on
// the submission are released and a retry owns a submission of its own.
func (ledger *runLedger) run(key string, entry *ledgerEntry, submit func() (artifact.ID, error)) (artifact.ID, error) {
	finished := false
	defer func() {
		if !finished {
			ledger.finish(key, entry, artifact.ID{}, errSubmissionAbandoned)
		}
	}()
	operation, err := submit()
	finished = true
	ledger.finish(key, entry, operation, err)
	return operation, err
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
