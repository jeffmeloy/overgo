package overgodb

import (
	"errors"
	"fmt"
	"strings"

	"overgo/internal/artifact"
)

// ErrProducerRefused reports a delta that introduces an artifact kind
// recorded under a producer other than the one committing it.
var ErrProducerRefused = errors.New("overgodb: producer refused")

// Producer is the capability to commit the artifact kinds recorded under
// its name. Holding the store API is not holding a Producer: Commit carries
// none, and the frame of every CommitAs records whose it was.
type Producer struct {
	name string
	// transplant is Rebuild's alone: it re-batches history that already
	// carries its producers into a new chain and introduces nothing of its
	// own, so the table does not bind it. No mint sets it.
	transplant bool
}

// NewProducer mints the capability for name. The storage authority audit
// pins every mint site, so a second producer of a guarded kind is a reviewed
// change to that audit, never a caller's choice.
func NewProducer(name string) Producer { return Producer{name: name} }

// producerKinds records which producer alone may commit which artifact
// kinds, by schema prefix so a version bump stays guarded. The table lives
// at the boundary, not with the schema owners, so a process that never links
// an owner is held to it all the same.
var producerKinds = []struct{ prefix, producer string }{
	// The kinds only the gate mints. The lifecycle is what the plan
	// completion authority discovers a landing through, and an attempt is
	// what it counts: exactly one, so a forged second would wedge it. Gate
	// results are absent on purpose. Model intake, the remote provider and
	// vqaparity mint them for their own verification runs, and that is safe:
	// the authority accepts a result only through a lifecycle finalization
	// introduced atomically with it, which no other producer can write.
	{"overgo/gate-attempt/", "gate"},
	{"overgo/gate-batch-evidence/", "gate"},
	{"overgo/gate-lane-obligation/", "gate"},
	{"overgo/gate-lifecycle/", "gate"},
	{"overgo/gate-package-receipt/", "gate"},
	{"overgo/gate-selection-cause/", "gate"},
	{"overgo/gate-suite-cost/", "gate"},
	{operationProofKind, "store-precheck"},
}

// admitProducer holds a delta to the table. It reads the delta, not the
// request: naming an artifact the store already holds mints nothing.
func admitProducer(delta artifact.Batch) error {
	for _, descriptor := range delta.Artifacts {
		for _, kind := range producerKinds {
			if strings.HasPrefix(descriptor.Schema, kind.prefix) && delta.Producer != kind.producer {
				return fmt.Errorf("%w: %s is committed by producer %q, and this batch is committed by %q",
					ErrProducerRefused, descriptor.Schema, kind.producer, delta.Producer)
			}
		}
	}
	return nil
}
