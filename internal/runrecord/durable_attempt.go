package runrecord

import (
	"context"
	"errors"
	"fmt"
	"strconv"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	// DurableInvocationMediaType identifies the invocation count of a durable unit.
	DurableInvocationMediaType = "application/vnd.overgo.durable-invocation+json"
	// DurableInvocationSchema identifies the durable invocation schema.
	DurableInvocationSchema = "overgo/durable-invocation/v1"
	// DurableEntryMediaType identifies one entry of a durable attempt log.
	DurableEntryMediaType = "application/vnd.overgo.durable-attempt-entry+json"
	// DurableEntrySchema identifies the durable attempt entry schema.
	DurableEntrySchema = "overgo/durable-attempt-entry/v1"
	// DurableAttemptAliasRoot scopes the invocation count and log of every durable unit.
	DurableAttemptAliasRoot = "durable-attempt/"
)

// DurableEntryKind names what a log entry records.
type DurableEntryKind string

const (
	// DurableMemo is a completed side effect keyed by the caller.
	DurableMemo DurableEntryKind = "memo"
	// DurableWait is a completed wait addressed by ordinal.
	DurableWait DurableEntryKind = "wait"
	// DurableChild is a completed child attempt addressed by ordinal.
	DurableChild DurableEntryKind = "child"
)

var (
	// ErrStaleInvocation refuses an attempt superseded by a newer invocation.
	ErrStaleInvocation = errors.New("run record: durable attempt invocation is stale")
	// ErrEntryRecorded refuses a second, different completion of one entry.
	ErrEntryRecorded = errors.New("run record: durable attempt entry is already recorded")
)

// DurableAttempt is the identity of one invocation of a durable unit: the
// unit and its invocation count.
type DurableAttempt struct {
	Unit       artifact.ID
	Invocation uint32
}

// DurableInvocation records the current invocation count of a unit; each
// open advances it by one under compare-and-set.
type DurableInvocation struct {
	Version    uint16      `json:"version"`
	Unit       artifact.ID `json:"unit"`
	Invocation uint32      `json:"invocation"`
	Previous   artifact.ID `json:"previous,omitzero"`
	ID         artifact.ID `json:"-"`
}

// DurableEntry records one completion in a unit's log: a memo by key or a
// wait or child by ordinal, with the invocation that recorded it and the
// result every later invocation reads.
type DurableEntry struct {
	Version    uint16           `json:"version"`
	Unit       artifact.ID      `json:"unit"`
	Invocation uint32           `json:"invocation"`
	Kind       DurableEntryKind `json:"kind"`
	Key        string           `json:"key"`
	Result     artifact.ID      `json:"result"`
	ID         artifact.ID      `json:"-"`
}

var durableInvocationCodec = artifact.JSONDocumentCodec(
	"durable invocation", artifact.KindEvidence, DurableInvocationMediaType, DurableInvocationSchema,
	canonicalizeDurableInvocation, func(v DurableInvocation) artifact.ID { return v.ID },
	func(v *DurableInvocation, id artifact.ID) { v.ID = id }, nil,
)
var durableEntryCodec = artifact.JSONDocumentCodec(
	"durable attempt entry", artifact.KindEvidence, DurableEntryMediaType, DurableEntrySchema,
	canonicalizeDurableEntry, func(v DurableEntry) artifact.ID { return v.ID },
	func(v *DurableEntry, id artifact.ID) { v.ID = id }, nil,
)

func durableInvocationAlias(unit artifact.ID) string {
	return DurableAttemptAliasRoot + unit.String() + "/invocation"
}

func durableEntryAlias(unit artifact.ID, kind DurableEntryKind, key string) string {
	return DurableAttemptAliasRoot + unit.String() + "/" + string(kind) + "/" + key
}

// OpenDurableAttempt advances the unit's invocation count by one and returns
// the new attempt; every earlier attempt of the unit is stale from here on.
func OpenDurableAttempt(ctx context.Context, repository artifact.Repository, unit artifact.ID) (DurableAttempt, error) {
	if ctx == nil || repository == nil || !unit.Valid() {
		return DurableAttempt{}, errors.New("run record: durable attempt requires a repository and a unit")
	}
	previous, found, err := durableInvocationCodec.Resolve(ctx, repository, durableInvocationAlias(unit))
	if err != nil {
		return DurableAttempt{}, err
	}
	next := DurableInvocation{Unit: unit, Invocation: previous.Invocation + 1}
	alias := artifact.AliasBinding{Name: durableInvocationAlias(unit)}
	if found {
		next.Previous = previous.ID
		alias.Previous = artifact.IDPointer(previous.ID)
	}
	next, err = durableInvocationCodec.NewInitial(next)
	if err != nil {
		return DurableAttempt{}, err
	}
	content, err := durableInvocationCodec.Content(next)
	if err != nil {
		return DurableAttempt{}, err
	}
	alias.Target = next.ID
	var lineage []artifact.Lineage
	if found {
		lineage = artifact.DependencyLineage(next.ID, previous.ID)
	}
	batch, err := artifact.NewDocumentBatch("durable-attempt/"+next.ID.String(), []artifact.Content{content}, lineage, []artifact.AliasBinding{alias})
	if err != nil {
		return DurableAttempt{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return DurableAttempt{}, err
	}
	return DurableAttempt{Unit: unit, Invocation: next.Invocation}, nil
}

// Current refuses the attempt when the unit's invocation count has moved past it.
func (attempt DurableAttempt) Current(ctx context.Context, reader artifact.Reader) error {
	if ctx == nil || reader == nil || !attempt.Unit.Valid() || attempt.Invocation == 0 {
		return errors.New("run record: durable attempt is unbound")
	}
	current, found, err := durableInvocationCodec.Resolve(ctx, reader, durableInvocationAlias(attempt.Unit))
	if err != nil {
		return err
	}
	if !found || current.Invocation != attempt.Invocation {
		return fmt.Errorf("%w: unit %s invocation %d, current %d", ErrStaleInvocation, attempt.Unit, attempt.Invocation, current.Invocation)
	}
	return nil
}

// Lookup reads the recorded completion of one entry; a stale attempt is
// refused before it reads anything.
func (attempt DurableAttempt) Lookup(ctx context.Context, reader artifact.Reader, kind DurableEntryKind, key string) (DurableEntry, bool, error) {
	if err := attempt.Current(ctx, reader); err != nil {
		return DurableEntry{}, false, err
	}
	if err := validateDurableEntryAddress(kind, key); err != nil {
		return DurableEntry{}, false, err
	}
	return durableEntryCodec.Resolve(ctx, reader, durableEntryAlias(attempt.Unit, kind, key))
}

// Record writes one completion once: a recorded entry with the same result
// is returned as is, a different result is refused, and a stale attempt
// records nothing.
func (attempt DurableAttempt) Record(ctx context.Context, repository artifact.Repository, kind DurableEntryKind, key string, result artifact.ID) (DurableEntry, error) {
	if repository == nil || !result.Valid() {
		return DurableEntry{}, errors.New("run record: durable entry requires a repository and a result")
	}
	existing, found, err := attempt.Lookup(ctx, repository, kind, key)
	if err != nil {
		return DurableEntry{}, err
	}
	if found {
		if existing.Result != result {
			return existing, fmt.Errorf("%w: %s %s of unit %s holds %s", ErrEntryRecorded, kind, key, attempt.Unit, existing.Result)
		}
		return existing, nil
	}
	entry, err := durableEntryCodec.NewInitial(DurableEntry{Unit: attempt.Unit, Invocation: attempt.Invocation, Kind: kind, Key: key, Result: result})
	if err != nil {
		return DurableEntry{}, err
	}
	content, err := durableEntryCodec.Content(entry)
	if err != nil {
		return DurableEntry{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"durable-attempt/"+entry.ID.String(), []artifact.Content{content},
		artifact.DependencyLineage(entry.ID, result),
		[]artifact.AliasBinding{{Name: durableEntryAlias(attempt.Unit, kind, key), Target: entry.ID}},
	)
	if err != nil {
		return DurableEntry{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return DurableEntry{}, err
	}
	return entry, nil
}

func validateDurableEntryAddress(kind DurableEntryKind, key string) error {
	switch kind {
	case DurableMemo:
		if !validAgentScope(key) || key == "" {
			return errors.New("run record: durable memo key is invalid")
		}
	case DurableWait, DurableChild:
		if _, err := strconv.ParseUint(key, 10, 32); err != nil {
			return fmt.Errorf("run record: durable %s ordinal %q is invalid", kind, key)
		}
	default:
		return fmt.Errorf("run record: durable entry kind %q is unknown", kind)
	}
	return nil
}

func canonicalizeDurableInvocation(v *DurableInvocation) error {
	if v == nil || v.Version != artifact.InitialDocumentVersion || !v.Unit.Valid() || v.Invocation == 0 ||
		(v.Previous.Valid() && v.Previous.Kind() != artifact.KindEvidence) {
		return errors.New("run record: invalid durable invocation")
	}
	return nil
}

func canonicalizeDurableEntry(v *DurableEntry) error {
	if v == nil || v.Version != artifact.InitialDocumentVersion || !v.Unit.Valid() || v.Invocation == 0 || !v.Result.Valid() ||
		!textcheck.Bounded(v.Key, len(v.Key), "\x00\r\n") {
		return errors.New("run record: invalid durable attempt entry")
	}
	return validateDurableEntryAddress(v.Kind, v.Key)
}
