package runrecord

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

const (
	// LoopObligationMediaType identifies durable loop obligations.
	LoopObligationMediaType = "application/vnd.overgo.loop-obligation+json"
	// LoopObligationSchema is the obligation document version.
	LoopObligationSchema = "overgo/loop-obligation/v1"
	// LoopObligationCompletionMediaType identifies completions.
	LoopObligationCompletionMediaType = "application/vnd.overgo.loop-obligation-completion+json"
	// LoopObligationCompletionSchema is the completion document version.
	LoopObligationCompletionSchema = "overgo/loop-obligation-completion/v1"
	// LoopObligationCompletionAliasRoot scopes one completion per
	// obligation; an obligation whose completion alias does not resolve
	// is open, with no inferred state.
	LoopObligationCompletionAliasRoot = "loop/obligation/complete/"
	// loopObligationMaxRecords bounds one replay's obligation scan.
	loopObligationMaxRecords = 4096
)

// LoopObligationKind is one canonical follow-up the loop can owe.
type LoopObligationKind string

const (
	// ObligationRecoveryDispatch owes a recovery dispatch for lost work.
	ObligationRecoveryDispatch LoopObligationKind = "recovery-dispatch"
	// ObligationReEvaluation owes a promised re-evaluation.
	ObligationReEvaluation LoopObligationKind = "re-evaluation"
	// ObligationRollbackNotification owes a rollback notification.
	ObligationRollbackNotification LoopObligationKind = "rollback-notification"
)

// CanonicalLoopObligationKinds is the closed kind vocabulary in
// declaration order.
var CanonicalLoopObligationKinds = []LoopObligationKind{
	ObligationRecoveryDispatch, ObligationReEvaluation, ObligationRollbackNotification,
}

// CompletionPredicateKind selects how a completion predicate is
// evaluated against the store.
type CompletionPredicateKind string

const (
	// PredicateAliasResolves completes when the named alias resolves.
	PredicateAliasResolves CompletionPredicateKind = "alias-resolves"
	// PredicateArtifactExists completes when the named artifact exists.
	PredicateArtifactExists CompletionPredicateKind = "artifact-exists"
)

// CompletionPredicate is the explicit, mechanically evaluable condition
// under which an obligation is complete. There is no free-text
// predicate: completeness is always a store query.
type CompletionPredicate struct {
	Kind     CompletionPredicateKind `json:"kind"`
	Alias    string                  `json:"alias,omitempty"`
	Artifact artifact.ID             `json:"artifact,omitzero"`
}

// Evaluate answers the predicate against committed facts, returning
// the evidence artifact that satisfies it when it holds.
func (predicate CompletionPredicate) Evaluate(ctx context.Context, reader artifact.Reader) (artifact.ID, bool, error) {
	switch predicate.Kind {
	case PredicateAliasResolves:
		return artifact.ResolveAlias(ctx, reader, predicate.Alias)
	case PredicateArtifactExists:
		_, found, err := reader.Artifact(ctx, predicate.Artifact)
		return predicate.Artifact, found, err
	}
	return artifact.ID{}, false, errors.New("run record: foreign completion predicate")
}

// LoopObligation is a durable follow-up the improvement loop owes: a
// typed record whose incompleteness is a query, not a memory. Replay
// finds it in the store every supervision round, so a crash between
// admitting work and following up cannot silently drop it.
type LoopObligation struct {
	Version       uint16              `json:"version"`
	Kind          LoopObligationKind  `json:"kind"`
	Subject       artifact.ID         `json:"subject"`
	Predicate     CompletionPredicate `json:"predicate"`
	CreatedUnixNS int64               `json:"created_unix_ns"`
	ID            artifact.ID         `json:"-"`
}

// LoopObligationCompletion closes one obligation with the evidence
// that satisfied its predicate.
type LoopObligationCompletion struct {
	Version         uint16      `json:"version"`
	Obligation      artifact.ID `json:"obligation"`
	Evidence        artifact.ID `json:"evidence"`
	CompletedUnixNS int64       `json:"completed_unix_ns"`
	ID              artifact.ID `json:"-"`
}

var loopObligationCodec = artifact.JSONDocumentCodec(
	"loop obligation", artifact.KindEvidence, LoopObligationMediaType, LoopObligationSchema,
	canonicalizeLoopObligation,
	func(value LoopObligation) artifact.ID { return value.ID },
	func(value *LoopObligation, id artifact.ID) { value.ID = id },
	func(value LoopObligation) LoopObligation { return value },
)

var loopObligationCompletionCodec = artifact.JSONDocumentCodec(
	"loop obligation completion", artifact.KindEvidence, LoopObligationCompletionMediaType, LoopObligationCompletionSchema,
	canonicalizeLoopObligationCompletion,
	func(value LoopObligationCompletion) artifact.ID { return value.ID },
	func(value *LoopObligationCompletion, id artifact.ID) { value.ID = id },
	func(value LoopObligationCompletion) LoopObligationCompletion { return value },
)

// PublishLoopObligation commits one durable follow-up.
func PublishLoopObligation(ctx context.Context, repository artifact.Repository, value LoopObligation) (LoopObligation, error) {
	if ctx == nil || repository == nil {
		return LoopObligation{}, errors.New("run record: loop obligation repository is absent")
	}
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	identified, err := loopObligationCodec.New(value)
	if err != nil {
		return LoopObligation{}, err
	}
	return commitFailureDocument(ctx, repository, loopObligationCodec,
		"loop/obligation/"+identified.ID.String(), identified,
		artifact.DependencyLineage(identified.ID, identified.Subject), nil)
}

// LoopObligationReplay is one unconditional supervision-round replay:
// obligations whose predicates now hold were completed durably, and
// Due carries every follow-up still owed, oldest first.
type LoopObligationReplay struct {
	Completed []LoopObligationCompletion
	Due       []LoopObligation
}

// ReplayLoopObligations queries every committed obligation, evaluates
// each open obligation's completion predicate against the store, and
// durably completes the satisfied ones. It runs from committed facts
// alone, so obligations lost to a crash are repaired by the replay
// itself; callers run it every supervision round whether or not
// anything else fired.
func ReplayLoopObligations(ctx context.Context, store *overgodb.Store, nowUnixNS int64) (LoopObligationReplay, error) {
	if ctx == nil || store == nil || nowUnixNS <= 0 {
		return LoopObligationReplay{}, errors.New("run record: obligation replay requires the store and the clock")
	}
	result, err := store.Query(ctx, overgodb.Query{
		Kind: artifact.KindEvidence, MediaType: LoopObligationMediaType, Schema: LoopObligationSchema,
		MaxResults: loopObligationMaxRecords, Projection: overgodb.ProjectContentPresence,
	})
	if err != nil {
		return LoopObligationReplay{}, err
	}
	obligations := make([]LoopObligation, 0, len(result.Contents))
	for _, content := range result.Contents {
		obligation, found, err := loopObligationCodec.Read(ctx, store, content.Artifact)
		if err != nil {
			return LoopObligationReplay{}, err
		}
		if found {
			obligations = append(obligations, obligation)
		}
	}
	slices.SortFunc(obligations, func(left, right LoopObligation) int {
		if left.CreatedUnixNS != right.CreatedUnixNS {
			return int(left.CreatedUnixNS - right.CreatedUnixNS)
		}
		return artifact.CompareID(left.ID, right.ID)
	})
	replay := LoopObligationReplay{}
	for _, obligation := range obligations {
		if _, closed, err := loopObligationCompletionCodec.Resolve(
			ctx, store, LoopObligationCompletionAliasRoot+obligation.ID.String()); err != nil {
			return LoopObligationReplay{}, err
		} else if closed {
			continue
		}
		evidence, satisfied, err := obligation.Predicate.Evaluate(ctx, store)
		if err != nil {
			return LoopObligationReplay{}, err
		}
		if !satisfied {
			replay.Due = append(replay.Due, obligation)
			continue
		}
		completion := LoopObligationCompletion{
			Version: artifact.InitialDocumentVersion, Obligation: obligation.ID,
			Evidence: evidence, CompletedUnixNS: nowUnixNS,
		}
		identified, err := loopObligationCompletionCodec.New(completion)
		if err != nil {
			return LoopObligationReplay{}, err
		}
		identified, err = commitFailureDocument(ctx, store, loopObligationCompletionCodec,
			"loop/obligation/complete/"+identified.ID.String(), identified,
			artifact.DependencyLineage(identified.ID, identified.Obligation, identified.Evidence),
			[]artifact.AliasBinding{{Name: LoopObligationCompletionAliasRoot + obligation.ID.String(), Target: identified.ID}})
		if err != nil {
			return LoopObligationReplay{}, err
		}
		replay.Completed = append(replay.Completed, identified)
	}
	return replay, nil
}

func canonicalizeLoopObligation(value *LoopObligation) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		!slices.Contains(CanonicalLoopObligationKinds, value.Kind) ||
		value.Subject.Kind() != artifact.KindEvidence || value.CreatedUnixNS <= 0 {
		return errors.New("run record: invalid loop obligation")
	}
	predicate := value.Predicate
	switch predicate.Kind {
	case PredicateAliasResolves:
		if predicate.Alias == "" || predicate.Artifact.Valid() {
			return errors.New("run record: alias predicate names exactly an alias")
		}
	case PredicateArtifactExists:
		if predicate.Alias != "" || !predicate.Artifact.Valid() {
			return errors.New("run record: artifact predicate names exactly an artifact")
		}
	default:
		return errors.New("run record: foreign completion predicate")
	}
	return nil
}

func canonicalizeLoopObligationCompletion(value *LoopObligationCompletion) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Obligation.Kind() != artifact.KindEvidence || !value.Evidence.Valid() ||
		value.CompletedUnixNS <= 0 {
		return errors.New("run record: invalid loop obligation completion")
	}
	return nil
}
