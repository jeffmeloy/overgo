package runrecord

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
)

// CausalTrigger names why an execution entered the system.
type CausalTrigger string

const (
	// TriggerManual is an operator-initiated execution.
	TriggerManual CausalTrigger = "manual"
	// TriggerSchedule is a time-driven execution.
	TriggerSchedule CausalTrigger = "schedule"
	// TriggerWebhook is an external ingress event.
	TriggerWebhook CausalTrigger = "webhook"
	// TriggerControllerProposal is a controller-admitted proposal.
	TriggerControllerProposal CausalTrigger = "controller-proposal"
	// TriggerStageWakeup is a staged workflow resuming a waiting node.
	TriggerStageWakeup CausalTrigger = "stage-wakeup"
	// TriggerRetry re-executes a failed attempt under the same root.
	TriggerRetry CausalTrigger = "retry"
	// TriggerRerun replays a completed execution under the same root.
	TriggerRerun CausalTrigger = "rerun"
	// TriggerDelegation runs on behalf of another execution.
	TriggerDelegation CausalTrigger = "delegation"
	// TriggerRecovery repairs work lost by a crash.
	TriggerRecovery CausalTrigger = "recovery"
	// TriggerFollowup consumes stimuli that arrived after an attempt boundary.
	TriggerFollowup CausalTrigger = "followup"
)

// OriginatingTriggers are the triggers that may start a causal root:
// something outside the execution graph caused them.
var OriginatingTriggers = []CausalTrigger{
	TriggerManual, TriggerSchedule, TriggerWebhook,
	TriggerControllerProposal, TriggerStageWakeup,
}

// DerivedTriggers are the triggers that only continue an existing
// causal chain; they can never mint a root.
var DerivedTriggers = []CausalTrigger{
	TriggerRetry, TriggerRerun, TriggerDelegation, TriggerRecovery, TriggerFollowup,
}

// CausalContext explains why one execution occurred, separately from
// artifact lineage (content derivation) and authority (permission).
// Causality never grants authority: holding a context proves nothing
// about what the execution may do. Root is copied verbatim through
// every derivation, so retry, replay, delegation, and recovery chains
// of any depth answer to the one causal principal that started them
// and a delegation cycle cannot manufacture a new one.
type CausalContext struct {
	Trigger CausalTrigger `json:"trigger"`
	// Root is the evidence artifact that started the causal chain.
	Root artifact.ID `json:"root"`
	// ParentAttempt is the attempt this execution continues, when any.
	ParentAttempt artifact.ID `json:"parent_attempt,omitzero"`
	// RetryOrdinal counts retries under this root; zero means not a retry.
	RetryOrdinal uint32 `json:"retry_ordinal,omitempty"`
	// ReplayOf names the completed execution a rerun repeats.
	ReplayOf artifact.ID `json:"replay_of,omitzero"`
	// DelegatedFrom names the execution this one runs on behalf of.
	DelegatedFrom artifact.ID `json:"delegated_from,omitzero"`
	// RecoveredFrom names the lost work a recovery repairs.
	RecoveredFrom artifact.ID `json:"recovered_from,omitzero"`
	// FollowupOf names the consumed stimulus boundary continued by this run.
	FollowupOf artifact.ID `json:"followup_of,omitzero"`
	// Motivation lists the evidence that motivated the execution,
	// sorted and unique.
	Motivation []artifact.ID `json:"motivation,omitempty"`
}

// NewCausalRoot starts one causal chain from an originating trigger
// and the evidence artifact standing for the external cause.
func NewCausalRoot(trigger CausalTrigger, root artifact.ID, motivation ...artifact.ID) (CausalContext, error) {
	if !slices.Contains(OriginatingTriggers, trigger) {
		return CausalContext{}, errors.New("run record: trigger cannot originate a causal root")
	}
	context := CausalContext{Trigger: trigger, Root: root, Motivation: slices.Clone(motivation)}
	slices.SortFunc(context.Motivation, artifact.CompareID)
	if err := context.Validate(); err != nil {
		return CausalContext{}, err
	}
	return context, nil
}

// Derive continues the chain with a derived trigger about one subject:
// the parent attempt for a retry, the replayed execution for a rerun,
// the delegating execution for a delegation, the lost work for a
// recovery. The root is copied verbatim, never re-derived.
func (context CausalContext) Derive(trigger CausalTrigger, subject artifact.ID) (CausalContext, error) {
	if err := context.Validate(); err != nil {
		return CausalContext{}, err
	}
	derived := CausalContext{Trigger: trigger, Root: context.Root, Motivation: slices.Clone(context.Motivation)}
	switch trigger {
	case TriggerRetry:
		derived.ParentAttempt, derived.RetryOrdinal = subject, context.RetryOrdinal+1
	case TriggerRerun:
		derived.ReplayOf = subject
	case TriggerDelegation:
		derived.DelegatedFrom = subject
	case TriggerRecovery:
		derived.RecoveredFrom = subject
	case TriggerFollowup:
		derived.FollowupOf = subject
	default:
		return CausalContext{}, errors.New("run record: originating trigger cannot derive from a chain")
	}
	if err := derived.Validate(); err != nil {
		return CausalContext{}, err
	}
	return derived, nil
}

// validCausal validates an optional causal binding on a record.
func validCausal(causal *CausalContext) error {
	if causal == nil {
		return nil
	}
	return causal.Validate()
}

// cloneCausal deep-copies an optional causal binding for codec clones.
func cloneCausal(causal *CausalContext) *CausalContext {
	if causal == nil {
		return nil
	}
	cloned := *causal
	cloned.Motivation = slices.Clone(causal.Motivation)
	return &cloned
}

// Link deterministically derives the storage-neutral projection input for one
// validated causal context and execution.
func (context CausalContext) Link(execution artifact.ID) (artifact.CausalLink, error) {
	if err := context.Validate(); err != nil {
		return artifact.CausalLink{}, err
	}
	link := artifact.CausalLink{
		Execution: execution, Root: context.Root, Trigger: string(context.Trigger),
		Subject: context.subject(), Motivation: slices.Clone(context.Motivation),
	}
	if err := link.Validate(); err != nil {
		return artifact.CausalLink{}, err
	}
	return link, nil
}

// BindCausality appends the storage-neutral projection input derived from a
// typed causal context. Nil contexts remain compatible with historical
// records; non-nil contexts are validated before the batch can be published.
func BindCausality(batch *artifact.Batch, execution artifact.ID, causal *CausalContext) error {
	if batch == nil {
		return errors.New("run record: causal batch is absent")
	}
	if causal == nil {
		return nil
	}
	link, err := causal.Link(execution)
	if err != nil {
		return err
	}
	batch.Causality = append(batch.Causality, link)
	return nil
}

func (context CausalContext) subject() artifact.ID {
	switch context.Trigger {
	case TriggerRetry:
		return context.ParentAttempt
	case TriggerRerun:
		return context.ReplayOf
	case TriggerDelegation:
		return context.DelegatedFrom
	case TriggerRecovery:
		return context.RecoveredFrom
	case TriggerFollowup:
		return context.FollowupOf
	default:
		return artifact.ID{}
	}
}

// Validate refuses contexts whose fields disagree with their trigger:
// each derived trigger requires exactly its own subject field, an
// originating trigger carries none of them, and retry ordinals exist
// only on retries.
func (context CausalContext) Validate() error {
	if context.Root.Kind() != artifact.KindEvidence {
		return errors.New("run record: causal root must be evidence")
	}
	subjects := map[CausalTrigger]bool{
		TriggerRetry:      context.ParentAttempt.Valid(),
		TriggerRerun:      context.ReplayOf.Valid(),
		TriggerDelegation: context.DelegatedFrom.Valid(),
		TriggerRecovery:   context.RecoveredFrom.Valid(),
		TriggerFollowup:   context.FollowupOf.Valid(),
	}
	carried, known := subjects[context.Trigger]
	if !known {
		if !slices.Contains(OriginatingTriggers, context.Trigger) {
			return errors.New("run record: foreign causal trigger")
		}
	} else if !carried {
		return errors.New("run record: derived trigger requires its subject")
	}
	for trigger, present := range subjects {
		if trigger != context.Trigger && present {
			return errors.New("run record: causal subject without its trigger")
		}
	}
	for _, subject := range []artifact.ID{
		context.ParentAttempt, context.ReplayOf, context.DelegatedFrom, context.RecoveredFrom, context.FollowupOf,
	} {
		if subject.Valid() && subject.Kind() != artifact.KindEvidence {
			return errors.New("run record: causal subject must be evidence")
		}
	}
	if (context.RetryOrdinal > 0) != (context.Trigger == TriggerRetry) {
		return errors.New("run record: retry ordinal exists exactly on retries")
	}
	if !slices.IsSortedFunc(context.Motivation, artifact.CompareID) || len(context.Motivation) != len(slices.CompactFunc(slices.Clone(context.Motivation), func(left, right artifact.ID) bool { return left == right })) {
		return errors.New("run record: causal motivation must be sorted and unique")
	}
	for _, evidence := range context.Motivation {
		if evidence.Kind() != artifact.KindEvidence {
			return errors.New("run record: causal motivation must be evidence")
		}
	}
	return nil
}
