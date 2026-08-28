package runrecord

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	// AutomationDeliveryAttemptMediaType identifies delivery attempt evidence.
	AutomationDeliveryAttemptMediaType = "application/vnd.overgo.automation-delivery-attempt+json"
	// AutomationDeliveryAttemptSchema identifies the attempt contract.
	AutomationDeliveryAttemptSchema = "overgo/automation-delivery-attempt/v1"
	// AutomationDeliveryAttemptAliasRoot scopes current attempts by idempotency identity.
	AutomationDeliveryAttemptAliasRoot = "automation/delivery-attempt/"
)

// AutomationDeliveryState is the durable external-effect lifecycle.
type AutomationDeliveryState string

const (
	// AutomationDeliveryAdmitted records ownership before the external effect.
	AutomationDeliveryAdmitted AutomationDeliveryState = "admitted"
	// AutomationDeliverySucceeded records a durable result after the effect.
	AutomationDeliverySucceeded AutomationDeliveryState = "succeeded"
	// AutomationDeliveryFailed records a transport failure.
	AutomationDeliveryFailed AutomationDeliveryState = "failed"
)

// AutomationDeliveryAttempt binds an effect to exact plan, operation, output,
// destination, tool, and idempotency authority.
type AutomationDeliveryAttempt struct {
	Version     uint16                  `json:"version"`
	Plan        artifact.ID             `json:"plan"`
	Operation   artifact.ID             `json:"operation"`
	Tool        artifact.ID             `json:"tool"`
	Destination string                  `json:"destination"`
	Outputs     []artifact.ID           `json:"outputs"`
	Idempotency artifact.ID             `json:"idempotency"`
	State       AutomationDeliveryState `json:"state"`
	Result      artifact.ID             `json:"result,omitzero"`
	Failure     string                  `json:"failure,omitempty"`
	Prior       artifact.ID             `json:"prior,omitzero"`
	// Causal explains why the delivery attempt occurred.
	Causal *CausalContext `json:"causal,omitempty"`
	ID     artifact.ID    `json:"-"`
}

var automationDeliveryAttemptCodec = artifact.JSONDocumentCodec(
	"automation delivery attempt", artifact.KindEvidence,
	AutomationDeliveryAttemptMediaType, AutomationDeliveryAttemptSchema,
	canonicalizeAutomationDeliveryAttempt,
	func(value AutomationDeliveryAttempt) artifact.ID { return value.ID },
	func(value *AutomationDeliveryAttempt, id artifact.ID) { value.ID = id },
	func(value AutomationDeliveryAttempt) AutomationDeliveryAttempt {
		value.Outputs = slices.Clone(value.Outputs)
		value.Causal = cloneCausal(value.Causal)
		return value
	},
)

// AutomationDeliveryAuthority owns external-effect idempotency evidence.
type AutomationDeliveryAuthority struct{ Repository artifact.Repository }

// Current resolves the latest evidence for an idempotency identity.
func (authority AutomationDeliveryAuthority) Current(
	ctx context.Context,
	idempotency artifact.ID,
) (AutomationDeliveryAttempt, bool, error) {
	return automationDeliveryAttemptCodec.Resolve(ctx, authority.Repository, AutomationDeliveryAttemptAliasRoot+idempotency.String())
}

// Begin atomically owns one external delivery. Won is false for an existing
// admitted or successful identity so a retry cannot duplicate the effect.
func (authority AutomationDeliveryAuthority) Begin(
	ctx context.Context,
	value AutomationDeliveryAttempt,
) (AutomationDeliveryAttempt, bool, error) {
	if ctx == nil || authority.Repository == nil || value.State != AutomationDeliveryAdmitted {
		return AutomationDeliveryAttempt{}, false, errors.New("run record: invalid automation delivery admission")
	}
	current, found, err := authority.Current(ctx, value.Idempotency)
	if err != nil {
		return AutomationDeliveryAttempt{}, false, err
	}
	if found && current.State != AutomationDeliveryFailed {
		return current, false, nil
	}
	if found {
		value.Prior = current.ID
	}
	value.Version, value.ID, value.Result, value.Failure = artifact.InitialDocumentVersion, artifact.ID{}, artifact.ID{}, ""
	value, err = automationDeliveryAttemptCodec.New(value)
	if err != nil {
		return AutomationDeliveryAttempt{}, false, err
	}
	if err := authority.publish(ctx, value, nil); err != nil {
		winner, winnerFound, resolveErr := authority.Current(ctx, value.Idempotency)
		if resolveErr == nil && winnerFound && winner.State != AutomationDeliveryFailed {
			return winner, false, nil
		}
		return AutomationDeliveryAttempt{}, false, errors.Join(err, resolveErr)
	}
	return value, true, nil
}

// Finish advances an admitted attempt to success or failure.
func (authority AutomationDeliveryAuthority) Finish(
	ctx context.Context,
	admitted AutomationDeliveryAttempt,
	result *artifact.Content,
	failure string,
) (AutomationDeliveryAttempt, error) {
	current, found, err := authority.Current(ctx, admitted.Idempotency)
	if err != nil || !found || current.ID != admitted.ID || current.State != AutomationDeliveryAdmitted {
		return AutomationDeliveryAttempt{}, errors.Join(errors.New("run record: automation delivery admission differs"), err)
	}
	finished := admitted
	finished.ID, finished.Prior = artifact.ID{}, admitted.ID
	if result == nil {
		finished.State, finished.Failure, finished.Result = AutomationDeliveryFailed, failure, artifact.ID{}
	} else {
		finished.State, finished.Failure, finished.Result = AutomationDeliverySucceeded, "", result.Descriptor.ID
	}
	finished, err = automationDeliveryAttemptCodec.New(finished)
	if err != nil {
		return AutomationDeliveryAttempt{}, err
	}
	var content []artifact.Content
	if result != nil {
		content = []artifact.Content{*result}
	}
	if err := authority.publish(ctx, finished, content); err != nil {
		return AutomationDeliveryAttempt{}, err
	}
	return finished, nil
}

func (authority AutomationDeliveryAuthority) publish(
	ctx context.Context,
	value AutomationDeliveryAttempt,
	extra []artifact.Content,
) error {
	content, err := automationDeliveryAttemptCodec.Content(value)
	if err != nil {
		return err
	}
	contents := append(slices.Clone(extra), content)
	binding := artifact.AliasBinding{
		Name: AutomationDeliveryAttemptAliasRoot + value.Idempotency.String(), Target: value.ID,
	}
	if value.Prior.Valid() {
		binding.Previous = artifact.IDPointer(value.Prior)
	}
	parents := []artifact.ID{value.Plan, value.Tool}
	parents = append(parents, value.Outputs...)
	if value.Prior.Valid() {
		parents = append(parents, value.Prior)
	}
	if value.Result.Valid() {
		parents = append(parents, value.Result)
	}
	var nonce [sha256.Size]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return err
	}
	batch, err := artifact.NewDocumentBatch(
		fmt.Sprintf("automation/delivery/%s/%x", value.ID, nonce[:]), contents,
		artifact.DependencyLineage(value.ID, parents...), []artifact.AliasBinding{binding},
	)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, authority.Repository, batch)
	return err
}

func canonicalizeAutomationDeliveryAttempt(value *AutomationDeliveryAttempt) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Plan.Kind() != artifact.KindProfile || value.Operation.Kind() != artifact.KindEvidence ||
		value.Tool.Kind() != artifact.KindRecipe || value.Idempotency.Kind() != artifact.KindEvidence ||
		strings.TrimSpace(value.Destination) != value.Destination || value.Destination == "" || len(value.Outputs) == 0 ||
		value.Prior.Valid() && value.Prior.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid automation delivery attempt")
	}
	for _, output := range value.Outputs {
		if !output.Valid() {
			return errors.New("run record: invalid automation delivery output")
		}
	}
	if err := validCausal(value.Causal); err != nil {
		return err
	}
	switch value.State {
	case AutomationDeliveryAdmitted:
		if value.Result.Valid() || value.Failure != "" {
			return errors.New("run record: admitted delivery has terminal facts")
		}
	case AutomationDeliverySucceeded:
		if value.Result.Kind() != artifact.KindOutput || value.Failure != "" || !value.Prior.Valid() {
			return errors.New("run record: successful delivery lacks result")
		}
	case AutomationDeliveryFailed:
		if value.Result.Valid() || value.Failure == "" || !value.Prior.Valid() {
			return errors.New("run record: failed delivery lacks failure")
		}
	default:
		return errors.New("run record: invalid automation delivery state")
	}
	value.Outputs = slices.Clone(value.Outputs)
	return nil
}
