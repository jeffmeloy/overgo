package runrecord

import (
	"context"
	"errors"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	// AutomationScheduleClaimMediaType identifies durable schedule claims.
	AutomationScheduleClaimMediaType = "application/vnd.overgo.automation-schedule-claim+json"
	// AutomationScheduleClaimSchema identifies the schedule claim contract.
	AutomationScheduleClaimSchema = "overgo/automation-schedule-claim/v1"
	// AutomationScheduleClaimAliasRoot scopes the latest claim by automation.
	AutomationScheduleClaimAliasRoot = "automation/schedule-claim/"
)

// AutomationScheduleClaim owns one exact due slot before execution admission.
type AutomationScheduleClaim struct {
	Version     uint16      `json:"version"`
	Name        string      `json:"name"`
	Plan        artifact.ID `json:"plan"`
	DueUnixNano int64       `json:"due_unix_nano"`
	Prior       artifact.ID `json:"prior,omitzero"`
	ID          artifact.ID `json:"-"`
}

// Due returns the claim slot as UTC time without losing nanosecond precision.
func (claim AutomationScheduleClaim) Due() time.Time {
	return time.UnixMicro(claim.DueUnixNano / int64(time.Microsecond)).
		Add(time.Duration(claim.DueUnixNano % int64(time.Microsecond))).UTC()
}

var automationScheduleClaimCodec = artifact.JSONDocumentCodec(
	"automation schedule claim", artifact.KindEvidence,
	AutomationScheduleClaimMediaType, AutomationScheduleClaimSchema,
	func(value *AutomationScheduleClaim) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion ||
			!textcheck.LowerIdentifier(value.Name, len(value.Name)) ||
			value.Plan.Kind() != artifact.KindProfile || value.DueUnixNano < 0 ||
			value.Prior.Valid() && value.Prior.Kind() != artifact.KindEvidence {
			return errors.New("run record: invalid automation schedule claim")
		}
		return nil
	},
	func(value AutomationScheduleClaim) artifact.ID { return value.ID },
	func(value *AutomationScheduleClaim, id artifact.ID) { value.ID = id },
	func(value AutomationScheduleClaim) AutomationScheduleClaim { return value },
)

// AutomationScheduleAuthority owns durable per-automation schedule claims.
type AutomationScheduleAuthority struct {
	Repository artifact.Repository
}

// Current resolves the latest durable schedule claim.
func (authority AutomationScheduleAuthority) Current(
	ctx context.Context,
	name string,
) (AutomationScheduleClaim, bool, error) {
	return automationScheduleClaimCodec.Resolve(ctx, authority.Repository, AutomationScheduleClaimAliasRoot+name)
}

// Require loads one exact claim for restart recovery.
func (authority AutomationScheduleAuthority) Require(
	ctx context.Context,
	id artifact.ID,
) (AutomationScheduleClaim, error) {
	return automationScheduleClaimCodec.Require(ctx, authority.Repository, id)
}

// Claim atomically owns a due slot. Exactly one concurrent caller returns won.
func (authority AutomationScheduleAuthority) Claim(
	ctx context.Context,
	name string,
	plan artifact.ID,
	due time.Time,
) (claim AutomationScheduleClaim, won bool, err error) {
	if ctx == nil || authority.Repository == nil {
		return claim, false, errors.New("run record: automation schedule authority is absent")
	}
	current, found, err := authority.Current(ctx, name)
	if err != nil {
		return claim, false, err
	}
	due = due.UTC()
	if found && current.DueUnixNano >= due.UnixNano() {
		return current, false, nil
	}
	prior := artifact.ID{}
	if found {
		prior = current.ID
	}
	claim, err = automationScheduleClaimCodec.New(AutomationScheduleClaim{
		Version: artifact.InitialDocumentVersion, Name: name, Plan: plan,
		DueUnixNano: due.UnixNano(), Prior: prior,
	})
	if err != nil {
		return AutomationScheduleClaim{}, false, err
	}
	content, err := automationScheduleClaimCodec.Content(claim)
	if err != nil {
		return AutomationScheduleClaim{}, false, err
	}
	lineageParents := []artifact.ID{plan}
	if prior.Valid() {
		lineageParents = append(lineageParents, prior)
	}
	binding := artifact.AliasBinding{Name: AutomationScheduleClaimAliasRoot + name, Target: claim.ID}
	if found {
		binding.Previous = artifact.IDPointer(prior)
	}
	key, err := uniquePublicationKey("automation/schedule/claim/", claim.ID)
	if err != nil {
		return AutomationScheduleClaim{}, false, err
	}
	batch, err := artifact.NewDocumentBatch(
		key, []artifact.Content{content},
		artifact.DependencyLineage(claim.ID, lineageParents...), []artifact.AliasBinding{binding},
	)
	if err == nil {
		_, err = artifact.CommitBatch(ctx, authority.Repository, batch)
	}
	if err == nil {
		return claim, true, nil
	}
	winner, winnerFound, resolveErr := authority.Current(ctx, name)
	if resolveErr == nil && winnerFound && winner.DueUnixNano >= due.UnixNano() {
		return winner, false, nil
	}
	return AutomationScheduleClaim{}, false, errors.Join(err, resolveErr)
}
