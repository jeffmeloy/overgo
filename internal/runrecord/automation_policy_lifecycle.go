package runrecord

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	AutomationPolicyLifecycleMediaType = "application/vnd.overgo.automation-policy-lifecycle+json"
	AutomationPolicyLifecycleSchema    = "overgo/automation-policy-lifecycle/v1"
	AutomationPolicyLifecycleAliasRoot = "automation/policy/lifecycle/"
	AutomationPolicyActiveAliasRoot    = "automation/policy/active/"
)

type AutomationPolicyState string

const (
	AutomationPolicyDeclared     AutomationPolicyState = "declared"
	AutomationPolicyExperimental AutomationPolicyState = "experimental"
	AutomationPolicyVerified     AutomationPolicyState = "verified"
	AutomationPolicyActive       AutomationPolicyState = "active"
	AutomationPolicyContained    AutomationPolicyState = "contained"
	AutomationPolicyRolledBack   AutomationPolicyState = "rolled-back"
)

type AutomationPolicyLifecycle struct {
	Version            uint16                `json:"version"`
	ID                 artifact.ID           `json:"-"`
	Name               string                `json:"name"`
	Policy             artifact.ID           `json:"policy"`
	State              AutomationPolicyState `json:"state"`
	Incumbent          artifact.ID           `json:"incumbent"`
	Rollback           artifact.ID           `json:"rollback"`
	EvaluationPlan     artifact.ID           `json:"evaluation_plan"`
	EvaluationEvidence artifact.ID           `json:"evaluation_evidence"`
	Trajectories       []artifact.ID         `json:"trajectories"`
	Decision           artifact.ID           `json:"decision"`
	Prior              artifact.ID           `json:"prior,omitzero"`
}

var automationPolicyLifecycleCodec = artifact.JSONDocumentCodec(
	"automation policy lifecycle", artifact.KindEvidence, AutomationPolicyLifecycleMediaType, AutomationPolicyLifecycleSchema,
	canonicalizeAutomationPolicyLifecycle, func(v AutomationPolicyLifecycle) artifact.ID { return v.ID }, func(v *AutomationPolicyLifecycle, id artifact.ID) { v.ID = id },
	func(v AutomationPolicyLifecycle) AutomationPolicyLifecycle {
		v.Trajectories = slices.Clone(v.Trajectories)
		return v
	},
)

// PublishAutomationPolicyTransition advances the lifecycle and active alias in
// one CAS batch. The active alias points to immutable recipe-profile authority.
func PublishAutomationPolicyTransition(ctx context.Context, repository artifact.Repository, value AutomationPolicyLifecycle) (AutomationPolicyLifecycle, error) {
	if ctx == nil || repository == nil {
		return AutomationPolicyLifecycle{}, errors.New("run record: policy lifecycle authority is absent")
	}
	priorID, found, err := repository.ResolveAlias(ctx, AutomationPolicyLifecycleAliasRoot+value.Name)
	if err != nil {
		return AutomationPolicyLifecycle{}, err
	}
	var prior AutomationPolicyLifecycle
	if found {
		prior, err = automationPolicyLifecycleCodec.Require(ctx, repository, priorID)
		if err != nil {
			return AutomationPolicyLifecycle{}, err
		}
		value.Prior = prior.ID
	}
	if !validAutomationPolicyTransition(prior, found, value) {
		return AutomationPolicyLifecycle{}, errors.New("run record: invalid automation policy transition")
	}
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	value, err = automationPolicyLifecycleCodec.New(value)
	if err != nil {
		return AutomationPolicyLifecycle{}, err
	}
	content, err := automationPolicyLifecycleCodec.Content(value)
	if err != nil {
		return AutomationPolicyLifecycle{}, err
	}
	lifecycleAlias := artifact.AliasBinding{Name: AutomationPolicyLifecycleAliasRoot + value.Name, Target: value.ID}
	if found {
		lifecycleAlias.Previous = artifact.IDPointer(prior.ID)
	}
	aliases := []artifact.AliasBinding{lifecycleAlias}
	if value.State == AutomationPolicyActive || value.State == AutomationPolicyRolledBack {
		target := value.Policy
		if value.State == AutomationPolicyRolledBack {
			target = value.Rollback
		}
		active := artifact.AliasBinding{Name: AutomationPolicyActiveAliasRoot + value.Name, Target: target}
		if current, activeFound, resolveErr := repository.ResolveAlias(ctx, active.Name); resolveErr != nil {
			return AutomationPolicyLifecycle{}, resolveErr
		} else if activeFound {
			active.Previous = artifact.IDPointer(current)
		}
		aliases = append(aliases, active)
	}
	parents := []artifact.ID{value.Policy, value.Incumbent, value.Rollback, value.EvaluationPlan, value.EvaluationEvidence, value.Decision}
	parents = append(parents, value.Trajectories...)
	if value.Prior.Valid() {
		parents = append(parents, value.Prior)
	}
	batch, err := artifact.NewDocumentBatch("automation-policy/"+value.ID.String(), []artifact.Content{content}, artifact.DependencyLineage(value.ID, parents...), aliases)
	if err != nil {
		return AutomationPolicyLifecycle{}, err
	}
	seen := map[artifact.ID]bool{}
	for _, parent := range parents {
		_, alreadyStored, inspectErr := repository.Artifact(ctx, parent)
		if inspectErr != nil {
			return AutomationPolicyLifecycle{}, inspectErr
		}
		if !seen[parent] && !alreadyStored {
			batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: parent})
			seen[parent] = true
		}
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return value, err
}

func canonicalizeAutomationPolicyLifecycle(v *AutomationPolicyLifecycle) error {
	if v == nil || v.Version != artifact.InitialDocumentVersion || !textcheck.LowerIdentifier(v.Name, len(v.Name)) ||
		v.Policy.Kind() != artifact.KindProfile || v.Incumbent.Kind() != artifact.KindProfile || v.Rollback.Kind() != artifact.KindProfile ||
		v.EvaluationPlan.Kind() != artifact.KindProfile || v.EvaluationEvidence.Kind() != artifact.KindEvidence || v.Decision.Kind() != artifact.KindEvidence || len(v.Trajectories) == 0 {
		return errors.New("run record: invalid automation policy lifecycle")
	}
	for _, id := range v.Trajectories {
		if id.Kind() != artifact.KindEvidence {
			return errors.New("run record: invalid policy trajectory")
		}
	}
	slices.SortFunc(v.Trajectories, artifact.CompareID)
	v.Trajectories = slices.Compact(v.Trajectories)
	if v.Prior.Valid() && v.Prior.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid policy prior")
	}
	return nil
}

func validAutomationPolicyTransition(prior AutomationPolicyLifecycle, found bool, next AutomationPolicyLifecycle) bool {
	if !found {
		return next.State == AutomationPolicyDeclared && !next.Prior.Valid()
	}
	if next.Policy != prior.Policy || next.Incumbent != prior.Incumbent || next.Rollback != prior.Rollback || next.Prior != prior.ID {
		return false
	}
	switch prior.State {
	case AutomationPolicyDeclared:
		return next.State == AutomationPolicyExperimental
	case AutomationPolicyExperimental:
		return next.State == AutomationPolicyVerified
	case AutomationPolicyVerified:
		return next.State == AutomationPolicyActive
	case AutomationPolicyActive:
		return next.State == AutomationPolicyContained
	case AutomationPolicyContained:
		return next.State == AutomationPolicyRolledBack
	default:
		return false
	}
}
