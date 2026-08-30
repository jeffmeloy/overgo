package runrecord

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
)

const (
	// StimulusFollowupMediaType identifies one coalesced follow-up admission.
	StimulusFollowupMediaType = "application/vnd.overgo.stimulus-followup+json"
	// StimulusFollowupSchema identifies the admission document contract.
	StimulusFollowupSchema = "overgo/stimulus-followup/v1"
	// StimulusFollowupAliasRoot admits at most one follow-up per boundary.
	StimulusFollowupAliasRoot = "attempt/stimulus-followup/"
)

// StimulusFollowup is the single durable run admission caused by stimuli
// observed after one consumed attempt boundary.
type StimulusFollowup struct {
	Version      uint16            `json:"version"`
	Boundary     artifact.ID       `json:"boundary"`
	ObservedHead artifact.CommitID `json:"observed_head"`
	Sources      []artifact.ID     `json:"sources"`
	Causal       CausalContext     `json:"causal"`
	ID           artifact.ID       `json:"-"`
}

var stimulusFollowupCodec = artifact.JSONDocumentCodec(
	"stimulus follow-up", artifact.KindEvidence, StimulusFollowupMediaType, StimulusFollowupSchema,
	canonicalizeStimulusFollowup,
	func(value StimulusFollowup) artifact.ID { return value.ID },
	func(value *StimulusFollowup, id artifact.ID) { value.ID = id },
	func(value StimulusFollowup) StimulusFollowup {
		value.Sources = slices.Clone(value.Sources)
		value.Causal.Motivation = slices.Clone(value.Causal.Motivation)
		return value
	},
)

// StimulusFollowupAuthority owns atomic admission by consumed boundary.
type StimulusFollowupAuthority struct {
	Repository artifact.Repository
}

// Admit sorts and coalesces observed sources, then atomically admits at most one
// causally linked follow-up for the consumed boundary.
func (authority StimulusFollowupAuthority) Admit(ctx context.Context, boundaryID artifact.ID, sources []artifact.ID) (StimulusFollowup, bool, error) {
	if ctx == nil || authority.Repository == nil || len(sources) == 0 {
		return StimulusFollowup{}, false, errors.New("run record: follow-up authority requires late stimuli")
	}
	boundary, err := RequireAttemptStimulus(ctx, authority.Repository, boundaryID)
	if err != nil {
		return StimulusFollowup{}, false, err
	}
	if current, found, currentErr := stimulusFollowupCodec.Resolve(
		ctx, authority.Repository, StimulusFollowupAliasRoot+boundaryID.String(),
	); currentErr != nil {
		return StimulusFollowup{}, false, currentErr
	} else if found {
		return current, false, nil
	}
	coalesced := slices.Clone(sources)
	slices.SortFunc(coalesced, artifact.CompareID)
	coalesced = slices.Compact(coalesced)
	for _, source := range coalesced {
		if source.Kind() != artifact.KindEvidence || slices.ContainsFunc(boundary.Selection.Sources, func(consumed dataset.InteractionSelectionSource) bool {
			return consumed.Source == source
		}) {
			return StimulusFollowup{}, false, errors.New("run record: follow-up source is not late evidence")
		}
		if _, found, artifactErr := authority.Repository.Artifact(ctx, source); artifactErr != nil || !found {
			return StimulusFollowup{}, false, errors.Join(errors.New("run record: late stimulus is absent"), artifactErr)
		}
	}
	root, err := NewCausalRoot(TriggerManual, boundary.Operation, coalesced...)
	if err != nil {
		return StimulusFollowup{}, false, err
	}
	causal, err := root.Derive(TriggerFollowup, boundary.ID)
	if err != nil {
		return StimulusFollowup{}, false, err
	}
	head, _ := authority.Repository.Head()
	admission, err := stimulusFollowupCodec.New(StimulusFollowup{
		Version: artifact.InitialDocumentVersion, Boundary: boundary.ID,
		ObservedHead: head, Sources: coalesced, Causal: causal,
	})
	if err != nil {
		return StimulusFollowup{}, false, err
	}
	content, err := stimulusFollowupCodec.Content(admission)
	if err != nil {
		return StimulusFollowup{}, false, err
	}
	parents := append([]artifact.ID{admission.Boundary, admission.Causal.Root}, admission.Sources...)
	binding := artifact.AliasBinding{Name: StimulusFollowupAliasRoot + boundary.ID.String(), Target: admission.ID}
	key, err := uniquePublicationKey("attempt/stimulus-followup/", admission.ID)
	if err != nil {
		return StimulusFollowup{}, false, err
	}
	batch, err := artifact.NewDocumentBatch(
		key,
		[]artifact.Content{content}, artifact.DependencyLineage(admission.ID, parents...), []artifact.AliasBinding{binding},
	)
	if err == nil {
		err = BindCausality(&batch, admission.ID, &admission.Causal)
	}
	if err == nil {
		_, err = artifact.CommitBatch(ctx, authority.Repository, batch)
	}
	if err == nil {
		return admission, true, nil
	}
	winner, found, resolveErr := stimulusFollowupCodec.Resolve(
		ctx, authority.Repository, StimulusFollowupAliasRoot+boundary.ID.String(),
	)
	if resolveErr == nil && found {
		return winner, false, nil
	}
	return StimulusFollowup{}, false, errors.Join(err, resolveErr)
}

func canonicalizeStimulusFollowup(value *StimulusFollowup) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Boundary.Kind() != artifact.KindEvidence ||
		!value.ObservedHead.Valid() || len(value.Sources) == 0 || !slices.IsSortedFunc(value.Sources, artifact.CompareID) ||
		len(value.Sources) != len(slices.Compact(slices.Clone(value.Sources))) || value.Causal.Trigger != TriggerFollowup ||
		value.Causal.FollowupOf != value.Boundary || !slices.Equal(value.Causal.Motivation, value.Sources) {
		return errors.New("run record: invalid stimulus follow-up")
	}
	if err := value.Causal.Validate(); err != nil {
		return err
	}
	return nil
}
