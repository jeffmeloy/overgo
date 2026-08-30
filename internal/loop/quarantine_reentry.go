package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/textcheck"
)

const (
	// QuarantineReentryMediaType identifies reentry admission records.
	QuarantineReentryMediaType = "application/vnd.overgo.quarantine-reentry+json"
	// QuarantineReentrySchema identifies the reentry contract.
	QuarantineReentrySchema = "overgo/quarantine-reentry/v1"
	// reentryTextBytes bounds the recorded decider and reason.
	reentryTextBytes = 256
)

// QuarantineReentry is the complete admission a quarantined candidate or
// mechanism family must present before re-entering rollout: the prior
// finding it answers, the changed-mechanism evidence, the new evaluation,
// the explicit decider, and the fresh rollout plan that will govern it.
type QuarantineReentry struct {
	Candidate  artifact.ID `json:"candidate"`
	Finding    artifact.ID `json:"finding"`
	Mechanism  artifact.ID `json:"mechanism"`
	Evaluation artifact.ID `json:"evaluation"`
	Plan       artifact.ID `json:"plan"`
	Decider    string      `json:"decider"`
	Reason     string      `json:"reason"`
}

// reentryEvidenceReader is the store surface reentry admission needs: the
// ordinary repository plus introduction ordering, which proves evidence is
// genuinely newer than the quarantine it answers.
type reentryEvidenceReader interface {
	artifact.Repository
	ArtifactIntroduction(context.Context, artifact.ID) (overgodb.ArtifactIntroduction, bool, error)
}

func requireIntroducedAfter(
	ctx context.Context, store reentryEvidenceReader, id artifact.ID, boundary uint64, role string,
) error {
	introduction, found, err := store.ArtifactIntroduction(ctx, id)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("loop: reentry %s evidence %s is not committed", role, id)
	}
	if introduction.Sequence <= boundary {
		return fmt.Errorf(
			"loop: reentry %s evidence predates the quarantine finding; restart, replay, or a repeated proposal cannot clear quarantine",
			role,
		)
	}
	return nil
}

// AdmitQuarantineReentry admits one quarantined candidate back into
// rollout only through the complete new-evidence closure: the finding must
// be the published rollback record that quarantined the family, the
// changed mechanism, new evaluation, and fresh rollout plan must each have
// entered the store after that finding — pre-quarantine evidence replayed
// through a restart proves nothing — the plan must govern exactly the
// re-entering candidate, and an explicit decider signs the admission.
// The published record cites every part of the closure.
func AdmitQuarantineReentry(
	ctx context.Context,
	store reentryEvidenceReader,
	reentry QuarantineReentry,
) (artifact.ID, error) {
	if !textcheck.Bounded(reentry.Decider, reentryTextBytes, "\x00") || strings.TrimSpace(reentry.Decider) == "" {
		return artifact.ID{}, errors.New("loop: reentry requires an explicit recorded decider")
	}
	if !textcheck.Bounded(reentry.Reason, reentryTextBytes, "\x00") || strings.TrimSpace(reentry.Reason) == "" {
		return artifact.ID{}, errors.New("loop: reentry requires its recorded reason")
	}
	if reentry.Candidate.Kind() != artifact.KindRecipe {
		return artifact.ID{}, errors.New("loop: reentry requires an exact candidate recipe identity")
	}
	content, found, err := artifact.ReadContent(ctx, store, reentry.Finding)
	if err != nil {
		return artifact.ID{}, err
	}
	if !found || content.Descriptor.MediaType != LiveRollbackMediaType {
		return artifact.ID{}, errors.New("loop: reentry must name the published rollback finding it answers")
	}
	var finding LiveRollbackRecord
	if err := json.Unmarshal(content.Data, &finding); err != nil {
		return artifact.ID{}, err
	}
	findingIntroduction, found, err := store.ArtifactIntroduction(ctx, reentry.Finding)
	if err != nil || !found {
		return artifact.ID{}, errors.Join(errors.New("loop: reentry finding has no introduction record"), err)
	}
	boundary := findingIntroduction.Sequence
	if err := requireIntroducedAfter(ctx, store, reentry.Mechanism, boundary, "changed-mechanism"); err != nil {
		return artifact.ID{}, err
	}
	if err := requireIntroducedAfter(ctx, store, reentry.Evaluation, boundary, "evaluation"); err != nil {
		return artifact.ID{}, err
	}
	if err := requireIntroducedAfter(ctx, store, reentry.Plan, boundary, "rollout-plan"); err != nil {
		return artifact.ID{}, err
	}
	planContent, found, err := artifact.ReadContent(ctx, store, reentry.Plan)
	if err != nil {
		return artifact.ID{}, err
	}
	if !found {
		return artifact.ID{}, errors.New("loop: reentry rollout plan is not committed")
	}
	plan, err := runrecord.ParseRolloutPlan(planContent.Data)
	if err != nil {
		return artifact.ID{}, err
	}
	if plan.Candidate != reentry.Candidate {
		return artifact.ID{}, errors.New("loop: the fresh rollout plan does not govern the re-entering candidate")
	}
	if plan.Baseline != finding.Restored {
		return artifact.ID{}, errors.New("loop: the fresh rollout plan must baseline the predecessor the quarantine restored")
	}
	contract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: QuarantineReentryMediaType, Schema: QuarantineReentrySchema,
	}
	recordContent, err := artifact.JSONContent(contract, reentry)
	if err != nil {
		return artifact.ID{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"live-safety/reentry/"+recordContent.Descriptor.ID.String(),
		[]artifact.Content{recordContent},
		artifact.UniqueDependencyLineage(
			recordContent.Descriptor.ID,
			reentry.Candidate, reentry.Finding, reentry.Mechanism, reentry.Evaluation, reentry.Plan,
		),
		nil,
	)
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return artifact.ID{}, err
	}
	return recordContent.Descriptor.ID, nil
}
