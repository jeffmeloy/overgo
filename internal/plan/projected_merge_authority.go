package plan

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/worklease"
)

const (
	// FirstParentTargetMergeAuthorityMediaType identifies the durable audit
	// boundary that lets a target repository replay an explicitly projected
	// merge without retaining the source lane's authority store.
	FirstParentTargetMergeAuthorityMediaType = "application/vnd.overgo.first-parent-target-merge-authority+json"
	// FirstParentTargetMergeAuthoritySchema is the exact receipt schema.
	FirstParentTargetMergeAuthoritySchema = "overgo/first-parent-target-merge-authority/v1"
	// FirstParentTargetSourceAuditOnlyPolicy makes source-plan non-adoption an
	// explicit, content-addressed receipt invariant.
	FirstParentTargetSourceAuditOnlyPolicy = "source-audit-only"
)

// CompletionStoreCoordinate binds one replay-validated OvergoDB head.
type CompletionStoreCoordinate struct {
	Commit   artifact.CommitID `json:"commit"`
	Sequence uint64            `json:"sequence"`
}

// ProjectedCompletionEvidence is one immutable completion identity retained
// only as audit evidence in a first-parent target receipt. Source completion
// authority is never adopted by the target lane.
type ProjectedCompletionEvidence struct {
	Reference      string      `json:"reference"`
	Commit         string      `json:"commit"`
	Verify         string      `json:"verify"`
	ContractDigest string      `json:"contract_digest"`
	Manifest       artifact.ID `json:"manifest"`
	CodeManifest   artifact.ID `json:"code_manifest"`
	Attempt        artifact.ID `json:"attempt"`
	Result         artifact.ID `json:"result"`
	Finalization   artifact.ID `json:"finalization"`
	RetiredItem    bool        `json:"retired_item,omitzero"`
}

// ProjectedRetirementEvidence binds one item tombstone in an audited authority
// snapshot. It never grants target authority.
type ProjectedRetirementEvidence struct {
	Item     string                      `json:"item"`
	Evidence ProjectedCompletionEvidence `json:"evidence"`
}

// FirstParentTargetMergeAuthority is the content-addressed receipt for one
// explicit authority projection. The receipt binds the complete incoming
// authority digest and source-store coordinate as audit evidence while
// adopting none of it. Subsequent replay depends only on the first-parent Git
// graph and target OvergoDB. Machine-local worktree paths are excluded.
type FirstParentTargetMergeAuthority struct {
	Version                    uint16                        `json:"version"`
	Projection                 MergeProjection               `json:"projection"`
	LocalParent                string                        `json:"local_parent"`
	IncomingParent             string                        `json:"incoming_parent"`
	MergeBase                  string                        `json:"merge_base"`
	LocalPlanDigest            string                        `json:"local_plan_digest"`
	IncomingPlanDigest         string                        `json:"incoming_plan_digest"`
	MergeBasePlanDigest        string                        `json:"merge_base_plan_digest"`
	PreAdvancePlanDigest       string                        `json:"pre_advance_plan_digest"`
	ChildPlanDigest            string                        `json:"child_plan_digest"`
	PlanItem                   string                        `json:"plan_item"`
	PlanStep                   string                        `json:"plan_step"`
	Preparation                artifact.ID                   `json:"preparation"`
	PreparationCommit          artifact.CommitID             `json:"preparation_commit"`
	TargetStore                CompletionStoreCoordinate     `json:"target_store"`
	SourceStore                CompletionStoreCoordinate     `json:"source_store"`
	SharedStorePrefix          overgodb.CommitPrefix         `json:"shared_store_prefix"`
	LocalProtectionSeeds       []string                      `json:"local_protection_seeds"`
	IncomingProtectionSeeds    []string                      `json:"incoming_protection_seeds"`
	LocalAuthorityDigest       string                        `json:"local_authority_digest"`
	IncomingAuthorityDigest    string                        `json:"incoming_authority_digest"`
	SourceAuthorityPolicy      string                        `json:"source_authority_policy"`
	AuditedIncomingCompletions []ProjectedCompletionEvidence `json:"audited_incoming_completions"`
	AuditedIncomingRetirements []ProjectedRetirementEvidence `json:"audited_incoming_retirements"`
	ID                         artifact.ID                   `json:"-"`
}

var firstParentTargetMergeAuthorityCodec = artifact.JSONDocumentCodec(
	"first-parent target merge authority",
	artifact.KindEvidence,
	FirstParentTargetMergeAuthorityMediaType,
	FirstParentTargetMergeAuthoritySchema,
	canonicalizeFirstParentTargetMergeAuthority,
	func(value FirstParentTargetMergeAuthority) artifact.ID { return value.ID },
	func(value *FirstParentTargetMergeAuthority, id artifact.ID) { value.ID = id },
	cloneFirstParentTargetMergeAuthority,
)

// NewFirstParentTargetMergeAuthority derives one receipt from exact opaque
// parent authorities and stable read-only store coordinates. Callers must
// resolve each authority against the corresponding store immediately before
// invoking this constructor; a moved or substituted store is rejected.
func NewFirstParentTargetMergeAuthority(
	ctx context.Context,
	repository, localRevision, incomingRevision, mergeBaseRevision string,
	local, incoming, mergeBase, preAdvance, child Plan,
	item, step string,
	preparation artifact.ID,
	preparationCommit artifact.CommitID,
	localAuthority, incomingAuthority CompletionAuthority,
	targetStore, sourceStore *overgodb.Store,
) (FirstParentTargetMergeAuthority, error) {
	if ctx == nil || targetStore == nil || sourceStore == nil {
		return FirstParentTargetMergeAuthority{}, errors.New("plan: projected merge authority requires context and two stores")
	}
	if err := ctx.Err(); err != nil {
		return FirstParentTargetMergeAuthority{}, err
	}
	if !localAuthority.resolvesAt(local, repository, localRevision) ||
		!incomingAuthority.resolvesAt(incoming, repository, incomingRevision) {
		return FirstParentTargetMergeAuthority{}, errors.New("plan: projected merge authority lacks an exact parent authority")
	}
	prefix, err := VerifyFirstParentTargetMergeSources(
		ctx, repository, localRevision, incomingRevision, local, incoming, preAdvance,
		localAuthority, incomingAuthority, targetStore, sourceStore,
	)
	if err != nil {
		return FirstParentTargetMergeAuthority{}, err
	}
	targetHead, targetSequence := targetStore.Head()
	sourceHead, sourceSequence := sourceStore.Head()
	auditedCompletions, auditedRetirements, err := completionAuthoritySnapshot(incomingAuthority)
	if err != nil {
		return FirstParentTargetMergeAuthority{}, err
	}
	localDigest, err := completionAuthorityDigest(localAuthority)
	if err != nil {
		return FirstParentTargetMergeAuthority{}, err
	}
	incomingDigest, err := completionAuthorityDigest(incomingAuthority)
	if err != nil {
		return FirstParentTargetMergeAuthority{}, err
	}
	value := FirstParentTargetMergeAuthority{
		Version:                    artifact.InitialDocumentVersion,
		Projection:                 MergeProjectionFirstParentTarget,
		LocalParent:                strings.TrimSpace(localRevision),
		IncomingParent:             strings.TrimSpace(incomingRevision),
		MergeBase:                  strings.TrimSpace(mergeBaseRevision),
		LocalPlanDigest:            local.Digest(),
		IncomingPlanDigest:         incoming.Digest(),
		MergeBasePlanDigest:        mergeBase.Digest(),
		PreAdvancePlanDigest:       preAdvance.Digest(),
		ChildPlanDigest:            child.Digest(),
		PlanItem:                   item,
		PlanStep:                   step,
		Preparation:                preparation,
		PreparationCommit:          preparationCommit,
		TargetStore:                CompletionStoreCoordinate{Commit: targetHead, Sequence: targetSequence},
		SourceStore:                CompletionStoreCoordinate{Commit: sourceHead, Sequence: sourceSequence},
		SharedStorePrefix:          prefix,
		LocalProtectionSeeds:       sortedProtectionSeeds(localAuthority),
		IncomingProtectionSeeds:    sortedProtectionSeeds(incomingAuthority),
		LocalAuthorityDigest:       localDigest,
		IncomingAuthorityDigest:    incomingDigest,
		SourceAuthorityPolicy:      FirstParentTargetSourceAuditOnlyPolicy,
		AuditedIncomingCompletions: auditedCompletions,
		AuditedIncomingRetirements: auditedRetirements,
	}
	value, err = firstParentTargetMergeAuthorityCodec.New(value)
	if err != nil {
		return FirstParentTargetMergeAuthority{}, err
	}
	if err := VerifyProspectiveFirstParentTargetMergeAuthority(
		repository, localRevision, incomingRevision, mergeBaseRevision,
		local, incoming, mergeBase, preAdvance, child, localAuthority, incomingAuthority, value,
	); err != nil {
		return FirstParentTargetMergeAuthority{}, err
	}
	if observedHead, observedSequence := targetStore.Head(); observedHead != targetHead || observedSequence != targetSequence {
		return FirstParentTargetMergeAuthority{}, errors.New("plan: target authority store moved while deriving projected merge receipt")
	}
	if observedHead, observedSequence := sourceStore.Head(); observedHead != sourceHead || observedSequence != sourceSequence {
		return FirstParentTargetMergeAuthority{}, errors.New("plan: source authority store moved while deriving projected merge receipt")
	}
	return value, nil
}

// VerifyFirstParentTargetMergeSources is the preparation-free preflight used
// by plan staging. It verifies the exact parent plans and authorities, proves
// the stores are the ones against which those opaque authorities resolved,
// and requires a nonzero shared hash-chain prefix before an epoch join can be
// proposed. It creates no receipt and grants no historical authority.
func VerifyFirstParentTargetMergeSources(
	ctx context.Context,
	repository, localRevision, incomingRevision string,
	local, incoming, merged Plan,
	localAuthority, incomingAuthority CompletionAuthority,
	targetStore, sourceStore *overgodb.Store,
) (overgodb.CommitPrefix, error) {
	if ctx == nil || targetStore == nil || sourceStore == nil {
		return overgodb.CommitPrefix{}, errors.New("plan: projected merge source preflight requires context and two stores")
	}
	if err := ctx.Err(); err != nil {
		return overgodb.CommitPrefix{}, err
	}
	if err := verifyProspectiveMergeAuthority(
		repository, localRevision, incomingRevision, local, incoming, merged,
		localAuthority, incomingAuthority, MergeProjectionFirstParentTarget, false,
	); err != nil {
		return overgodb.CommitPrefix{}, err
	}
	// Both opaque parents must be internally coherent even though the source
	// authority will only be snapshotted for audit and never adopted.
	if err := validateProjectedRetirementAuthority("local", localAuthority); err != nil {
		return overgodb.CommitPrefix{}, err
	}
	if err := validateProjectedRetirementAuthority("incoming", incomingAuthority); err != nil {
		return overgodb.CommitPrefix{}, err
	}
	targetHead, targetSequence := targetStore.Head()
	sourceHead, sourceSequence := sourceStore.Head()
	if localAuthority.storeHead != targetHead || localAuthority.storeSequence != targetSequence ||
		incomingAuthority.storeHead != sourceHead || incomingAuthority.storeSequence != sourceSequence {
		return overgodb.CommitPrefix{}, errors.New("plan: projected merge source store differs from its resolved parent")
	}
	prefix, err := overgodb.CommonCommitPrefix(ctx, targetStore, sourceStore)
	if err != nil {
		return overgodb.CommitPrefix{}, fmt.Errorf("plan: derive projected merge store prefix: %w", err)
	}
	if !prefix.Commit.Valid() || prefix.Sequence == 0 {
		return overgodb.CommitPrefix{}, errors.New("plan: projected merge authority stores have no shared hash-chain prefix")
	}
	if observedHead, observedSequence := targetStore.Head(); observedHead != targetHead || observedSequence != targetSequence {
		return overgodb.CommitPrefix{}, errors.New("plan: target authority store moved during projected merge preflight")
	}
	if observedHead, observedSequence := sourceStore.Head(); observedHead != sourceHead || observedSequence != sourceSequence {
		return overgodb.CommitPrefix{}, errors.New("plan: source authority store moved during projected merge preflight")
	}
	return prefix, nil
}

// VerifyProspectiveFirstParentTargetMergeAuthority proves that a receipt is
// the canonical projection of the supplied exact parents and opaque parent
// authorities. It permits distinct Git protection roots only because the
// receipt binds their independently verified seeds and shared store ancestry.
func VerifyProspectiveFirstParentTargetMergeAuthority(
	repository, localRevision, incomingRevision, mergeBaseRevision string,
	local, incoming, mergeBase, preAdvance, child Plan,
	localAuthority, incomingAuthority CompletionAuthority,
	receipt FirstParentTargetMergeAuthority,
) error {
	if err := VerifyFirstParentTargetMergeAuthorityTransition(
		localRevision, incomingRevision, mergeBaseRevision,
		local, incoming, mergeBase, preAdvance, child,
		receipt.PlanItem, receipt.PlanStep, receipt.Preparation, receipt.PreparationCommit, receipt,
	); err != nil {
		return err
	}
	if !localAuthority.resolvesAt(local, repository, localRevision) ||
		!incomingAuthority.resolvesAt(incoming, repository, incomingRevision) {
		return errors.New("plan: projected merge receipt lacks exact parent authority")
	}
	if err := verifyProspectiveMergeAuthority(
		repository, localRevision, incomingRevision, local, incoming, preAdvance,
		localAuthority, incomingAuthority, MergeProjectionFirstParentTarget, false,
	); err != nil {
		return err
	}
	if err := VerifyFirstParentTargetLocalAuthority(
		repository, localRevision, local, preAdvance, child, localAuthority,
	); err != nil {
		return err
	}
	if receipt.TargetStore != (CompletionStoreCoordinate{Commit: localAuthority.storeHead, Sequence: localAuthority.storeSequence}) ||
		receipt.SourceStore != (CompletionStoreCoordinate{Commit: incomingAuthority.storeHead, Sequence: incomingAuthority.storeSequence}) {
		return errors.New("plan: projected merge receipt store coordinate differs from parent authority")
	}
	if !slices.Equal(receipt.LocalProtectionSeeds, sortedProtectionSeeds(localAuthority)) ||
		!slices.Equal(receipt.IncomingProtectionSeeds, sortedProtectionSeeds(incomingAuthority)) {
		return errors.New("plan: projected merge receipt protection seeds differ from parent authority")
	}
	localDigest, err := completionAuthorityDigest(localAuthority)
	if err != nil {
		return err
	}
	incomingDigest, err := completionAuthorityDigest(incomingAuthority)
	if err != nil {
		return err
	}
	if receipt.LocalAuthorityDigest != localDigest || receipt.IncomingAuthorityDigest != incomingDigest {
		return errors.New("plan: projected merge receipt parent authority digest differs")
	}
	auditedCompletions, auditedRetirements, err := completionAuthoritySnapshot(incomingAuthority)
	if err != nil {
		return err
	}
	if receipt.SourceAuthorityPolicy != FirstParentTargetSourceAuditOnlyPolicy ||
		!slices.Equal(receipt.AuditedIncomingCompletions, auditedCompletions) ||
		!slices.Equal(receipt.AuditedIncomingRetirements, auditedRetirements) {
		return errors.New("plan: projected merge receipt incoming authority audit differs")
	}
	return nil
}

// VerifyFirstParentTargetLocalAuthority proves the source-free target side of
// one projected merge. The first parent must be exact and protected; the
// completion may consume either an existing target row (preAdvance == local)
// or one ephemeral merge row (child == local). Reuse, retirement, and pruned
// dependencies are checked only against parent-0 authority, so an audited
// source completion can never authorize target work.
func VerifyFirstParentTargetLocalAuthority(
	repository, localRevision string,
	local, preAdvance, child Plan,
	localAuthority CompletionAuthority,
) error {
	if err := Validate(preAdvance); err != nil {
		return fmt.Errorf("plan: invalid first-parent target pre-advance plan: %w", err)
	}
	if err := Validate(child); err != nil {
		return fmt.Errorf("plan: invalid first-parent target child plan: %w", err)
	}
	if !localAuthority.resolvesAt(local, repository, localRevision) {
		return errors.New("plan: first-parent target lacks authority for the exact local parent")
	}
	if !localAuthority.ProtectsRevision() {
		return errors.New("plan: first-parent target local parent is outside the protected completion epoch")
	}
	if err := verifyFirstParentTargetPlanOwnership(local, preAdvance, child); err != nil {
		return err
	}
	if !preservesPlanIdentities(local, preAdvance, false) {
		return errors.New("plan: first-parent target pre-advance plan deleted a local item or step")
	}
	return verifyCompletionAuthorityConstraints(
		preAdvance, localAuthority.completedReferences, localAuthority.retiredItems,
	)
}

func verifyFirstParentTargetPlanOwnership(local, preAdvance, child Plan) error {
	preAdvanceIsLocal, err := exactCanonicalJSONEqual("projected merge target pre-advance plan", preAdvance, local)
	if err != nil {
		return err
	}
	childIsLocal, err := exactCanonicalJSONEqual("projected merge target child plan", child, local)
	if err != nil {
		return err
	}
	if !preAdvanceIsLocal && !childIsLocal {
		return errors.New("plan: projected merge completion adopts rows outside the first-parent target plan")
	}
	return nil
}

// VerifyFirstParentTargetMergeAuthorityTransition validates the portable
// portion of a projected-merge receipt. It deliberately needs neither parent
// authority stores nor their opaque in-memory authorities, so gate recovery
// can revalidate exact receipt bytes after the source worktree disappears.
func VerifyFirstParentTargetMergeAuthorityTransition(
	localRevision, incomingRevision, mergeBaseRevision string,
	local, incoming, mergeBase, preAdvance, child Plan,
	item, step string,
	preparation artifact.ID,
	preparationCommit artifact.CommitID,
	receipt FirstParentTargetMergeAuthority,
) error {
	if err := firstParentTargetMergeAuthorityCodec.ValidateIdentity(receipt); err != nil {
		return err
	}
	if !worklease.ValidCommit(mergeBaseRevision) || receipt.LocalParent != strings.TrimSpace(localRevision) ||
		receipt.IncomingParent != strings.TrimSpace(incomingRevision) || receipt.MergeBase != strings.TrimSpace(mergeBaseRevision) {
		return errors.New("plan: projected merge receipt differs from the exact Git parents or merge base")
	}
	if receipt.PlanItem != item || receipt.PlanStep != step || receipt.Preparation != preparation ||
		receipt.PreparationCommit != preparationCommit {
		return errors.New("plan: projected merge receipt differs from the completion authority tuple")
	}
	expectedChild, err := Advance(preAdvance, item, step)
	if err != nil {
		return fmt.Errorf("plan: derive projected merge completion: %w", err)
	}
	if equal, equalErr := exactCanonicalJSONEqual("projected merge child plan", expectedChild, child); equalErr != nil {
		return equalErr
	} else if !equal {
		return errors.New("plan: projected merge receipt child is not the exact completion transition")
	}
	if err := verifyFirstParentTargetPlanOwnership(local, preAdvance, child); err != nil {
		return err
	}
	if receipt.LocalPlanDigest != local.Digest() ||
		receipt.IncomingPlanDigest != incoming.Digest() ||
		receipt.MergeBasePlanDigest != mergeBase.Digest() ||
		receipt.PreAdvancePlanDigest != preAdvance.Digest() ||
		receipt.ChildPlanDigest != child.Digest() {
		return errors.New("plan: projected merge receipt plan digest differs")
	}
	return nil
}

// Content returns the canonical receipt document for the gate's final batch.
func (value FirstParentTargetMergeAuthority) Content() (artifact.Content, error) {
	return firstParentTargetMergeAuthorityCodec.Content(value)
}

// Lineage binds the receipt to the exact gate preparation that authorized its
// eventual merge completion. The successful GateResult separately depends on
// this receipt in the same atomic final batch.
func (value FirstParentTargetMergeAuthority) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Preparation)
}

// ParseFirstParentTargetMergeAuthority decodes one canonical receipt.
func ParseFirstParentTargetMergeAuthority(data []byte) (FirstParentTargetMergeAuthority, error) {
	return firstParentTargetMergeAuthorityCodec.Parse(data)
}

// RequireFirstParentTargetMergeAuthority loads one canonical receipt and its
// exact preparation lineage.
func RequireFirstParentTargetMergeAuthority(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (FirstParentTargetMergeAuthority, error) {
	return firstParentTargetMergeAuthorityCodec.RequireExactLineage(
		ctx, reader, id, FirstParentTargetMergeAuthority.Lineage,
	)
}

func requireHistoricalFirstParentTargetMergeAuthority(
	ctx context.Context,
	repository string,
	store *overgodb.Store,
	commit gitCompletionMessage,
	trailers completionTrailers,
	transition completionTransition,
	evidence completionEvidence,
	localAuthority CompletionAuthority,
) (FirstParentTargetMergeAuthority, error) {
	if len(commit.parents) != completionParentCount ||
		trailers.mergeProjection != MergeProjectionFirstParentTarget ||
		trailers.mergeAuthority.Kind() != artifact.KindEvidence {
		return FirstParentTargetMergeAuthority{}, errors.New("plan: invalid historical first-parent target boundary")
	}
	receipt, err := RequireFirstParentTargetMergeAuthority(ctx, store, trailers.mergeAuthority)
	if err != nil {
		return FirstParentTargetMergeAuthority{}, fmt.Errorf("plan: load projected merge authority: %w", err)
	}
	preAdvance, err := reconstructCompletionPrePlan(transition.child, trailers)
	if err != nil {
		return FirstParentTargetMergeAuthority{}, err
	}
	if err := VerifyFirstParentTargetMergeAuthorityTransition(
		commit.parents[completionLocalParentIndex], commit.parents[completionIncomingParentIndex],
		transition.mergeBaseRevision, transition.local, transition.incoming, transition.mergeBase,
		preAdvance, transition.child, trailers.item, trailers.step, trailers.preparation,
		trailers.preparationCommit, receipt,
	); err != nil {
		return FirstParentTargetMergeAuthority{}, err
	}
	if !localAuthority.resolvesAt(
		transition.local, repository, commit.parents[completionLocalParentIndex],
	) {
		return FirstParentTargetMergeAuthority{}, errors.New(
			"plan: projected merge local boundary lacks exact completion authority",
		)
	}
	if err := VerifyFirstParentTargetLocalAuthority(
		repository, commit.parents[completionLocalParentIndex], transition.local,
		preAdvance, transition.child, localAuthority,
	); err != nil {
		return FirstParentTargetMergeAuthority{}, err
	}
	localDigest, err := completionAuthorityDigest(localAuthority)
	if err != nil {
		return FirstParentTargetMergeAuthority{}, err
	}
	if receipt.LocalAuthorityDigest != localDigest {
		return FirstParentTargetMergeAuthority{}, errors.New(
			"plan: projected merge receipt differs from its exact local boundary authority",
		)
	}
	localSeeds := sortedProtectionSeeds(localAuthority)
	if !slices.Equal(receipt.LocalProtectionSeeds, localSeeds) {
		return FirstParentTargetMergeAuthority{}, errors.New("plan: projected merge authority differs from local protection epoch")
	}
	targetCoordinate, found, err := store.CommitAt(ctx, receipt.TargetStore.Sequence)
	if err != nil {
		return FirstParentTargetMergeAuthority{}, err
	}
	if !found || targetCoordinate.ID != receipt.TargetStore.Commit {
		return FirstParentTargetMergeAuthority{}, errors.New("plan: projected merge target-store coordinate is absent")
	}
	prefixCoordinate, found, err := store.CommitAt(ctx, receipt.SharedStorePrefix.Sequence)
	if err != nil {
		return FirstParentTargetMergeAuthority{}, err
	}
	if !found || prefixCoordinate.ID != receipt.SharedStorePrefix.Commit {
		return FirstParentTargetMergeAuthority{}, errors.New("plan: projected merge shared-store prefix is absent")
	}
	receiptIntroduction, found, err := store.ArtifactIntroduction(ctx, receipt.ID)
	if err != nil {
		return FirstParentTargetMergeAuthority{}, err
	}
	if !found || receiptIntroduction.Sequence <= receipt.TargetStore.Sequence {
		return FirstParentTargetMergeAuthority{}, errors.New("plan: projected merge receipt does not follow its target-store observation")
	}
	for label, id := range map[string]artifact.ID{
		"attempt": evidence.attempt, "gate result": evidence.result, "finalization": evidence.finalization,
	} {
		introduction, introduced, introductionErr := store.ArtifactIntroduction(ctx, id)
		if introductionErr != nil {
			return FirstParentTargetMergeAuthority{}, introductionErr
		}
		if !introduced || introduction.Commit != receiptIntroduction.Commit ||
			introduction.Sequence != receiptIntroduction.Sequence {
			return FirstParentTargetMergeAuthority{}, fmt.Errorf(
				"plan: projected merge receipt was not published atomically with its %s", label,
			)
		}
	}
	parents, err := store.Parents(ctx, evidence.result)
	if err != nil {
		return FirstParentTargetMergeAuthority{}, err
	}
	projectedReceipts := make([]artifact.ID, 0, 1)
	for _, edge := range parents {
		if edge.Child != evidence.result || edge.Relation != artifact.RelationDependsOn ||
			edge.Parent.Kind() != artifact.KindEvidence {
			continue
		}
		content, typed, readErr := artifact.ReadContent(ctx, store, edge.Parent)
		if readErr != nil {
			return FirstParentTargetMergeAuthority{}, readErr
		}
		if typed && content.Descriptor.MediaType == FirstParentTargetMergeAuthorityMediaType &&
			content.Descriptor.Schema == FirstParentTargetMergeAuthoritySchema {
			projectedReceipts = append(projectedReceipts, edge.Parent)
		}
	}
	if len(projectedReceipts) != 1 || projectedReceipts[0] != receipt.ID {
		return FirstParentTargetMergeAuthority{}, errors.New("plan: gate result lacks one exact projected merge receipt lineage")
	}
	return receipt, nil
}

func canonicalizeFirstParentTargetMergeAuthority(value *FirstParentTargetMergeAuthority) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Projection != MergeProjectionFirstParentTarget ||
		!worklease.ValidCommit(value.LocalParent) || !worklease.ValidCommit(value.IncomingParent) ||
		!worklease.ValidCommit(value.MergeBase) || value.LocalParent == value.IncomingParent ||
		!validDigestText(value.LocalPlanDigest) || !validDigestText(value.IncomingPlanDigest) ||
		!validDigestText(value.MergeBasePlanDigest) || !validDigestText(value.PreAdvancePlanDigest) ||
		!validDigestText(value.ChildPlanDigest) || !validDigestText(value.LocalAuthorityDigest) ||
		!validDigestText(value.IncomingAuthorityDigest) || !worklease.ValidPlanID(value.PlanItem) ||
		!worklease.ValidPlanID(value.PlanStep) || value.Preparation.Kind() != artifact.KindEvidence ||
		!value.PreparationCommit.Valid() || !validStoreCoordinate(value.TargetStore) ||
		!validStoreCoordinate(value.SourceStore) || !value.SharedStorePrefix.Commit.Valid() ||
		value.SharedStorePrefix.Sequence == 0 || value.SharedStorePrefix.Sequence > value.TargetStore.Sequence ||
		value.SharedStorePrefix.Sequence > value.SourceStore.Sequence {
		return errors.New("plan: invalid first-parent target merge authority")
	}
	if err := canonicalProtectionSeeds(value.LocalProtectionSeeds); err != nil {
		return fmt.Errorf("plan: invalid local protection seeds: %w", err)
	}
	if err := canonicalProtectionSeeds(value.IncomingProtectionSeeds); err != nil {
		return fmt.Errorf("plan: invalid incoming protection seeds: %w", err)
	}
	if value.SourceAuthorityPolicy != FirstParentTargetSourceAuditOnlyPolicy {
		return errors.New("plan: first-parent target receipt lacks source-audit-only policy")
	}
	if err := validateProjectedAuthoritySnapshot(
		value.AuditedIncomingCompletions, value.AuditedIncomingRetirements,
	); err != nil {
		return fmt.Errorf("plan: invalid audited incoming authority: %w", err)
	}
	incomingDigest, err := completionAuthoritySnapshotDigest(
		value.IncomingParent, value.IncomingPlanDigest, value.IncomingProtectionSeeds,
		value.AuditedIncomingCompletions, value.AuditedIncomingRetirements,
	)
	if err != nil {
		return err
	}
	if value.IncomingAuthorityDigest != incomingDigest {
		return errors.New("plan: audited incoming authority digest differs")
	}
	return nil
}

func cloneFirstParentTargetMergeAuthority(value FirstParentTargetMergeAuthority) FirstParentTargetMergeAuthority {
	value.LocalProtectionSeeds = slices.Clone(value.LocalProtectionSeeds)
	value.IncomingProtectionSeeds = slices.Clone(value.IncomingProtectionSeeds)
	value.AuditedIncomingCompletions = slices.Clone(value.AuditedIncomingCompletions)
	value.AuditedIncomingRetirements = slices.Clone(value.AuditedIncomingRetirements)
	return value
}

func validStoreCoordinate(value CompletionStoreCoordinate) bool {
	return value.Commit.Valid() && value.Sequence != 0
}

func validDigestText(value string) bool {
	if len(value) != hex.EncodedLen(sha256.Size) || value != strings.ToLower(value) {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size
}

func sortedProtectionSeeds(authority CompletionAuthority) []string {
	return slices.Sorted(maps.Keys(authority.protectionSeeds))
}

func canonicalProtectionSeeds(seeds []string) error {
	if len(seeds) == 0 {
		return errors.New("protection seeds are absent")
	}
	for index, seed := range seeds {
		if !worklease.ValidCommit(seed) || index != 0 && seeds[index-1] >= seed {
			return errors.New("protection seeds are not uniquely sorted")
		}
	}
	return nil
}

// validateProjectedRetirementAuthority keeps the completion and item-level
// tombstone indexes bidirectionally exact before either can cross a projected
// boundary. A retired item must name exactly one retired completion under that
// item, and every retired completion must be that item's tombstone.
func validateProjectedRetirementAuthority(label string, authority CompletionAuthority) error {
	for item, retirement := range authority.retiredItems {
		matches := 0
		if !retirement.retiredItem {
			return fmt.Errorf("plan: projected merge %s retirement authority for %s is not retired", label, item)
		}
		for reference, completion := range authority.completedReferences {
			if strings.HasPrefix(reference, item+"/") && completion == retirement {
				matches++
			}
		}
		if matches != 1 {
			return fmt.Errorf(
				"plan: projected merge %s retirement authority for %s has %d exact completion references",
				label, item, matches,
			)
		}
	}
	for reference, completion := range authority.completedReferences {
		if !completion.retiredItem {
			continue
		}
		item, _, _ := strings.Cut(reference, "/")
		if retirement, found := authority.retiredItems[item]; !found || retirement != completion {
			return fmt.Errorf(
				"plan: projected merge %s retired completion %s lacks its exact item authority",
				label, reference,
			)
		}
	}
	return nil
}

func exportProjectedCompletion(reference string, evidence completionEvidence) ProjectedCompletionEvidence {
	return ProjectedCompletionEvidence{
		Reference: reference, Commit: evidence.commit, Verify: evidence.verify,
		ContractDigest: hex.EncodeToString(evidence.contract[:]),
		Manifest:       evidence.manifest, CodeManifest: evidence.codeManifest,
		Attempt: evidence.attempt, Result: evidence.result, Finalization: evidence.finalization,
		RetiredItem: evidence.retiredItem,
	}
}

func validateProjectedCompletionEvidence(value ProjectedCompletionEvidence) error {
	item, step, found := strings.Cut(value.Reference, "/")
	if !found || !worklease.ValidPlanID(item) || !worklease.ValidPlanID(step) || strings.Contains(step, "/") ||
		!worklease.ValidCommit(value.Commit) || value.Verify != strings.TrimSpace(value.Verify) ||
		!validAutomationDetail(value.Verify) || !validDigestText(value.ContractDigest) ||
		value.ContractDigest == strings.Repeat("0", hex.EncodedLen(sha256.Size)) ||
		value.Manifest.Kind() != artifact.KindRecipe ||
		value.CodeManifest.Kind() != artifact.KindProfile || value.Attempt.Kind() != artifact.KindEvidence ||
		value.Result.Kind() != artifact.KindEvidence || value.Finalization.Kind() != artifact.KindEvidence {
		return errors.New("plan: invalid projected completion evidence")
	}
	return nil
}

type completionAuthorityDigestDocument struct {
	Revision            string                        `json:"revision"`
	PlanDigest          string                        `json:"plan_digest"`
	ProtectionSeeds     []string                      `json:"protection_seeds"`
	CompletedReferences []ProjectedCompletionEvidence `json:"completed_references"`
	RetiredItems        []ProjectedRetirementEvidence `json:"retired_items"`
}

func completionAuthoritySnapshot(
	authority CompletionAuthority,
) ([]ProjectedCompletionEvidence, []ProjectedRetirementEvidence, error) {
	if err := validateProjectedRetirementAuthority("digest", authority); err != nil {
		return nil, nil, err
	}
	completed := make([]ProjectedCompletionEvidence, 0, len(authority.completedReferences))
	for reference, evidence := range authority.completedReferences {
		completed = append(completed, exportProjectedCompletion(reference, evidence))
	}
	slices.SortFunc(completed, func(left, right ProjectedCompletionEvidence) int {
		return cmp.Compare(left.Reference, right.Reference)
	})
	retired := make([]ProjectedRetirementEvidence, 0, len(authority.retiredItems))
	for item, evidence := range authority.retiredItems {
		reference := ""
		for candidate, completion := range authority.completedReferences {
			if completion == evidence && strings.HasPrefix(candidate, item+"/") {
				if reference != "" {
					return nil, nil, fmt.Errorf("plan: retirement authority for %s is ambiguous", item)
				}
				reference = candidate
			}
		}
		if reference == "" {
			return nil, nil, fmt.Errorf("plan: retirement authority for %s lacks completion evidence", item)
		}
		retired = append(retired, ProjectedRetirementEvidence{
			Item: item, Evidence: exportProjectedCompletion(reference, evidence),
		})
	}
	slices.SortFunc(retired, func(left, right ProjectedRetirementEvidence) int {
		return cmp.Compare(left.Item, right.Item)
	})
	return completed, retired, nil
}

func completionAuthoritySnapshotDigest(
	revision, planDigest string,
	protectionSeeds []string,
	completed []ProjectedCompletionEvidence,
	retired []ProjectedRetirementEvidence,
) (string, error) {
	document := completionAuthorityDigestDocument{
		Revision: revision, PlanDigest: planDigest, ProtectionSeeds: protectionSeeds,
		CompletedReferences: completed, RetiredItems: retired,
	}
	encoded, err := json.Marshal(document)
	if err != nil {
		return "", fmt.Errorf("plan: encode completion authority digest: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return hex.EncodeToString(digest[:]), nil
}

func completionAuthorityDigest(authority CompletionAuthority) (string, error) {
	completed, retired, err := completionAuthoritySnapshot(authority)
	if err != nil {
		return "", err
	}
	return completionAuthoritySnapshotDigest(
		authority.revision, hex.EncodeToString(authority.planDigest[:]),
		sortedProtectionSeeds(authority), completed, retired,
	)
}

func validateProjectedAuthoritySnapshot(
	completed []ProjectedCompletionEvidence,
	retired []ProjectedRetirementEvidence,
) error {
	if completed == nil || retired == nil {
		return errors.New("authority snapshot arrays are absent")
	}
	completionIndex := make(map[string]ProjectedCompletionEvidence, len(completed))
	for index, evidence := range completed {
		if err := validateProjectedCompletionEvidence(evidence); err != nil {
			return err
		}
		if index != 0 && completed[index-1].Reference >= evidence.Reference {
			return errors.New("completion evidence is not uniquely sorted")
		}
		completionIndex[evidence.Reference] = evidence
	}
	retirementIndex := make(map[string]ProjectedRetirementEvidence, len(retired))
	for index, retirement := range retired {
		if !worklease.ValidPlanID(retirement.Item) ||
			!strings.HasPrefix(retirement.Evidence.Reference, retirement.Item+"/") ||
			!retirement.Evidence.RetiredItem {
			return errors.New("invalid retirement evidence")
		}
		if index != 0 && retired[index-1].Item >= retirement.Item {
			return errors.New("retirement evidence is not uniquely sorted")
		}
		completion, found := completionIndex[retirement.Evidence.Reference]
		if !found || completion != retirement.Evidence {
			return errors.New("retirement evidence differs from its completion")
		}
		retirementIndex[retirement.Item] = retirement
	}
	for _, completion := range completed {
		if !completion.RetiredItem {
			continue
		}
		item, _, _ := strings.Cut(completion.Reference, "/")
		retirement, found := retirementIndex[item]
		if !found || retirement.Evidence != completion {
			return errors.New("retired completion lacks its exact item authority")
		}
	}
	return nil
}
