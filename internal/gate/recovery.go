package gate

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/atomicfile"
	"overgo/internal/automationcheck"
	"overgo/internal/clioptions"
	"overgo/internal/fsatomic"
	"overgo/internal/gitauthority"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/processmeasure"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
)

func restoreCapturedIndex(repo string, intent gateCommitIntent) error {
	err := withGateGitStateLock(
		repo, fs.FileMode(intent.IndexMode), intent,
		gateIndexCASBeforeLockHook, gateIndexCASLockedHook,
		func(indexPath string) error {
			if err := requireGateIntentKeepalive(repo, intent); err != nil {
				return err
			}
			if err := normalizeGateIndexToStates(
				repo, indexPath, fs.FileMode(intent.IndexMode),
				intent.IndexBefore, intent.IndexAfter, intent.IndexRestore,
			); err != nil {
				return err
			}
			return replaceGateIndexStatesUnderLock(
				indexPath,
				[][]byte{intent.IndexAfter},
				[][]byte{intent.IndexBefore, intent.IndexRestore},
				intent.IndexRestore,
				fs.FileMode(intent.IndexMode),
			)
		},
	)
	if err != nil {
		if errors.Is(err, errGateIndexMoved) {
			return errors.New("gate: Git index moved beyond the write-ahead states; automatic recovery refused")
		}
		return err
	}
	return nil
}

// requireGateDebtAdmission revalidates the append-only lifecycle immediately
// before a saved final batch is reconciled. A fresh debt may advance only the
// exact still-current preparation with no finalization. One matching terminal
// lifecycle is accepted only as an idempotent replay candidate: its alias must
// already be terminal, and Store.Commit below proves the same batch key and
// payload before returning success. An unaliased or different finalization can
// therefore never be followed by a second finalization from stale debt.
func requireGateDebtAdmission(
	ctx context.Context,
	store *overgodb.Store,
	preparationID artifact.ID,
	debtFinalization runrecord.GateLifecycle,
) error {
	if ctx == nil || store == nil || preparationID.Kind() != artifact.KindEvidence ||
		debtFinalization.State != runrecord.GateFinalized || debtFinalization.Preparation == nil ||
		*debtFinalization.Preparation != preparationID {
		return errors.New("gate: record debt lacks exact lifecycle authority")
	}
	preparation, err := runrecord.RequireGateLifecycle(ctx, store, preparationID)
	if err != nil {
		return fmt.Errorf("gate: debt preparation is unavailable: %w", err)
	}
	if preparation.State != runrecord.GatePrepared ||
		debtFinalization.TreeKey != preparation.TreeKey ||
		debtFinalization.Environment != preparation.Environment ||
		debtFinalization.Started != preparation.Started {
		return errors.New("gate: debt finalization contradicts its preparation")
	}
	finalization, finalized, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
	if err != nil {
		return err
	}
	if !finalized {
		introduction, found, err := store.ArtifactIntroduction(ctx, preparation.ID)
		if err != nil {
			return err
		}
		if !found {
			return errors.New("gate: debt preparation has no durable introduction authority")
		}
		return requireSoleCurrentGatePreparation(ctx, store, preparation, introduction.Commit)
	}
	if finalization.ID != debtFinalization.ID {
		return errors.New("gate: record debt preparation already has another finalization")
	}
	current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		return err
	}
	if !found || current != debtFinalization.ID {
		return errors.New("gate: matching debt finalization is not terminal lifecycle authority")
	}
	if err := validateCompleteGateFinalization(ctx, store, preparation, finalization); err != nil {
		return fmt.Errorf("gate: matching debt finalization is incomplete: %w", err)
	}
	return nil
}

func (g *gateContext) heartbeat(state runrecord.GateHeartbeatState) runrecord.GateHeartbeat {
	return runrecord.GateHeartbeat{
		Version: artifact.InitialDocumentVersion, State: state, Preparation: g.preparation.ID,
		TreeKey: g.preparation.TreeKey, Environment: g.environment.ID,
		PID: os.Getpid(), Updated: time.Now().UTC(),
	}
}

func (g *gateContext) writeHeartbeat(state runrecord.GateHeartbeatState) error {
	heartbeat := g.heartbeat(state)
	if err := heartbeat.Validate(); err != nil {
		return err
	}
	return writeJSON(g.repo, gateHeartbeatFile, heartbeat, clioptions.OutputFileMode)
}

type gateDebtEnvelope struct {
	Version     uint16         `json:"version"`
	Preparation artifact.ID    `json:"preparation"`
	Batch       artifact.Batch `json:"batch"`
}

func recoverInterruptedCommit(repo, storePath string) (recovered artifact.ID, err error) {
	recoveryStarted := processmeasure.NewStopwatch()
	if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(gateDebtFile))); err == nil {
		return artifact.ID{}, errors.New("gate: record debt exists for the interrupted commit; run `go run ./cmd/gate -reconcile`")
	} else if !errors.Is(err, os.ErrNotExist) {
		return artifact.ID{}, err
	}
	if err := gitauthority.RequireCompleteHistory(context.Background(), repo); err != nil {
		return artifact.ID{}, fmt.Errorf("gate: %w", err)
	}
	var intent gateCommitIntent
	if err := readJSON(repo, gateCommitIntentFile, &intent); err != nil {
		return artifact.ID{}, fmt.Errorf("read interrupted commit intent: %w", err)
	}
	if err := intent.validate(); err != nil {
		return artifact.ID{}, err
	}
	if err := validateGateIntentIndexes(repo, intent); err != nil {
		return artifact.ID{}, err
	}
	// The reachability ref is published before the intent, so a crash can leave
	// only a harmless orphan ref, never a durable intent whose rollback objects
	// were exposed to pruning. Requiring it here rejects any later ref drift;
	// every destructive transition below repeats that check under lock.
	if err := ensureGateIntentKeepalive(repo, intent); err != nil {
		return artifact.ID{}, err
	}
	defer func() {
		if err == nil && recovered.Valid() {
			err = finalizeMatchingHeartbeat(repo, recovered)
		}
	}()
	headReference, err := currentHeadReference(repo)
	if err != nil {
		return artifact.ID{}, err
	}
	if headReference != intent.HeadReference {
		return artifact.ID{}, fmt.Errorf(
			"gate: HEAD reference moved from %s to %s; automatic recovery refused",
			intent.HeadReference, headReference,
		)
	}
	head, err := command(repo, "git", "rev-parse", "HEAD")
	if err != nil {
		return artifact.ID{}, err
	}
	head = strings.TrimSpace(head)
	if intent.Commit == "" && head != intent.Parent {
		return artifact.ID{}, errors.New(
			"gate: HEAD moved before the interrupted intent durably named its commit; automatic recovery refused",
		)
	}
	if intent.Commit != "" {
		if err := validateInterruptedCommitIdentity(repo, intent); err != nil {
			return artifact.ID{}, err
		}
	}
	if head != intent.Parent && head != intent.Commit {
		return artifact.ID{}, fmt.Errorf(
			"gate: HEAD %.12s moved beyond interrupted commit %.12s; automatic recovery refused",
			head, intent.Commit,
		)
	}
	store, err := overgodb.OpenContext(context.Background(), filepath.Join(repo, storePath))
	if err != nil {
		return artifact.ID{}, err
	}
	defer store.Close()
	ctx := context.Background()
	preparation, err := runrecord.RequireGateLifecycle(ctx, store, intent.Preparation)
	if err != nil || preparation.State != runrecord.GatePrepared {
		return artifact.ID{}, errors.Join(errors.New("gate: interrupted preparation is unavailable"), err)
	}
	finalization, finalized, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
	if err != nil {
		return artifact.ID{}, err
	}
	if finalized {
		switch finalization.Outcome {
		case runrecord.OutcomeSucceeded:
			if intent.Commit == "" || finalization.CodeCommit != intent.Commit {
				return artifact.ID{}, errors.New("gate: successful interrupted finalization contradicts Git")
			}
			if head == intent.Parent {
				planRestored, planErr := planBytesMatch(repo, intent.Plan)
				if planErr != nil {
					return artifact.ID{}, planErr
				}
				if planRestored {
					if err := removeGateCommitIntent(repo); err != nil {
						return artifact.ID{}, err
					}
					return preparation.ID, nil
				}
			}
			if head == intent.Commit {
				if err := validateInterruptedCommitAuthority(repo, intent, store); err != nil {
					return artifact.ID{}, err
				}
				complete, completionErr := interruptedCompletionRecorded(ctx, store, intent)
				if completionErr != nil {
					return artifact.ID{}, completionErr
				}
				if complete {
					if err := clearCommittedMergeState(repo, intent); err != nil {
						return artifact.ID{}, err
					}
					if err := removeGateCommitIntent(repo); err != nil {
						return artifact.ID{}, err
					}
					return preparation.ID, nil
				}
			}
			// The store is append-only, so an incomplete success record cannot
			// be erased. Keep it unreachable by rolling Git and the plan back;
			// no cancellation is added because that would make the preparation's
			// finalization lineage ambiguous.
		case runrecord.OutcomeCancelled:
			planRestored, planErr := planBytesMatch(repo, intent.Plan)
			if planErr != nil {
				return artifact.ID{}, planErr
			}
			if head != intent.Parent || !planRestored {
				return artifact.ID{}, errors.New("gate: interrupted cancellation contradicts Git or the restored plan")
			}
			complete, completionErr := interruptedCancellationRecorded(ctx, store, intent, preparation, finalization)
			if completionErr != nil {
				return artifact.ID{}, completionErr
			}
			if !complete {
				return artifact.ID{}, errors.New("gate: interrupted cancellation is not the exact recovery record")
			}
			if err := removeGateCommitIntent(repo); err != nil {
				return artifact.ID{}, err
			}
			return preparation.ID, nil
		default:
			return artifact.ID{}, errors.New("gate: interrupted preparation already has a non-recoverable finalization")
		}
	}
	if err := restoreInterruptedState(repo, intent, head); err != nil {
		return artifact.ID{}, err
	}
	if !finalized {
		codeCommit := intent.Parent
		if intent.Commit != "" {
			codeCommit = intent.Commit
		}
		durationNS, err := completedGateMeasurement(recoveryStarted, "interrupted recovery")
		if err != nil {
			return artifact.ID{}, err
		}
		record, err := runrecord.NewGateRecord(
			intent.Recipe, preparation.Environment, codeCommit, runrecord.OutcomeCancelled, "", durationNS,
			[]runrecord.GateStep{{
				Name: "recovery", Phase: runrecord.PhaseValidate,
				Outcome: runrecord.StepCancelled, DurationNS: durationNS,
			}},
		)
		if err != nil {
			return artifact.ID{}, err
		}
		batch, err := record.Batch("gate/recovered/" + preparation.ID.String())
		if err != nil {
			return artifact.ID{}, err
		}
		finalized, err := runrecord.NewGateFinalization(preparation, codeCommit, record.Result.ID, runrecord.OutcomeCancelled)
		if err != nil {
			return artifact.ID{}, err
		}
		finalizedContent, err := finalized.Content()
		if err != nil {
			return artifact.ID{}, err
		}
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: intent.Recipe})
		batch.Contents = append(batch.Contents, finalizedContent)
		batch.Lineage = append(batch.Lineage, finalized.Lineage()...)
		appendGateFinalizationAlias(&batch, preparation.ID, finalized.ID)
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			return artifact.ID{}, err
		}
	}
	if err := removeGateCommitIntent(repo); err != nil {
		return artifact.ID{}, err
	}
	return preparation.ID, nil
}

func interruptedCompletionRecorded(
	ctx context.Context,
	store *overgodb.Store,
	intent gateCommitIntent,
) (bool, error) {
	introduction, found, err := store.ArtifactIntroduction(ctx, intent.Preparation)
	if err != nil {
		return false, err
	}
	if !found || introduction.Commit != intent.PreparationCommit {
		return false, nil
	}
	attempts, err := runrecord.AttemptsForPreparation(ctx, store, intent.Preparation)
	if err != nil {
		return false, err
	}
	item, step, _ := strings.Cut(intent.PlanRef, "/")
	var matches []runrecord.AttemptRecord
	for _, attempt := range attempts {
		if attempt.PlanItem == item && attempt.PlanStep == step && attempt.Recipe == intent.Recipe &&
			attempt.CandidateManifest == intent.CandidateManifest && attempt.CodeCommit == intent.Commit &&
			attempt.Outcome == runrecord.OutcomeSucceeded {
			matches = append(matches, attempt)
		}
	}
	if len(matches) > 1 {
		return false, errors.New("gate: interrupted completion has ambiguous successful attempts")
	}
	if len(matches) == 0 {
		return false, nil
	}
	verification, err := runrecord.VerifyAttemptGate(ctx, store, matches[0])
	if err != nil || verification.Preparation.ID != intent.Preparation {
		return false, nil
	}
	document, err := plan.Parse(intent.Plan)
	if err != nil {
		return false, err
	}
	var required plan.Step
	for _, candidateItem := range document.Items {
		if candidateItem.ID != item {
			continue
		}
		for _, candidateStep := range candidateItem.Steps {
			if candidateStep.ID == step && candidateStep.Status == plan.StatusOpen {
				required = candidateStep
				break
			}
		}
	}
	if required.Verify == "" {
		return false, errors.New("gate: interrupted completion row is absent from its recorded plan")
	}
	if err := plan.VerifyPreparedStepAcceptance(ctx, store, verification, matches[0], intent.PlanRef, required); err != nil {
		return false, nil
	}
	return true, nil
}

func interruptedCancellationRecorded(
	ctx context.Context,
	store *overgodb.Store,
	intent gateCommitIntent,
	preparation, finalization runrecord.GateLifecycle,
) (bool, error) {
	if finalization.Result == nil || finalization.Preparation == nil ||
		*finalization.Preparation != preparation.ID {
		return false, nil
	}
	codeCommit := intent.Parent
	if intent.Commit != "" {
		codeCommit = intent.Commit
	}
	if finalization.CodeCommit != codeCommit {
		return false, nil
	}
	gate, err := runrecord.RequireGateResult(ctx, store, *finalization.Result)
	if err != nil {
		return false, nil
	}
	if gate.Recipe != intent.Recipe || gate.Environment != preparation.Environment ||
		gate.CodeCommit != codeCommit || gate.Outcome != runrecord.OutcomeCancelled || gate.Failure != "" ||
		len(gate.Steps) != 1 || gate.Steps[0].Name != "recovery" ||
		gate.Steps[0].Phase != runrecord.PhaseValidate || gate.Steps[0].Outcome != runrecord.StepCancelled {
		return false, nil
	}
	gateIntroduction, gateFound, err := store.ArtifactIntroduction(ctx, gate.ID)
	if err != nil || !gateFound {
		return false, err
	}
	finalIntroduction, finalFound, err := store.ArtifactIntroduction(ctx, finalization.ID)
	if err != nil || !finalFound {
		return false, err
	}
	return gateIntroduction.Commit == finalIntroduction.Commit &&
		gateIntroduction.Sequence == finalIntroduction.Sequence, nil
}

func validateInterruptedCommitAuthority(repo string, intent gateCommitIntent, store *overgodb.Store) error {
	if err := gitauthority.RequireCompleteHistory(context.Background(), repo); err != nil {
		return fmt.Errorf("gate: %w", err)
	}
	if err := validateInterruptedCommitIdentity(repo, intent); err != nil {
		return err
	}
	messageBytes, err := gitAuthorityOutput(repo, "show", "-s", "--format=%B", intent.Commit)
	if err != nil {
		return err
	}
	message := string(messageBytes)
	preAdvance, err := plan.Parse(intent.Plan)
	if err != nil {
		return err
	}
	child, err := plan.Parse(intent.AdvancedPlan)
	if err != nil {
		return err
	}
	item, step, found := strings.Cut(intent.PlanRef, "/")
	if !found {
		return errors.New("gate: interrupted commit has an invalid completion reference")
	}
	mergeAuthority, err := projectedMergeAuthorityFromIntent(intent)
	if err != nil {
		return err
	}
	mergeAuthorityID := artifact.ID{}
	if mergeAuthority != nil {
		mergeAuthorityID = mergeAuthority.ID
	}
	if err := plan.VerifyCompletionCommitMessageWithMergeAuthority(
		message, preAdvance, item, step, intent.Recipe, intent.CandidateManifest,
		intent.Preparation, intent.PreparationCommit, intent.PlanProjection, mergeAuthorityID,
	); err != nil {
		return fmt.Errorf("gate: interrupted commit has the wrong completion authority: %w", err)
	}
	if err := verifyProspectiveGateCompletion(repo, intent, preAdvance, child, message, store); err != nil {
		return fmt.Errorf("gate: interrupted commit has the wrong completion authority: %w", err)
	}
	return nil
}

func validateInterruptedCommitIdentity(repo string, intent gateCommitIntent) error {
	parentsOutput, err := gitAuthorityOutput(repo, "show", "-s", "--format=%P", intent.Commit)
	if err != nil {
		return err
	}
	parentFields := strings.Fields(string(parentsOutput))
	wantParents := []string{intent.Parent}
	if intent.Merge != nil {
		wantParents = append(wantParents, intent.Merge.parents()...)
	}
	if !slices.Equal(parentFields, wantParents) {
		return errors.New("gate: interrupted commit does not descend from its recorded parent")
	}
	treeOutput, err := gitAuthorityOutput(repo, "rev-parse", intent.Commit+"^{tree}")
	if err != nil {
		return err
	}
	if strings.TrimSpace(string(treeOutput)) != intent.Tree {
		return errors.New("gate: interrupted commit tree differs from its write-ahead intent")
	}
	planBytes, err := gitCompletionFile(repo, intent.Commit, plan.Path)
	if err != nil {
		return err
	}
	if !bytes.Equal(planBytes, intent.AdvancedPlan) {
		return errors.New("gate: interrupted commit plan differs from its write-ahead transition")
	}
	return nil
}

func restoreInterruptedParentIndexLocked(repo string, intent gateCommitIntent, indexPath string) error {
	if err := normalizeGateIndexToStates(
		repo, indexPath, fs.FileMode(intent.IndexMode),
		intent.IndexBefore, intent.IndexAfter, intent.IndexRestore,
	); err != nil {
		return err
	}
	if err := replaceGateIndexStatesUnderLock(
		indexPath,
		[][]byte{intent.IndexAfter},
		[][]byte{intent.IndexBefore, intent.IndexRestore},
		intent.IndexRestore,
		fs.FileMode(intent.IndexMode),
	); err != nil {
		return fmt.Errorf("gate: restore interrupted parent index: %w", err)
	}
	return nil
}

// restoreInterruptedState holds Git's actual index lock and the AUTO_MERGE
// pseudoref lock across the complete branch/index/plan/merge-state recovery.
// The admission check is repeated only after both locks are held, and no
// cancellation authority is recorded until the final exact-state check passes.
// The seven manually owned lock paths carry exact intent markers and are
// reclaimed under the process guard after abrupt death. Git's update-ref child
// separately owns branch and HEAD locks: ordinary parent death closes its pipe
// and makes Git abort them, while machine failure can still leave those
// indistinguishable Git locks requiring conventional manual cleanup.
func restoreInterruptedState(repo string, intent gateCommitIntent, observedHead string) error {
	err := withGateGitStateLock(
		repo,
		fs.FileMode(intent.IndexMode),
		intent,
		gateRecoveryBeforeLockHook,
		gateRecoveryLockedHook,
		func(indexPath string) (err error) {
			if err := requireGateIntentKeepalive(repo, intent); err != nil {
				return err
			}
			if err := requireRecoverableGitState(repo, intent, observedHead); err != nil {
				return err
			}
			if !exactGateHead(repo, intent.HeadReference, observedHead) {
				return errors.New("gate: interrupted HEAD changed before prepared state recovery")
			}
			transaction, err := prepareGateReferenceTransaction(
				repo, intent.HeadReference, intent.Parent, observedHead, "overgo gate recovery",
			)
			if err != nil {
				return fmt.Errorf("gate: prepare interrupted state rollback: %w", err)
			}
			defer func() { err = errors.Join(err, transaction.abort()) }()
			if !exactGateHead(repo, intent.HeadReference, observedHead) {
				return errors.New("gate: interrupted HEAD moved while preparing state recovery")
			}
			if err := requireGateIntentKeepalive(repo, intent); err != nil {
				return err
			}
			if gateRecoveryAfterStepHook != nil {
				gateRecoveryAfterStepHook(repo, "admission")
			}
			// RecoverSwap may restore a detached plan pathname and retire its
			// exact phase evidence. Do that only after the prepared Git transaction
			// has locked the captured branch and HEAD referent.
			observedPlan, err := requireRecoverablePlan(repo, intent, observedHead)
			if err != nil {
				return err
			}
			// Publish the parent before changing the index, plan, or merge
			// metadata. Each of those exact write-ahead states is recoverable
			// with HEAD at the parent, while publishing the parent last would
			// leave a crash window where the restored index appears as foreign
			// staged work against the completion commit. Reacquire a verify-only
			// transaction before the first state mutation so a competing branch
			// writer either wins this mutation-free handoff or remains locked out
			// through the rest of recovery.
			if observedHead != intent.Parent {
				if err := transaction.commit(); err != nil {
					return err
				}
				if !exactGateHead(repo, intent.HeadReference, intent.Parent) {
					return errors.New("gate: prepared recovery published inexact parent authority")
				}
				if gateRecoveryAfterStepHook != nil {
					gateRecoveryAfterStepHook(repo, "publication")
				}
				parentGuard, err := prepareGateReferenceTransaction(
					repo, intent.HeadReference, intent.Parent, intent.Parent, "overgo gate recovery guard",
				)
				if err != nil {
					return fmt.Errorf("gate: guard restored parent authority: %w", err)
				}
				transaction = parentGuard
				if !exactGateHead(repo, intent.HeadReference, intent.Parent) {
					return errors.New("gate: restored parent moved while preparing state recovery")
				}
			}
			if err := restoreInterruptedParentIndexLocked(repo, intent, indexPath); err != nil {
				return err
			}
			if gateRecoveryAfterStepHook != nil {
				gateRecoveryAfterStepHook(repo, "parent")
			}
			planPath := filepath.Join(repo, filepath.FromSlash(plan.Path))
			if gatePlanRecoveryBeforeSwapHook != nil {
				gatePlanRecoveryBeforeSwapHook(planPath)
			}
			if err := restoreInterruptedPlan(repo, planPath, observedPlan, intent); err != nil {
				return fmt.Errorf("gate: restore interrupted plan: %w", err)
			}
			if gateRecoveryAfterStepHook != nil {
				gateRecoveryAfterStepHook(repo, "plan")
			}
			if err := restoreInterruptedMergeLocked(repo, intent, indexPath); err != nil {
				return err
			}
			if gateRecoveryAfterStepHook != nil {
				gateRecoveryAfterStepHook(repo, "merge")
			}
			if err := requireExactRecoveredGitStateExceptHead(repo, intent, indexPath); err != nil {
				return err
			}
			if !exactGateHead(repo, intent.HeadReference, intent.Parent) {
				return errors.New("gate: prepared recovery branch moved before state recovery completed")
			}
			if err := transaction.commit(); err != nil {
				return err
			}
			return requireExactRecoveredGitState(repo, intent, indexPath)
		},
	)
	if errors.Is(err, errGateIndexMoved) {
		return errors.New("gate: Git index moved before exact interrupted-state recovery")
	}
	if errors.Is(err, errGateMergeMetadataMoved) {
		return fmt.Errorf("gate: merge metadata moved before exact interrupted-state recovery: %w", err)
	}
	return err
}

func requireExactRecoveredGitState(repo string, intent gateCommitIntent, indexPath string) error {
	if !exactGateHead(repo, intent.HeadReference, intent.Parent) {
		return errors.New("gate: HEAD moved before interrupted-state recovery completed")
	}
	return requireExactRecoveredGitStateExceptHead(repo, intent, indexPath)
}

func requireExactRecoveredGitStateExceptHead(repo string, intent gateCommitIntent, indexPath string) error {
	if err := normalizeGateIndexToStates(
		repo, indexPath, fs.FileMode(intent.IndexMode), intent.IndexBefore, intent.IndexRestore,
	); err != nil {
		return err
	}
	if err := exactGateIndexState(
		indexPath, fs.FileMode(intent.IndexMode), intent.IndexBefore, intent.IndexRestore,
	); err != nil {
		return err
	}
	if matches, err := planBytesMatch(repo, intent.Plan); err != nil {
		return err
	} else if !matches {
		return errors.New("gate: plan moved before interrupted-state recovery completed")
	}
	if err := requireNoGitOperation(repo, "gate: Git operation %s appeared during interrupted-state recovery"); err != nil {
		return err
	}
	files, err := gateMergeMetadataFiles(repo, intent.Merge)
	if err != nil {
		return err
	}
	states, err := readGateMergeMetadata(files)
	if err != nil {
		return err
	}
	present := intent.Merge != nil
	progress, exact := gateMergeMetadataProgress(files, states, present)
	if !exact || progress != len(gateMergeManagedMetadata(files)) {
		return errors.New("gate: merge metadata is not in its exact recovered state")
	}
	return nil
}

func requireRecoverablePlan(repo string, intent gateCommitIntent, head string) ([]byte, error) {
	planPath := filepath.Join(repo, filepath.FromSlash(plan.Path))
	scratch, err := gatePlanScratchPath(repo, intent.Preparation)
	if err != nil {
		return nil, err
	}
	current, err := atomicfile.RecoverSwap(
		planPath, scratch, intent.Plan, intent.AdvancedPlan, fs.FileMode(intent.PlanMode),
	)
	if err != nil {
		return nil, fmt.Errorf("gate: resolve interrupted plan publication: %w", err)
	}
	if !bytes.Equal(intent.Plan, current) && !bytes.Equal(intent.AdvancedPlan, current) {
		return nil, errors.New("gate: plan moved after the interrupted commit; automatic recovery refused")
	}
	if head == intent.Commit {
		status, err := gitAuthorityOutput(
			repo, "--no-optional-locks", "status", "--porcelain=v1", "-z", "--untracked-files=no",
		)
		if err != nil {
			return nil, err
		}
		dirty, err := repoanalysis.ParseDirtyStatus(status)
		if err != nil {
			return nil, err
		}
		prePublicationPlan := bytes.Equal(intent.Plan, current)
		exactPrePublicationDelta := prePublicationPlan && len(dirty) == 1 &&
			dirty[0].Path == plan.Path && dirty[0].OriginalPath == "" &&
			dirty[0].IndexStatus == " " && dirty[0].WorktreeStatus == "M"
		if len(dirty) != 0 && !exactPrePublicationDelta {
			return nil, errors.New("gate: tracked worktree changes appeared after the interrupted commit")
		}
	}
	return current, nil
}

func restoreInterruptedPlan(repo, path string, observed []byte, intent gateCommitIntent) error {
	if bytes.Equal(observed, intent.Plan) {
		return nil
	}
	scratch, err := gatePlanScratchPath(repo, intent.Preparation)
	if err != nil {
		return err
	}
	return atomicfile.CompareAndSwap(
		path, scratch, observed, intent.Plan, fs.FileMode(intent.PlanMode),
	)
}

func requireRecoverableGitState(repo string, intent gateCommitIntent, head string) error {
	indexPath, err := gateIndexPath(repo)
	if err != nil {
		return err
	}
	if err := normalizeGateIndexToStates(
		repo, indexPath, fs.FileMode(intent.IndexMode),
		intent.IndexBefore, intent.IndexAfter, intent.IndexRestore,
	); err != nil {
		return err
	}
	indexBytes, err := gateIndexState(indexPath, fs.FileMode(intent.IndexMode))
	if err != nil {
		return err
	}
	if !bytes.Equal(indexBytes, intent.IndexBefore) && !bytes.Equal(indexBytes, intent.IndexAfter) &&
		!bytes.Equal(indexBytes, intent.IndexRestore) {
		return errors.New("gate: Git index moved beyond the write-ahead states; automatic recovery refused")
	}
	if err := requireNoGitOperation(repo, "gate: Git operation %s appeared after the interrupted commit"); err != nil {
		return err
	}
	files, err := gateMergeMetadataFiles(repo, intent.Merge)
	if err != nil {
		return err
	}
	states, err := readGateMergeMetadata(files)
	if err != nil {
		return err
	}
	if intent.Merge == nil {
		if _, exact := gateMergeMetadataProgress(files, states, false); !exact {
			return errors.New("gate: unrelated merge metadata appeared after the interrupted commit")
		}
		return nil
	}
	if _, exact := gateMergeMetadataProgress(files, states, false); exact {
		return nil
	}
	if _, exact := gateMergeMetadataProgress(files, states, true); exact {
		return nil
	}
	return errors.New("gate: current merge metadata differs from the interrupted merge")
}

func restoreInterruptedMergeLocked(repo string, intent gateCommitIntent, indexPath string) error {
	if intent.Merge == nil {
		return nil
	}
	files, err := gateMergeMetadataFiles(repo, intent.Merge)
	if err != nil {
		return err
	}
	if err := exactGateIndexState(
		indexPath, fs.FileMode(intent.IndexMode), intent.IndexBefore, intent.IndexRestore,
	); err != nil {
		return err
	}
	states, err := readGateMergeMetadata(files)
	if err != nil {
		return err
	}
	if progress, exact := gateMergeMetadataProgress(files, states, true); exact {
		return restoreGateMergeMetadata(repo, files, progress)
	}
	// A crash during committed-state cleanup leaves an exact absent prefix.
	// Finish that cleanup (including removing MERGE_HEAD last), then restore
	// MODE, MSG, and AUTO_MERGE before publishing MERGE_HEAD.
	progress, exact := gateMergeMetadataProgress(files, states, false)
	if !exact {
		return errGateMergeMetadataMoved
	}
	if err := removeGateMergeMetadata(repo, "recovery-cleanup", files, progress); err != nil {
		return err
	}
	var restoreProgress int
	return restoreGateMergeMetadata(repo, files, restoreProgress)
}

func (g *gateContext) oweRecord(batch artifact.Batch, cause error) error {
	debt := gateDebtEnvelope{Version: artifact.InitialDocumentVersion, Preparation: g.preparation.ID, Batch: batch}
	if err := validateGateDebt(debt); err != nil {
		return fmt.Errorf("%w; invalid record debt: %v", cause, err)
	}
	if err := writeJSON(g.repo, gateDebtFile, debt, gatePrivateFileMode); err != nil {
		return fmt.Errorf("%w; persist record debt: %v", cause, err)
	}
	return cause
}

func reconcileGateDebt(repo, storePath string) (artifact.ID, error) {
	var debt gateDebtEnvelope
	if err := readJSON(repo, gateDebtFile, &debt); err != nil {
		return artifact.ID{}, fmt.Errorf("read record debt: %w", err)
	}
	finalization, err := gateDebtFinalization(debt)
	if err != nil {
		return artifact.ID{}, err
	}
	var successIntent *gateCommitIntent
	if finalization.Outcome == runrecord.OutcomeSucceeded {
		intent, found, intentErr := readGateCommitIntentForPreparation(repo, debt.Preparation)
		if intentErr != nil {
			return artifact.ID{}, intentErr
		}
		if !found || intent.Commit == "" {
			return artifact.ID{}, errors.New("gate: successful record debt requires its completed Git intent")
		}
		headReference, headErr := currentHeadReference(repo)
		if headErr != nil {
			return artifact.ID{}, headErr
		}
		head, headErr := command(repo, "git", "rev-parse", "HEAD")
		if headErr != nil {
			return artifact.ID{}, headErr
		}
		if headReference != intent.HeadReference || strings.TrimSpace(head) != intent.Commit {
			return artifact.ID{}, errors.New("gate: successful record debt does not name the current Git commit")
		}
		successIntent = &intent
	}
	store, err := overgodb.OpenContext(context.Background(), filepath.Join(repo, storePath))
	if err != nil {
		return artifact.ID{}, err
	}
	ctx := context.Background()
	if err := requireGateDebtAdmission(ctx, store, debt.Preparation, finalization); err != nil {
		return artifact.ID{}, errors.Join(err, store.Close())
	}
	if successIntent != nil {
		if err := validateInterruptedCommitAuthority(repo, *successIntent, store); err != nil {
			return artifact.ID{}, errors.Join(err, store.Close())
		}
	}
	if _, err := store.Commit(ctx, debt.Batch); err != nil {
		return artifact.ID{}, errors.Join(err, store.Close())
	}
	if err := store.Close(); err != nil {
		return artifact.ID{}, err
	}
	if err := fsatomic.Remove(filepath.Join(repo, filepath.FromSlash(gateDebtFile))); err != nil {
		return artifact.ID{}, err
	}
	if err := finalizeMatchingHeartbeat(repo, debt.Preparation); err != nil {
		return artifact.ID{}, err
	}
	if successIntent != nil {
		if _, err := recoverInterruptedCommit(repo, storePath); err != nil {
			return debt.Preparation, err
		}
		head, err := command(repo, "git", "rev-parse", "HEAD")
		if err != nil {
			return debt.Preparation, err
		}
		switch strings.TrimSpace(head) {
		case successIntent.Commit:
			return debt.Preparation, nil
		case successIntent.Parent:
			return debt.Preparation, errors.New("gate: reconciled success lacked exact completion authority; commit was rolled back")
		default:
			return debt.Preparation, errors.New("gate: debt recovery left HEAD at an unknown commit")
		}
	}
	if err := removeGateCommitIntentForPreparation(repo, debt.Preparation); err != nil {
		return artifact.ID{}, err
	}
	return debt.Preparation, nil
}

type gateLifecycleRecoveryCensus struct {
	Head               artifact.CommitID
	Outstanding        runrecord.GateLifecycle
	OutstandingCount   int
	OutstandingDebt    []runrecord.GateLifecycle
	NewestFinalization runrecord.GateLifecycle
	HasFinalization    bool
}

func inspectGateLifecycleRecoveryCensus(
	ctx context.Context,
	store *overgodb.Store,
) (gateLifecycleRecoveryCensus, error) {
	if ctx == nil || store == nil {
		return gateLifecycleRecoveryCensus{}, errors.New("gate: lifecycle recovery requires the canonical store")
	}
	census := gateLifecycleRecoveryCensus{}
	census.Head, _ = store.Head()
	lifecycles, err := runrecord.GateLifecyclesInStore(ctx, store)
	if err != nil {
		return gateLifecycleRecoveryCensus{}, err
	}
	preparations := make(map[artifact.ID]runrecord.GateLifecycle)
	var finalizations []runrecord.GateLifecycle
	for _, lifecycle := range lifecycles {
		if err := lifecycle.ValidateIdentity(); err != nil {
			return gateLifecycleRecoveryCensus{}, fmt.Errorf(
				"gate: validate lifecycle %s during recovery: %w", lifecycle.ID, err,
			)
		}
		if lifecycle.State == runrecord.GatePrepared {
			preparations[lifecycle.ID] = lifecycle
			continue
		}
		finalizations = append(finalizations, lifecycle)
	}
	validatedFinalizations := make(map[artifact.ID]artifact.ID)
	for _, finalization := range finalizations {
		preparation, found := preparations[*finalization.Preparation]
		if !found {
			return gateLifecycleRecoveryCensus{}, fmt.Errorf(
				"gate: finalization %s lacks its exact preparation during recovery", finalization.ID,
			)
		}
		canonical, found, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
		if err != nil {
			return gateLifecycleRecoveryCensus{}, fmt.Errorf(
				"gate: validate lifecycle %s during recovery: %w", preparation.ID, err,
			)
		}
		if !found || canonical.ID != finalization.ID {
			return gateLifecycleRecoveryCensus{}, fmt.Errorf(
				"gate: finalization %s is not the sole canonical finalization for preparation %s",
				finalization.ID, preparation.ID,
			)
		}
		if err := validateCompleteGateFinalization(ctx, store, preparation, finalization); err != nil {
			return gateLifecycleRecoveryCensus{}, fmt.Errorf(
				"gate: validate lifecycle %s during recovery: %w", preparation.ID, err,
			)
		}
		validatedFinalizations[preparation.ID] = finalization.ID
		// GateLifecyclesInStore owns the deterministic durable order, including
		// its artifact-ID tie-break for documents co-introduced by legacy import.
		census.NewestFinalization = finalization
		census.HasFinalization = true
	}
	var outstanding []runrecord.GateLifecycle
	for _, preparation := range lifecycles {
		if preparation.State != runrecord.GatePrepared {
			continue
		}
		finalization, finalized, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
		if err != nil {
			return gateLifecycleRecoveryCensus{}, err
		}
		if !finalized {
			outstanding = append(outstanding, preparation)
			continue
		}
		if validated, found := validatedFinalizations[preparation.ID]; !found || validated != finalization.ID {
			return gateLifecycleRecoveryCensus{}, fmt.Errorf(
				"gate: preparation %s lacks one validated canonical finalization during recovery",
				preparation.ID,
			)
		}
	}
	census.OutstandingCount = len(outstanding)
	census.OutstandingDebt = outstanding
	if len(outstanding) != 1 {
		return census, nil
	}
	if _, err := runrecord.RequireEnvironment(ctx, store, outstanding[0].Environment); err != nil {
		return gateLifecycleRecoveryCensus{}, fmt.Errorf(
			"gate: load outstanding lifecycle environment during recovery: %w", err,
		)
	}
	census.Outstanding = outstanding[0]
	return census, nil
}

func requireGateLifecycleRecoveryCensus(
	ctx context.Context,
	store *overgodb.Store,
) (gateLifecycleRecoveryCensus, error) {
	census, err := inspectGateLifecycleRecoveryCensus(ctx, store)
	if err != nil {
		return gateLifecycleRecoveryCensus{}, err
	}
	if census.OutstandingCount != 1 {
		return gateLifecycleRecoveryCensus{}, fmt.Errorf(
			"gate: lifecycle recovery found %d unresolved preparations, want 1",
			census.OutstandingCount,
		)
	}
	return census, nil
}

// requireSelectedGateLifecycleRecovery resolves ambiguous recovery debt by an
// exact operator-named preparation instead of the sole-debt cardinality rule.
// The selection is argument-bound: an identity that is not currently
// outstanding, complete recovery debt refuses rather than degrading to any
// nearest candidate.
func requireSelectedGateLifecycleRecovery(
	ctx context.Context,
	store *overgodb.Store,
	selected string,
) (gateLifecycleRecoveryCensus, error) {
	id, err := artifact.ParseID(selected)
	if err != nil || id.Kind() != artifact.KindEvidence {
		return gateLifecycleRecoveryCensus{}, fmt.Errorf(
			"gate: -preparation %q is not an exact evidence artifact ID", selected,
		)
	}
	census, err := inspectGateLifecycleRecoveryCensus(ctx, store)
	if err != nil {
		return gateLifecycleRecoveryCensus{}, err
	}
	for _, preparation := range census.OutstandingDebt {
		if preparation.ID != id {
			continue
		}
		if _, err := runrecord.RequireEnvironment(ctx, store, preparation.Environment); err != nil {
			return gateLifecycleRecoveryCensus{}, fmt.Errorf(
				"gate: load selected lifecycle environment during recovery: %w", err,
			)
		}
		census.Outstanding = preparation
		return census, nil
	}
	return gateLifecycleRecoveryCensus{}, fmt.Errorf(
		"gate: preparation %s is not outstanding recovery debt", selected,
	)
}

func validateLegacyFinalizedHeartbeat(
	ctx context.Context,
	store *overgodb.Store,
	heartbeat runrecord.GateHeartbeat,
) error {
	preparation, err := runrecord.RequireGateLifecycle(ctx, store, heartbeat.Preparation)
	if err != nil {
		return fmt.Errorf("gate: load finalized lifecycle locator preparation: %w", err)
	}
	if preparation.State != runrecord.GatePrepared || preparation.TreeKey != heartbeat.TreeKey ||
		preparation.Environment != heartbeat.Environment {
		return errors.New("gate: finalized lifecycle locator contradicts its exact preparation")
	}
	finalization, found, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
	if err != nil {
		return err
	}
	if !found {
		return errors.New("gate: finalized lifecycle locator preparation lacks a finalization")
	}
	if err := validateHeartbeatFinalization(ctx, store, preparation, finalization); err != nil {
		return err
	}
	return nil
}

func writeFinalizedHeartbeatForAuthority(
	repo string,
	store *overgodb.Store,
	finalization runrecord.GateLifecycle,
) error {
	ctx := context.Background()
	if err := requireCompleteGateFinalization(ctx, store, finalization); err != nil {
		return fmt.Errorf("gate: validate bootstrapped lifecycle authority: %w", err)
	}
	preparation, err := runrecord.RequireGateLifecycle(ctx, store, *finalization.Preparation)
	if err != nil {
		return err
	}
	heartbeat := runrecord.GateHeartbeat{
		Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatFinalized,
		Preparation: preparation.ID, TreeKey: preparation.TreeKey, Environment: preparation.Environment,
		PID: os.Getpid(), Updated: time.Now().UTC(),
	}
	return writeJSON(repo, gateHeartbeatFile, heartbeat, clioptions.OutputFileMode)
}

func repairCompletedLifecycleHeartbeat(
	repo string,
	ctx context.Context,
	store *overgodb.Store,
	current runrecord.GateLifecycle,
	heartbeat runrecord.GateHeartbeat,
) (artifact.ID, bool, error) {
	if current.State != runrecord.GateFinalized || current.Preparation == nil ||
		*current.Preparation == heartbeat.Preparation {
		return artifact.ID{}, false, nil
	}
	census, err := inspectGateLifecycleRecoveryCensus(ctx, store)
	if err != nil {
		return artifact.ID{}, false, err
	}
	if census.OutstandingCount != 0 {
		return artifact.ID{}, false, nil
	}
	if err := requireCompleteGateFinalization(ctx, store, current); err != nil {
		return artifact.ID{}, false, fmt.Errorf("gate: validate completed retry authority: %w", err)
	}
	preparation, err := runrecord.RequireGateLifecycle(ctx, store, heartbeat.Preparation)
	if err != nil {
		return artifact.ID{}, false, fmt.Errorf("gate: load completed retry locator preparation: %w", err)
	}
	if preparation.State != runrecord.GatePrepared || preparation.TreeKey != heartbeat.TreeKey ||
		preparation.Environment != heartbeat.Environment {
		return artifact.ID{}, false, errors.New("gate: completed retry locator contradicts its exact preparation")
	}
	finalization, found, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
	if err != nil {
		return artifact.ID{}, false, err
	}
	if !found {
		return artifact.ID{}, false, errors.New("gate: completed retry locator preparation lacks a finalization")
	}
	if err := validateHeartbeatFinalization(ctx, store, preparation, finalization); err != nil {
		return artifact.ID{}, false, err
	}
	currentID, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		return artifact.ID{}, false, err
	}
	if !found || currentID != current.ID {
		return artifact.ID{}, false, errors.New("gate: completed retry lifecycle authority moved")
	}
	if err := writeFinalizedHeartbeatForAuthority(repo, store, current); err != nil {
		return artifact.ID{}, false, err
	}
	return preparation.ID, true, nil
}

var gateRecordFailureBeforeStoreCommitHook func(*overgodb.Store)

var gateRecordFailureAfterStoreCommitHook func(*overgodb.Store) error

// recordSelectedUnbatchableFailure additionally accepts an exact preparation
// selector so an operator can close one named stale debt when the census finds
// more than one. The selector is honored only behind a complete finalized
// current authority whose locator matches it; every other recovery shape keeps
// the census cardinality rule.
func recordSelectedUnbatchableFailure(repo, storePath, selected string) (artifact.ID, error) {
	recordStarted := processmeasure.NewStopwatch()
	if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(gateCommitIntentFile))); err == nil {
		return artifact.ID{}, errors.New("gate: interrupted commit intent exists; run `go run ./cmd/gate -recover-interrupted` instead of recording failure")
	} else if !errors.Is(err, os.ErrNotExist) {
		return artifact.ID{}, err
	}
	store, err := overgodb.OpenContext(context.Background(), filepath.Join(repo, storePath))
	if err != nil {
		return artifact.ID{}, err
	}
	defer store.Close()
	ctx := context.Background()
	currentID, aliasFound, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		return artifact.ID{}, err
	}
	var current runrecord.GateLifecycle
	if aliasFound {
		current, err = runrecord.RequireGateLifecycle(ctx, store, currentID)
		if err != nil {
			return artifact.ID{}, err
		}
	}
	var heartbeat runrecord.GateHeartbeat
	heartbeatFound := true
	if err := readJSON(repo, gateHeartbeatFile, &heartbeat); errors.Is(err, os.ErrNotExist) {
		heartbeatFound = false
	} else if err != nil {
		return artifact.ID{}, err
	} else if err := heartbeat.Validate(); err != nil {
		return artifact.ID{}, errors.New("gate: no valid unresolved lifecycle locator to finalize")
	}
	if selected != "" && (!heartbeatFound || heartbeat.State != runrecord.HeartbeatFinalized || !aliasFound) {
		return artifact.ID{}, errors.New(
			"gate: -preparation recovery requires a finalized lifecycle locator and current terminal authority",
		)
	}
	preserveCurrentAuthority := false
	preserveHeartbeat := false
	legacyBootstrap := false
	censusBoundRecovery := false
	var recoveryCensus gateLifecycleRecoveryCensus
	if !heartbeatFound {
		var preparation runrecord.GateLifecycle
		if aliasFound && current.State == runrecord.GatePrepared {
			preparation = current
		} else {
			if aliasFound && current.State != runrecord.GateFinalized {
				return artifact.ID{}, errors.New("gate: current lifecycle authority cannot identify recovery debt")
			}
			if aliasFound {
				if err := requireCompleteGateFinalization(ctx, store, current); err != nil {
					return artifact.ID{}, fmt.Errorf("gate: validate current lifecycle authority: %w", err)
				}
				recoveryCensus, err = requireGateLifecycleRecoveryCensus(ctx, store)
				if err != nil {
					return artifact.ID{}, err
				}
				preparation = recoveryCensus.Outstanding
				censusBoundRecovery = true
				preserveCurrentAuthority = true
				preserveHeartbeat = true
			} else {
				recoveryCensus, err = requireGateLifecycleRecoveryCensus(ctx, store)
				if err != nil {
					return artifact.ID{}, err
				}
				preparation = recoveryCensus.Outstanding
				censusBoundRecovery = true
				legacyBootstrap = true
			}
		}
		heartbeat = runrecord.GateHeartbeat{
			Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatRunning,
			Preparation: preparation.ID, TreeKey: preparation.TreeKey, Environment: preparation.Environment,
			PID: os.Getpid(), Updated: time.Now().UTC(),
		}
	} else if heartbeat.State == runrecord.HeartbeatFinalized && !aliasFound {
		recoveryCensus, err = requireGateLifecycleRecoveryCensus(ctx, store)
		if err != nil {
			return artifact.ID{}, err
		}
		if err := validateLegacyFinalizedHeartbeat(ctx, store, heartbeat); err != nil {
			return artifact.ID{}, err
		}
		preparation := recoveryCensus.Outstanding
		heartbeat = runrecord.GateHeartbeat{
			Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatRunning,
			Preparation: preparation.ID, TreeKey: preparation.TreeKey, Environment: preparation.Environment,
			PID: os.Getpid(), Updated: time.Now().UTC(),
		}
		censusBoundRecovery = true
		legacyBootstrap = true
		preserveHeartbeat = true
	} else if heartbeat.State == runrecord.HeartbeatFinalized {
		if current.State != runrecord.GateFinalized || current.Preparation == nil {
			return artifact.ID{}, errors.New("gate: finalized lifecycle locator lacks current terminal authority")
		}
		currentPreparation, err := runrecord.RequireGateLifecycle(ctx, store, *current.Preparation)
		if err != nil {
			return artifact.ID{}, err
		}
		if heartbeat.Preparation != currentPreparation.ID || heartbeat.TreeKey != currentPreparation.TreeKey ||
			heartbeat.Environment != currentPreparation.Environment {
			if selected != "" {
				return artifact.ID{}, errors.New(
					"gate: -preparation recovery requires the lifecycle locator to match current terminal authority",
				)
			}
			preparation, repaired, repairErr := repairCompletedLifecycleHeartbeat(
				repo, ctx, store, current, heartbeat,
			)
			if repairErr != nil {
				return artifact.ID{}, repairErr
			}
			if repaired {
				return preparation, nil
			}
			return artifact.ID{}, errors.New("gate: finalized lifecycle locator differs from current terminal authority")
		}
		if err := requireCompleteGateFinalization(ctx, store, current); err != nil {
			return artifact.ID{}, fmt.Errorf("gate: validate current lifecycle authority: %w", err)
		}
		if selected != "" {
			recoveryCensus, err = requireSelectedGateLifecycleRecovery(ctx, store, selected)
		} else {
			recoveryCensus, err = requireGateLifecycleRecoveryCensus(ctx, store)
		}
		if err != nil {
			return artifact.ID{}, err
		}
		preparation := recoveryCensus.Outstanding
		heartbeat = runrecord.GateHeartbeat{
			Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatRunning,
			Preparation: preparation.ID, TreeKey: preparation.TreeKey, Environment: preparation.Environment,
			PID: os.Getpid(), Updated: time.Now().UTC(),
		}
		censusBoundRecovery = true
		preserveCurrentAuthority = true
		preserveHeartbeat = true
	} else if heartbeat.State != runrecord.HeartbeatRunning && heartbeat.State != runrecord.HeartbeatRecordDebt {
		return artifact.ID{}, errors.New("gate: no valid unresolved lifecycle locator to finalize")
	}
	content, ok, err := artifact.ReadContent(context.Background(), store, heartbeat.Preparation)
	if err != nil {
		return artifact.ID{}, fmt.Errorf("gate: prepared lifecycle unavailable: %w", err)
	}
	if !ok && !aliasFound && heartbeatFound &&
		(heartbeat.State == runrecord.HeartbeatRunning || heartbeat.State == runrecord.HeartbeatRecordDebt) {
		recoveryCensus, err = inspectGateLifecycleRecoveryCensus(ctx, store)
		if err != nil {
			return artifact.ID{}, err
		}
		switch recoveryCensus.OutstandingCount {
		case 0:
		case 1:
			preparation := recoveryCensus.Outstanding
			heartbeat.Preparation = preparation.ID
			heartbeat.TreeKey = preparation.TreeKey
			heartbeat.Environment = preparation.Environment
			censusBoundRecovery = true
			legacyBootstrap = true
			content, ok, err = artifact.ReadContent(ctx, store, preparation.ID)
			if err != nil {
				return artifact.ID{}, fmt.Errorf("gate: prepared lifecycle unavailable: %w", err)
			}
		default:
			return artifact.ID{}, fmt.Errorf(
				"gate: lifecycle recovery found %d unresolved preparations, want 1",
				recoveryCensus.OutstandingCount,
			)
		}
	}
	if !ok {
		priorFinalizedAuthority := aliasFound && current.State == runrecord.GateFinalized &&
			current.Preparation != nil && *current.Preparation != heartbeat.Preparation
		if !heartbeatFound || heartbeat.State != runrecord.HeartbeatRunning || aliasFound && !priorFinalizedAuthority {
			return artifact.ID{}, errors.New("gate: prepared lifecycle unavailable")
		}
		if err := fsatomic.Remove(filepath.Join(repo, filepath.FromSlash(gateHeartbeatFile))); err != nil {
			return artifact.ID{}, fmt.Errorf("gate: remove uncommitted lifecycle locator: %w", err)
		}
		return heartbeat.Preparation, nil
	}
	if aliasFound && !preserveCurrentAuthority {
		switch current.State {
		case runrecord.GatePrepared:
			if current.ID != heartbeat.Preparation {
				return artifact.ID{}, errors.New("gate: lifecycle locator differs from current store authority")
			}
		case runrecord.GateFinalized:
			if current.Preparation == nil || *current.Preparation != heartbeat.Preparation {
				preparation, repaired, repairErr := repairCompletedLifecycleHeartbeat(
					repo, ctx, store, current, heartbeat,
				)
				if repairErr != nil {
					return artifact.ID{}, repairErr
				}
				if repaired {
					return preparation, nil
				}
				return artifact.ID{}, errors.New("gate: finalized lifecycle authority differs from its locator")
			}
		}
	}
	preparation, err := runrecord.ParseGateLifecycle(content.Data)
	if err != nil || preparation.State != runrecord.GatePrepared || preparation.Environment != heartbeat.Environment {
		return artifact.ID{}, errors.New("gate: lifecycle locator contradicts its preparation")
	}
	if !aliasFound && !legacyBootstrap {
		recoveryCensus, err = requireGateLifecycleRecoveryCensus(ctx, store)
		if err != nil {
			return artifact.ID{}, err
		}
		if recoveryCensus.Outstanding.ID != preparation.ID ||
			recoveryCensus.Outstanding.TreeKey != preparation.TreeKey ||
			recoveryCensus.Outstanding.Environment != preparation.Environment {
			return artifact.ID{}, errors.New("gate: lifecycle locator differs from the sole legacy recovery debt")
		}
		censusBoundRecovery = true
		legacyBootstrap = true
	}
	finalization, finalized, err := runrecord.GateFinalizationForPreparation(
		context.Background(), store, preparation.ID,
	)
	if err != nil {
		return artifact.ID{}, err
	}
	if finalized {
		if preserveCurrentAuthority || legacyBootstrap {
			return artifact.ID{}, errors.New("gate: recovered stale preparation unexpectedly has a finalization")
		}
		if aliasFound && current.State == runrecord.GateFinalized && current.ID != finalization.ID {
			return artifact.ID{}, errors.New("gate: current lifecycle authority names another finalization")
		}
		if err := validateHeartbeatFinalization(context.Background(), store, preparation, finalization); err != nil {
			return artifact.ID{}, err
		}
		if err := bindRecoveredGateFinalization(
			context.Background(), store, preparation, finalization, currentID, aliasFound,
		); err != nil {
			return artifact.ID{}, err
		}
		return preparation.ID, finalizeHeartbeat(repo, heartbeat)
	}
	if !censusBoundRecovery {
		recoveryCensus, err = requireGateLifecycleRecoveryCensus(ctx, store)
		if err != nil {
			return artifact.ID{}, err
		}
		if recoveryCensus.Outstanding.ID != preparation.ID ||
			recoveryCensus.Outstanding.TreeKey != preparation.TreeKey ||
			recoveryCensus.Outstanding.Environment != preparation.Environment {
			return artifact.ID{}, errors.New("gate: lifecycle locator differs from the sole recovery debt")
		}
		censusBoundRecovery = true
		legacyBootstrap = !aliasFound
	}
	codeCommit, err := command(repo, "git", "rev-parse", "HEAD")
	if err != nil {
		return artifact.ID{}, err
	}
	codeCommit = strings.TrimSpace(codeCommit)
	recipeID, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte(gateRecipeSeed))
	if err != nil {
		return artifact.ID{}, err
	}
	outcome, failure, stepName, stepOutcome := runrecord.OutcomeFailed, "record", "record", runrecord.StepFailed
	if heartbeat.State == runrecord.HeartbeatRunning {
		outcome, failure, stepName, stepOutcome = runrecord.OutcomeCancelled, "", "recovery", runrecord.StepCancelled
	}
	durationNS, err := completedGateMeasurement(recordStarted, "record recovery")
	if err != nil {
		return artifact.ID{}, err
	}
	record, err := runrecord.NewGateRecord(
		recipeID, preparation.Environment, codeCommit, outcome, failure, durationNS,
		[]runrecord.GateStep{{
			Name: stepName, Phase: runrecord.PhaseValidate, Outcome: stepOutcome, DurationNS: durationNS,
		}},
	)
	if err != nil {
		return artifact.ID{}, err
	}
	batch, err := record.Batch("gate/final/" + preparation.ID.String())
	if err != nil {
		return artifact.ID{}, err
	}
	newFinalization, err := runrecord.NewGateFinalization(preparation, codeCommit, record.Result.ID, outcome)
	if err != nil {
		return artifact.ID{}, err
	}
	finalizedContent, err := newFinalization.Content()
	if err != nil {
		return artifact.ID{}, err
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: recipeID})
	batch.Contents = append(batch.Contents, finalizedContent)
	batch.Lineage = append(batch.Lineage, newFinalization.Lineage()...)
	bootstrapFinalization := newFinalization
	if censusBoundRecovery {
		batch.ExpectedHead = &recoveryCensus.Head
	}
	if legacyBootstrap {
		if recoveryCensus.HasFinalization {
			bootstrapFinalization = recoveryCensus.NewestFinalization
		}
		batch.Aliases = append(batch.Aliases, artifact.AliasBinding{
			Name: runrecord.GateLifecycleCurrentAlias, Target: bootstrapFinalization.ID,
		})
	} else if !preserveCurrentAuthority {
		if aliasFound {
			appendGateFinalizationAlias(&batch, preparation.ID, newFinalization.ID)
		} else {
			batch.Aliases = append(batch.Aliases, artifact.AliasBinding{
				Name: runrecord.GateLifecycleCurrentAlias, Target: newFinalization.ID,
			})
		}
	}
	if gateRecordFailureBeforeStoreCommitHook != nil {
		gateRecordFailureBeforeStoreCommitHook(store)
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return artifact.ID{}, err
	}
	if gateRecordFailureAfterStoreCommitHook != nil {
		if err := gateRecordFailureAfterStoreCommitHook(store); err != nil {
			return preparation.ID, err
		}
	}
	if legacyBootstrap {
		return preparation.ID, writeFinalizedHeartbeatForAuthority(repo, store, bootstrapFinalization)
	}
	if preserveHeartbeat {
		return preparation.ID, nil
	}
	if err := finalizeHeartbeat(repo, heartbeat); err != nil {
		return artifact.ID{}, err
	}
	return preparation.ID, nil
}

func bindRecoveredGateFinalization(
	ctx context.Context,
	store *overgodb.Store,
	preparation, finalization runrecord.GateLifecycle,
	current artifact.ID,
	aliasFound bool,
) error {
	if ctx == nil || store == nil || preparation.State != runrecord.GatePrepared ||
		finalization.State != runrecord.GateFinalized || finalization.Preparation == nil ||
		*finalization.Preparation != preparation.ID {
		return errors.New("gate: recovered lifecycle alias authority is invalid")
	}
	if aliasFound && current == finalization.ID {
		return nil
	}
	binding := artifact.AliasBinding{
		Name: runrecord.GateLifecycleCurrentAlias, Target: finalization.ID,
	}
	if aliasFound {
		if current != preparation.ID {
			return errors.New("gate: recovered lifecycle alias moved beyond its preparation")
		}
		previous := current
		binding.Previous = &previous
	}
	_, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key:     "gate/recovered-alias/" + preparation.ID.String(),
		Aliases: []artifact.AliasBinding{binding},
	})
	if err != nil {
		return fmt.Errorf("gate: bind recovered lifecycle finalization: %w", err)
	}
	return nil
}

func validateHeartbeatFinalization(
	ctx context.Context,
	store *overgodb.Store,
	preparation, finalization runrecord.GateLifecycle,
) error {
	if finalization.Preparation == nil || *finalization.Preparation != preparation.ID {
		return errors.New("gate: lifecycle locator has a contradictory finalization")
	}
	if err := requireCompleteGateFinalization(ctx, store, finalization); err != nil {
		return fmt.Errorf("gate: lifecycle locator finalization: %w", err)
	}
	return nil
}

func finalizeHeartbeat(repo string, heartbeat runrecord.GateHeartbeat) error {
	heartbeat.State, heartbeat.PID, heartbeat.Updated = runrecord.HeartbeatFinalized, os.Getpid(), time.Now().UTC()
	return writeJSON(repo, gateHeartbeatFile, heartbeat, clioptions.OutputFileMode)
}

func finalizeMatchingHeartbeat(repo string, preparation artifact.ID) error {
	var heartbeat runrecord.GateHeartbeat
	if err := readJSON(repo, gateHeartbeatFile, &heartbeat); errors.Is(err, os.ErrNotExist) {
		return nil
	} else if err != nil {
		return err
	}
	if err := heartbeat.Validate(); err != nil {
		return fmt.Errorf("gate: invalid lifecycle locator after recovery: %w", err)
	}
	if heartbeat.Preparation != preparation {
		return errors.New("gate: lifecycle locator differs from the recovered preparation")
	}
	if heartbeat.State == runrecord.HeartbeatFinalized {
		return nil
	}
	return finalizeHeartbeat(repo, heartbeat)
}

func validateGateDebt(debt gateDebtEnvelope) error {
	_, err := gateDebtFinalization(debt)
	return err
}

func gateDebtFinalization(debt gateDebtEnvelope) (runrecord.GateLifecycle, error) {
	if debt.Version != artifact.InitialDocumentVersion || debt.Preparation.Kind() != artifact.KindEvidence {
		return runrecord.GateLifecycle{}, errors.New("gate: invalid record debt envelope")
	}
	if err := debt.Batch.Validate(); err != nil {
		return runrecord.GateLifecycle{}, err
	}
	if debt.Batch.Key != "gate/final/"+debt.Preparation.String() {
		return runrecord.GateLifecycle{}, errors.New("gate: record debt is not bound to its preparation")
	}
	var matching []runrecord.GateLifecycle
	for _, content := range debt.Batch.Contents {
		if content.Descriptor.MediaType == automationcheck.ManifestAnalysisMediaType &&
			content.Descriptor.Schema == automationcheck.ManifestAnalysisSchema {
			if _, err := automationcheck.ParseManifestAnalysis(content.Data); err != nil {
				return runrecord.GateLifecycle{}, err
			}
		}
		if content.Descriptor.MediaType != runrecord.GateLifecycleMediaType || content.Descriptor.Schema != runrecord.GateLifecycleSchema {
			continue
		}
		lifecycle, err := runrecord.ParseGateLifecycle(content.Data)
		if err != nil {
			return runrecord.GateLifecycle{}, err
		}
		if lifecycle.State == runrecord.GateFinalized && lifecycle.Preparation != nil && *lifecycle.Preparation == debt.Preparation {
			matching = append(matching, lifecycle)
		}
	}
	if len(matching) != 1 {
		return runrecord.GateLifecycle{}, fmt.Errorf("gate: record debt has %d matching finalizations, want 1", len(matching))
	}
	return matching[0], nil
}

type watchdogStatus struct {
	Version   uint16                       `json:"version"`
	State     runrecord.GateHeartbeatState `json:"state"`
	Heartbeat *runrecord.GateHeartbeat     `json:"heartbeat,omitempty"`
}

func printGateWatchdog(repo string, staleAfter time.Duration) error {
	var heartbeat runrecord.GateHeartbeat
	err := readJSON(repo, gateHeartbeatFile, &heartbeat)
	if errors.Is(err, os.ErrNotExist) {
		return printJSON(watchdogStatus{Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatAbsent})
	}
	if err != nil {
		return err
	}
	if err := heartbeat.Validate(); err != nil {
		return err
	}
	return printJSON(watchdogStatus{
		Version: heartbeat.Version, State: heartbeat.Watchdog(time.Now().UTC(), staleAfter), Heartbeat: &heartbeat,
	})
}
