package gate

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
	"overgo/internal/codeprofile"
	"overgo/internal/fsatomic"
	"overgo/internal/loop"
	"overgo/internal/plan"
	"overgo/internal/processmeasure"
	"overgo/internal/runrecord"
)

func (g *gateContext) packagePassObserver(ctx context.Context, ledger *packageEvidenceLedger, mode string, inputs map[string]artifact.ID) func(string, bool) error {
	return func(packagePath string, passed bool) error {
		input, found := inputs[packagePath]
		if !found || !input.Valid() || g.retryCache == nil {
			return errors.New("package evidence: terminal event lacks declared inputs")
		}
		// Flush observed terminal facts even when cancellation stops later work.
		if err := ledger.record(context.WithoutCancel(ctx), packagePath, passed); err != nil {
			return err
		}
		if passed {
			if err := g.retryCache.RecordPackagePass(packagePath, mode, input); err != nil {
				return err
			}
		} else {
			invocation, err := automationcheck.PackageInvocation(packagePath, mode)
			if err != nil {
				return err
			}
			delete(g.retryCache.Entries, invocation.String())
		}
		return g.saveRetryCache(*g.retryCache)
	}
}

func completedGateMeasurement(started processmeasure.Stopwatch, operation string) (uint64, error) {
	duration, err := started.Elapsed()
	if err != nil {
		return 0, fmt.Errorf("gate: %s duration: %w", operation, err)
	}
	if duration == 0 {
		return 0, fmt.Errorf("gate: %s duration is unavailable", operation)
	}
	return duration, nil
}

func (g *gateContext) manifestAnalysisContent() (artifact.Content, error) {
	if g.manifestPlan == nil || g.manifestDelta == nil || g.manifestImpact == nil {
		return artifact.Content{}, nil
	}
	analysis, err := automationcheck.NewManifestAnalysis(
		*g.manifestDelta, *g.manifestImpact, *g.manifestPlan, g.selection, g.manifestMetrics,
	)
	if err != nil {
		return artifact.Content{}, err
	}
	return analysis.Content()
}

func (g *gateContext) record(outcome runrecord.Outcome, failure string) error {
	// A checkpoint publication records at the plan head it verified: it
	// commits nothing and completes no step.
	codeCommit := g.planHead
	if outcome == runrecord.OutcomeSucceeded {
		codeCommit = g.committedHead
	}
	if !validGitObjectID(codeCommit) {
		return errors.New("gate: exact code commit is unavailable for the final record")
	}

	recipeID, err := g.gateRecipeID()
	if err != nil {
		return err
	}
	wall, err := g.clock.Elapsed()
	if err != nil {
		return err
	}
	record, err := runrecord.NewGateRecord(recipeID, g.environment.ID, codeCommit, outcome, failure, wall, g.steps)
	if err != nil {
		return err
	}
	batch, err := record.Batch("gate/final/" + g.preparation.ID.String())
	if err != nil {
		return err
	}
	if err := g.appendSuiteCost(&batch, record.Result.ID); err != nil {
		return err
	}
	if err := g.appendSelectionCauses(&batch, record.Result.ID); err != nil {
		return err
	}
	environmentContent, err := g.environment.Content()
	if err != nil {
		return err
	}
	finalized, err := runrecord.NewGateFinalization(g.preparation, codeCommit, record.Result.ID, outcome)
	if err != nil {
		return err
	}
	finalizedContent, err := finalized.Content()
	if err != nil {
		return err
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: recipeID})
	batch.Contents = append(batch.Contents, environmentContent, finalizedContent)
	if outcome == runrecord.OutcomeSucceeded {
		if err := g.appendLaneObligation(&batch, codeCommit, record.Result.ID); err != nil {
			return err
		}
	}
	if outcome == runrecord.OutcomeSucceeded && g.dispatchClaim != nil {
		release, err := g.dispatchClaim.ReleaseBatch(g.dispatchClaim.Worker, "completed")
		if err != nil {
			return err
		}
		batch.Aliases = append(batch.Aliases, release.Aliases...)
	}

	batch.Lineage = append(batch.Lineage, finalized.Lineage()...)
	if outcome == runrecord.OutcomeSucceeded {
		switch g.planProjection {
		case plan.MergeProjectionFirstParentTarget:
			if g.mergeAuthority == nil {
				return errors.New("gate: successful first-parent-target merge lacks authority receipt")
			}
			receiptContent, err := g.mergeAuthority.Content()
			if err != nil {
				return err
			}
			batch.Contents = append(batch.Contents, receiptContent)
			batch.Lineage = append(batch.Lineage, g.mergeAuthority.Lineage()...)
			batch.Lineage = append(batch.Lineage, artifact.Lineage{
				Child: record.Result.ID, Parent: g.mergeAuthority.ID, Relation: artifact.RelationDependsOn,
			})
		case plan.MergeProjectionSemanticUnion:
			if g.mergeAuthority != nil {
				return errors.New("gate: semantic-union completion carries projected merge authority")
			}
		default:
			return errors.New("gate: successful completion has invalid plan projection")
		}
	}
	appendGateFinalizationAlias(&batch, g.preparation.ID, finalized.ID)
	for _, manifest := range []*codemanifest.Manifest{g.baseManifest, g.candidateManifest} {
		if manifest == nil {
			continue
		}
		// Register the manifest identity in this batch so the record's
		// lineage stays resolvable even when the batch lands as owed
		// debt before the digest publication ran.
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: manifest.ID})
		batch.Lineage = append(batch.Lineage, artifact.Lineage{
			Child: record.Result.ID, Parent: manifest.ID, Relation: artifact.RelationDependsOn,
		})
	}
	if analysis, analysisErr := g.manifestAnalysisContent(); analysisErr != nil {
		return g.oweRecord(batch, analysisErr)
	} else if analysis.Descriptor.ID.Valid() {
		batch.Contents = append(batch.Contents, analysis)
		batch.Lineage = append(batch.Lineage, artifact.Lineage{
			Child: record.Result.ID, Parent: analysis.Descriptor.ID, Relation: artifact.RelationDependsOn,
		})
	}
	if outcome == runrecord.OutcomeSucceeded {
		if err := g.appendProfileEvidence(&batch, codeCommit, record.Result.ID); err != nil {
			return err
		}
	}
	if err := g.appendAttemptRecord(&batch, codeCommit, recipeID, record.Result.ID, outcome, failure); err != nil {
		return err
	}
	// A wall-time evaluation rides every successful run: evaluations are the
	// advisory layer's observation unit, so the gate's own history becomes
	// the calibration corpus (first run calibrates, second enforces).
	if outcome == runrecord.OutcomeSucceeded {
		workloadID, err := artifact.IdentifyBytes(artifact.KindDataset, []byte(gateWorkloadSeed))
		if err != nil {
			return err
		}
		evaluation, err := runrecord.NewEvaluation(recipeID, record.Run.ID, workloadID, []runrecord.Metric{{
			Name: "gate_wall_ns", Value: float64(wall),
			Unit: "ns", Direction: runrecord.DirectionMinimize,
		}})
		if err != nil {
			return err
		}
		evaluationContent, err := evaluation.Content()
		if err != nil {
			return err
		}
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: workloadID})
		batch.Contents = append(batch.Contents, evaluationContent)
		batch.Lineage = append(batch.Lineage, evaluation.Lineage()...)
	}

	store, err := g.openStore()
	if err != nil {
		return g.oweRecord(batch, err)
	}
	// Cost per accepted checkpoint and reuse saving against this row's prior
	// gate results; advisory audit, derived from stored results only.
	g.batchCostAudit(context.Background(), store, g.steps, record.Run.MeasuredNS)
	if err := requireSoleCurrentGatePreparation(
		context.Background(), store, g.preparation, g.preparationCommit,
	); err != nil {
		// This batch would create a second finalization or advance an alias
		// the gate no longer owns. Do not persist it as debt: retain the
		// existing lifecycle and heartbeat for explicit recovery instead.
		return fmt.Errorf("gate: final record lost preparation authority: %w", err)
	}
	// Digests only: the full manifest is derivable from git at this
	// commit, and persisting it per run was the store's growth curve.
	for _, manifest := range []*codemanifest.Manifest{g.baseManifest, g.candidateManifest} {
		if manifest == nil {
			continue
		}
		if _, err := codemanifest.PublishDigest(context.Background(), store, *manifest); err != nil {
			return g.oweRecord(batch, err)
		}
	}
	if err := appendGateAdvisoryFinding(context.Background(), store, &batch, g.paths, g.audit); err != nil {
		return g.oweRecord(batch, err)
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return g.oweRecord(batch, err)
	}
	_ = fsatomic.Remove(filepath.Join(g.repo, filepath.FromSlash(gateDebtFile)))
	return nil
}

func (g *gateContext) gateRecipeID() (artifact.ID, error) {
	if g.manifestPlan != nil {
		if err := g.manifestPlan.Validate(); err != nil {
			return artifact.ID{}, err
		}
		return g.manifestPlan.ID, nil
	}
	return artifact.IdentifyBytes(artifact.KindRecipe, []byte(gateRecipeSeed))
}

func (g *gateContext) appendProfileEvidence(batch *artifact.Batch, codeCommit string, gateResult artifact.ID) error {
	if g.profile == nil {
		return nil
	}
	if g.profileDirty {
		g.note("code profile evidence not persisted: unplanned Go dirt is outside the committed target")
		return nil
	}
	evidence, err := codeprofile.NewEvidence(codeCommit, gateResult, *g.profile)
	if err != nil {
		return err
	}
	content, err := evidence.Content()
	if err != nil {
		return err
	}
	batch.Contents = append(batch.Contents, content)
	batch.Lineage = append(batch.Lineage, evidence.Lineage()...)
	return nil
}

// appendAttemptRecord rides the gate's own record batch with the
// typed attempt document: the plan step this run served, the manifest
// selection it observed, and the diff it carried, on success and
// failure alike -- the measurement unit for automation effectiveness.
func (g *gateContext) appendAttemptRecord(
	batch *artifact.Batch,
	codeCommit string,
	recipeID, resultID artifact.ID,
	outcome runrecord.Outcome,
	failure string,
) error {
	item, step, bound := strings.Cut(g.planRef, "/")
	if !bound {
		return fmt.Errorf("gate: attempt record requires an item/step plan reference, got %q", g.planRef)
	}
	wall, err := g.clock.Elapsed()
	if err != nil {
		return err
	}
	attempt := runrecord.AttemptRecord{
		PlanItem: item, PlanStep: step, Result: resultID, Recipe: recipeID,
		// The driver exports the declared strategy; an interactive
		// session leaves it empty and the record stays exact.
		Strategy:   os.Getenv(loop.StrategyEnvironment),
		CodeCommit: codeCommit, Outcome: outcome, Failure: failure,
		WallNS: wall,
		Selection: runrecord.AttemptSelection{
			Defined: g.manifestMetrics.Defined, Selected: g.manifestMetrics.Selected,
			Excluded: g.manifestMetrics.Excluded, Uncertainty: g.manifestMetrics.Uncertainty,
			CacheEligible: g.manifestMetrics.CacheEligible, CacheHits: g.manifestMetrics.CacheHits,
			PlanningNS: g.manifestMetrics.PlanningNS,
		},
		Diff: g.diff,
	}
	if g.baseManifest != nil {
		attempt.BaseManifest = g.baseManifest.ID
	}
	if g.candidateManifest != nil {
		attempt.CandidateManifest = g.candidateManifest.ID
	}
	var published runrecord.AttemptRecord
	if g.strategy == nil {
		published, err = runrecord.NewAttemptRecord(attempt)
	} else {
		published, err = g.strategy.StampAttempt(attempt)
	}
	if err != nil {
		return err
	}
	content, err := published.Content()
	if err != nil {
		return err
	}
	batch.Contents = append(batch.Contents, content)
	batch.Lineage = append(batch.Lineage, published.Lineage()...)
	return nil
}
