package gate

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/authoritylock"
	"overgo/internal/automationcheck"
	"overgo/internal/gitauthority"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
)

// Options carries the parsed command surface into the one gate
// transaction path. cmd/gate owns the flags; this package owns everything
// after them.
type Options struct {
	MessageFile        string
	PathsCSV           string
	StorePath          string
	Merge              bool
	PlanProjection     string
	MergeSourceStore   string
	PlanRef            string
	Reconcile          bool
	RecordFailure      bool
	RecoverPreparation string
	RecoverInterrupted bool
	AdmitReview        string
	Watchdog           bool
	InspectPlan        bool
	StaleAfter         time.Duration
}

// Run executes the single gate transaction path for one parsed option set.
func Run(options Options) error {
	messageFile := &options.MessageFile
	pathsCSV := &options.PathsCSV
	storePath := &options.StorePath
	merge := &options.Merge
	planProjectionFlag := &options.PlanProjection
	mergeSourceStoreFlag := &options.MergeSourceStore
	planRef := &options.PlanRef
	reconcile := &options.Reconcile
	recordFailure := &options.RecordFailure
	recoverPreparation := &options.RecoverPreparation
	recoverInterrupted := &options.RecoverInterrupted
	admitReview := &options.AdmitReview
	watchdog := &options.Watchdog
	inspectPlan := &options.InspectPlan
	staleAfter := &options.StaleAfter
	repo, err := os.Getwd()
	if err != nil {
		return err
	}
	if err := gitauthority.RequireRepositoryRoot(context.Background(), repo); err != nil {
		return fmt.Errorf("gate: establish exact repository authority: %w", err)
	}
	cleanStore := filepath.Clean(*storePath)
	if cleanStore == "." || filepath.IsAbs(cleanStore) || cleanStore == ".." || strings.HasPrefix(cleanStore, ".."+string(filepath.Separator)) {
		return errors.New("gate: store path must stay below the repository root")
	}
	if err := requireExclusiveGateMode(
		*reconcile, *recordFailure, *recoverInterrupted, *admitReview != "", *watchdog, *inspectPlan, *merge,
	); err != nil {
		return err
	}
	planProjection, err := gatePlanProjection(*planProjectionFlag, *merge)
	if err != nil {
		return err
	}
	mergeSourceStore, err := gateMergeSourceStore(*mergeSourceStoreFlag, *merge, planProjection)
	if err != nil {
		return err
	}
	mutating := *reconcile || *recordFailure || *recoverInterrupted || *admitReview != "" || !*watchdog && !*inspectPlan
	writesGit := gateWritesGit(
		*reconcile, *recordFailure, *recoverInterrupted, *admitReview != "", *watchdog, *inspectPlan, *merge,
	)
	if writesGit {
		if err := gitauthority.RequireWriterSupport(context.Background(), repo); err != nil {
			return fmt.Errorf("gate: durable Git writer unavailable: %w", err)
		}
	}
	if err := requireCanonicalGateStore(cleanStore, mutating); err != nil {
		return err
	}
	if mutating {
		lock, err := authoritylock.Acquire(repo)
		if err != nil {
			return err
		}
		defer lock.Close()
	}
	if *reconcile {
		preparation, err := reconcileGateDebt(repo, cleanStore)
		if err == nil {
			fmt.Printf("gate: reconciled OvergoDB record debt for %s\n", preparation)
		}
		return err
	}
	if *recoverPreparation != "" && !*recordFailure {
		return errors.New("gate: -preparation requires -record-failure")
	}
	if *recordFailure {
		preparation, err := recordSelectedUnbatchableFailure(repo, cleanStore, *recoverPreparation)
		if err == nil {
			fmt.Printf("gate: recorded failed finalization for %s\n", preparation)
		}
		return err
	}
	if *recoverInterrupted {
		preparation, err := recoverInterruptedCommit(repo, cleanStore)
		if err == nil {
			fmt.Printf("gate: recovered interrupted commit for %s\n", preparation)
		}
		return err
	}
	if *admitReview != "" {
		head, err := command(repo, "git", "rev-parse", "HEAD")
		if err != nil {
			return err
		}
		if err := admitStoredReview(repo, cleanStore, *admitReview, strings.TrimSpace(head)); err != nil {
			return err
		}
		fmt.Printf("gate: review %s admitted at %s\n", *admitReview, strings.TrimSpace(head))
		return nil
	}
	if *watchdog {
		return printGateWatchdog(repo, *staleAfter)
	}
	if *inspectPlan && *merge {
		return errors.New("gate: -inspect-plan requires explicit -paths and cannot inspect an in-progress merge")
	}
	if (*pathsCSV == "" && !*merge) || (!*inspectPlan && *messageFile == "") {
		return fmt.Errorf("usage: gate -message-file <path> (-paths <csv> | -merge) -plan <item>/<step> [-store <dir>]")
	}
	// Every commit -- including a merge finalize -- is bound to the plan's current
	// open step. Merges are no longer exempt: a sync/merge is a first-class plan
	// task (inject it with `plan -add`, then finalize with -plan <item>/do).
	var completionAuthority plan.CompletionAuthority
	var planHead string
	var indexBefore gateIndexSnapshot
	var mergeBefore *gateMergeIntent
	var admissionStore *overgodb.Store
	defer func() {
		if admissionStore != nil {
			_ = admissionStore.Close()
		}
	}()
	if !*inspectPlan {
		err = reportGateAdmissionPhase("capture candidate state", func() error {
			mergeBefore, indexBefore, err = captureGateStartState(repo)
			return err
		})
		if err != nil {
			return err
		}
		err = reportGateAdmissionPhase("open canonical store", func() error {
			admissionStore, err = overgodb.Open(filepath.Join(repo, cleanStore))
			return err
		})
		if err != nil {
			return fmt.Errorf("gate: open admission store: %w", err)
		}
		if err := reportGateAdmissionPhase("validate pending lifecycle state", func() error {
			return requireNoPendingGateStateWithStore(repo, admissionStore)
		}); err != nil {
			return err
		}
		var accelerated bool
		err = reportGateAdmissionPhase("publish replay checkpoint", func() error {
			accelerated, err = ensureGateStoreAcceleration(context.Background(), admissionStore)
			return err
		})
		if err != nil {
			return err
		}
		if accelerated {
			fmt.Fprintln(os.Stderr, "gate: admission published a checkpoint for the verified cold replay")
		}
		err = reportGateAdmissionPhase("resolve plan authority", func() error {
			completionAuthority, planHead, err = resolvePlanBindingWithStore(repo, *planRef, admissionStore)
			return err
		})
		if err != nil {
			return err
		}
	}
	g := &gateContext{
		repo: repo, planRef: *planRef, messageFile: *messageFile, storePath: cleanStore, start: time.Now(),
		stepEvidence: map[string]string{}, terminal: map[string]automationcheck.Evidence{},
		completionAuthority: completionAuthority, planHead: planHead,
		indexBefore: indexBefore, mergeBefore: mergeBefore,
		planProjection: planProjection, mergeSourceStore: mergeSourceStore,
	}
	defer g.closeCompletionStore()
	if *merge {
		// Merge mode: the staged merge IS the plan. Deriving -paths from the
		// staged set makes the scope step trivially pass, and the existing
		// commit step constructs the exact two-parent commit and advances the
		// captured branch by compare-and-swap. Merges otherwise
		// bypass the gate entirely; this runs full hygiene + impacted tests on
		// the merged tree before the commit lands.
		if _, err := command(repo, "git", "rev-parse", "--verify", "-q", "MERGE_HEAD"); err != nil {
			return fmt.Errorf("-merge needs an in-progress merge (no MERGE_HEAD): run `git merge --no-ff --no-commit <branch>` first")
		}
		if g.mergeSourceStore != "" {
			if g.mergeBefore == nil {
				return errors.New("-merge-source-store requires captured merge metadata")
			}
			parents := g.mergeBefore.parents()
			if len(parents) != 1 {
				return errors.New("-merge-source-store requires exactly one captured merge parent")
			}
			registered, err := gitauthority.RequireRegisteredWorktreeStore(
				context.Background(), repo, g.mergeSourceStore, parents[0],
			)
			if err != nil {
				return fmt.Errorf("-merge-source-store: %w", err)
			}
			localStore, err := filepath.Abs(filepath.Join(repo, gateStorePath))
			if err != nil {
				return err
			}
			if strings.EqualFold(filepath.Clean(registered), filepath.Clean(localStore)) {
				return errors.New("-merge-source-store must be owned by the distinct source worktree")
			}
			g.mergeSourceStore = registered
		}
		staged, err := stagedPaths(repo)
		if err != nil {
			return err
		}
		if len(staged) == 0 {
			return fmt.Errorf("-merge: the staged merge set is empty")
		}
		for _, p := range staged {
			g.paths = append(g.paths, filepath.ToSlash(p))
		}
	} else {
		for _, p := range strings.Split(*pathsCSV, ",") {
			if p = strings.TrimSpace(p); p != "" {
				g.paths = append(g.paths, filepath.ToSlash(p))
			}
		}
	}
	if err := validatePlannedPaths(g.paths); err != nil {
		return err
	}
	if err := g.expandDirectoryPaths(); err != nil {
		return err
	}
	if !slices.Contains(g.paths, plan.Path) {
		g.paths = append(g.paths, plan.Path)
	}
	if err := validatePlannedPaths(g.paths); err != nil {
		return err
	}
	if *inspectPlan {
		planned, err := g.planPipeline()
		if err != nil {
			return err
		}
		return writeGatePlanReport(os.Stdout, planned)
	}
	err = reportGateAdmissionPhase("discover verification environment", func() error {
		g.environment, err = discoverEnvironment(repo)
		return err
	})
	if err != nil {
		return err
	}
	if err := reportGateAdmissionPhase("publish lifecycle preparation", func() error {
		return g.prepareWithStore(admissionStore)
	}); err != nil {
		return err
	}
	if err := admissionStore.Close(); err != nil {
		return fmt.Errorf("gate: close admission store before verification: %w", err)
	}
	admissionStore = nil
	stopHeartbeat, err := g.startHeartbeat()
	if err != nil {
		return err
	}
	defer stopHeartbeat()

	// The candidate's change size is observed before the pipeline can
	// commit it; after a successful commit the worktree diff is gone.
	g.diff = observeDiff(repo)
	outcome := runrecord.OutcomeSucceeded
	failureCode := ""
	var pipelineErr error
	if pipelineErr = g.pipeline(); pipelineErr != nil {
		outcome = runrecord.OutcomeFailed
		failureCode = "planning"
		if len(g.steps) != 0 {
			failureCode = g.steps[len(g.steps)-1].Name
		}
	}
	// No running heartbeat writer may outlive the gate execution and race a
	// terminal store/heartbeat publication. A crash after this point leaves the
	// last running locator for the explicit recovery path.
	stopHeartbeat()
	if g.commitInterrupted {
		closeErr := g.closeCompletionStore()
		_, recoveryErr := recoverInterruptedCommit(repo, cleanStore)
		g.printSummary(os.Stdout, outcome, pipelineErr.Error())
		if closeErr != nil || recoveryErr != nil {
			_ = g.writeHeartbeat(runrecord.HeartbeatRecordDebt)
			return errors.Join(
				pipelineErr,
				fmt.Errorf(
					"commit landed but automatic recovery failed; run `go run ./cmd/gate -recover-interrupted`: %w",
					errors.Join(closeErr, recoveryErr),
				),
			)
		}
		_ = g.writeHeartbeat(runrecord.HeartbeatFinalized)
		return pipelineErr
	}
	recordErr := g.record(outcome, failureCode)
	recordErr = errors.Join(recordErr, g.closeCompletionStore())
	// Intent removal mutates recovery authority and therefore must remain lazy;
	// cmp.Or would evaluate the removal even when record publication failed.
	switch {
	case recordErr != nil:
	default:
		recordErr = removeGateCommitIntent(repo)
	}
	rolledBackRecord := false
	if recordErr != nil && g.committedHead != "" {
		if _, debtErr := os.Stat(filepath.Join(repo, filepath.FromSlash(gateDebtFile))); errors.Is(debtErr, os.ErrNotExist) {
			if _, recoveryErr := recoverInterruptedCommit(repo, cleanStore); recoveryErr != nil {
				recordErr = errors.Join(
					recordErr,
					fmt.Errorf("automatic rollback after unbatchable record failure: %w", recoveryErr),
				)
			} else if recoveredHead, headErr := command(repo, "git", "rev-parse", "HEAD"); headErr != nil {
				recordErr = errors.Join(recordErr, fmt.Errorf("inspect automatic record recovery: %w", headErr))
			} else {
				switch strings.TrimSpace(recoveredHead) {
				case g.planHead:
					rolledBackRecord = true
					outcome = runrecord.OutcomeCancelled
				case g.committedHead:
					// The final batch landed and recovery only cleared a stale
					// marker, so the originally reported record error is resolved.
					recordErr = nil
				default:
					recordErr = errors.Join(recordErr, errors.New("automatic record recovery left HEAD at an unknown commit"))
				}
			}
		} else if debtErr != nil {
			recordErr = errors.Join(recordErr, fmt.Errorf("inspect gate record debt: %w", debtErr))
		}
	}
	failureDetail := ""
	if pipelineErr != nil {
		failureDetail = pipelineErr.Error()
	}
	g.printSummary(os.Stdout, outcome, failureDetail)
	if recordErr != nil {
		if rolledBackRecord {
			_ = g.writeHeartbeat(runrecord.HeartbeatFinalized)
			fmt.Fprintf(os.Stderr, "gate: store record failed; commit was rolled back and the preparation cancelled: %v\n", recordErr)
		} else {
			_ = g.writeHeartbeat(runrecord.HeartbeatRecordDebt)
			fmt.Fprintf(os.Stderr, "gate: store record failed (recovery or record debt remains): %v\n", recordErr)
		}
	} else {
		_ = g.writeHeartbeat(runrecord.HeartbeatFinalized)
	}
	if pipelineErr != nil {
		return pipelineErr
	}
	if recordErr != nil {
		if rolledBackRecord {
			return fmt.Errorf("commit rolled back because its OvergoDB record could not be constructed: %w", recordErr)
		}
		return fmt.Errorf("commit landed but OvergoDB record debt remains: %w", recordErr)
	}
	return nil
}

// reportGateAdmissionPhase makes pre-heartbeat work observable. Admission must
// finish before a durable lifecycle exists, so these progress records describe
// the exact phase and duration without pretending that a preparation has been
// published already.
func reportGateAdmissionPhase(name string, action func() error) error {
	started := time.Now()
	fmt.Fprintf(os.Stderr, "gate: admission %s started\n", name)
	err := action()
	status := "completed"
	if err != nil {
		status = "failed"
	}
	fmt.Fprintf(
		os.Stderr,
		"gate: admission %s %s in %s\n",
		name,
		status,
		time.Since(started).Round(time.Millisecond),
	)
	return err
}
