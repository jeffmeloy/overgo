// Package gate is the commit gate: one transaction path owns scope
// refusal, hygiene, derived test scope, claim/manifest/SBOM verification,
// the scoped commit, and the store record. Never raw `git commit` during a
// campaign — the guard enforces that; Run is the sanctioned path and
// constructs the exact accepted commit object before compare-and-swapping
// the captured branch reference. cmd/gate owns only flags and composition.
//
// The hierarchy is phase-named: admission (mode and scope refusal),
// preparation (manifest planning and preparation evidence), verification
// (the check pipeline), commit (the two-phase Git transaction),
// finalization (the durable record), and recovery (interrupted-commit and
// debt reconciliation). Splitting never created a second gate: every phase
// runs only inside Run.
//
// Message files preserve shell-sensitive prose, scope is exact, and OvergoDB is
// authoritative; green output names what did not run.
package gate

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/clioptions"
	"overgo/internal/codemanifest"
	"overgo/internal/codeprofile"
	"overgo/internal/finding"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/loop"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
)

const (
	gateRecipeSeed   = "overgo-gate/v1"
	gateWorkloadSeed = "overgo-gate-workload/v1"
	gateStorePath    = "overgodb-store"
	// Gate state lives in tmp/, the sanctioned scrap home: bin/ holds
	// executables only and the release refuses anything else in it.
	gateDebtFile         = "tmp/gate_debt.json"
	gateCommitIntentFile = "tmp/gate_commit_intent.json"
	gateHeartbeatFile    = "tmp/gate_lifecycle.json"
	gateRetryFile        = "tmp/gate_cache.json"
	gatePlanScratchDir   = "tmp/gate_plan_scratch"
	gateGitStateLockFile = "overgo-gate-git-state.lock"
	gateProgressLine     = "gate: phase=%s heartbeat=%s\n"
	// Recovery roots and transient Git authority must not be readable by
	// other users. Directory traversal is likewise restricted to the owner.
	gatePrivateFileMode      = clioptions.PrivateFileMode
	gatePrivateDirectoryMode = fs.FileMode(0o700)
)

type gateContext struct {
	repo                string
	candidateRoot       string
	candidateTree       string
	paths               []string
	planRef             string
	messageFile         string
	storePath           string
	steps               []runrecord.GateStep
	audit               []string
	start               time.Time
	environment         runrecord.Environment
	preparation         runrecord.GateLifecycle
	preparationCommit   artifact.CommitID
	source              *repoanalysis.SourceSnapshot
	baseSource          *repoanalysis.SourceSnapshot
	profile             *codeprofile.Profile
	profileDirty        bool
	preflight           bool
	stepEvidence        map[string]string
	cachePaths          []string
	retryCache          *automationcheck.EvidenceCache
	checkpointMemos     map[string]checkpointMemoEntry
	verificationBatch   *plan.VerificationBatch
	structural          *codeprofile.FunctionImpact
	packageGraph        *packageInputGraph
	selection           automationcheck.SelectionMetrics
	selectionID         string
	manifestPlan        *automationcheck.ManifestPlan
	terminal            map[string]automationcheck.Evidence
	baseManifest        *codemanifest.Manifest
	candidateManifest   *codemanifest.Manifest
	manifestDelta       *codemanifest.Delta
	manifestImpact      *codemanifest.Impact
	manifestMetrics     automationcheck.ManifestMeasurements
	strategy            *loop.Strategy
	diff                runrecord.AttemptDiff
	completionAuthority plan.CompletionAuthority
	dispatchClaim       *plan.WorkLease
	store               *overgodb.Store
	indexBefore         gateIndexSnapshot
	mergeBefore         *gateMergeIntent
	planProjection      plan.MergeProjection
	mergeSourceStore    string
	mergeAuthority      *plan.FirstParentTargetMergeAuthority
	planHead            string
	committedHead       string
	commitInterrupted   bool
	acceptedTree        string
	// runCommand overrides supervised command execution for remediation
	// tests; nil routes through the package command runner.
	runCommand func(repo, name string, args ...string) (string, error)
	// The validate wave runs its checks concurrently; the snapshots and
	// the audit they share are written under these locks.
	sourceMutex   sync.Mutex
	auditMutex    sync.Mutex
	terminalMutex sync.Mutex
	modernInput   *modernGoInput
	// The planned tree is built once per candidate state and the plan is
	// parsed once for the admission and verification readers; the commit
	// phase rewrites the plan after every such reader has run.
	plannedTreeMutex       sync.Mutex
	plannedTreeID          string
	plannedTreeFingerprint string
	plannedTreeBuilds      int
	planDocument           *plan.Plan
	planLoads              int
	// testPlan carries the prepared test groups from the changed-owners
	// check to the check that runs the rest beside the lanes.
	testPlan *testGroups
	// Protected by auditMutex; each invocation retains its own cost and profile.
	testExecutions []packageExecutionBatch
	// testStep names the check whose package executions are being recorded;
	// the test checks run one after another.
	testStep string
	// selectionCauses retains every requested package's selection
	// attribution and observed execution for the final record.
	selectionCauses []runrecord.SelectionPackage
}

// setTestStep names the check that owns the next package executions.
func (g *gateContext) setTestStep(name string) {
	g.auditMutex.Lock()
	defer g.auditMutex.Unlock()
	g.testStep = name
}

// appends one audit line under the lock the concurrent validate wave shares
func (g *gateContext) note(line string) {
	g.auditMutex.Lock()
	defer g.auditMutex.Unlock()
	g.audit = append(g.audit, line)
}

// returns the plan parsed once for the gate's readers that run before the
// commit phase rewrites it
func (g *gateContext) loadPlan() (plan.Plan, error) {
	if g.planDocument != nil {
		return *g.planDocument, nil
	}
	document, err := plan.Load(filepath.Join(g.repo, plan.Path))
	if err != nil {
		return plan.Plan{}, err
	}
	g.planDocument = &document
	g.planLoads++
	return document, nil
}

func (g *gateContext) runGateCommand(root, name string, args ...string) (string, error) {
	if g.runCommand != nil {
		return g.runCommand(root, name, args...)
	}
	environment, err := g.sourceEnvironment()
	if err != nil {
		return "", err
	}
	return commandEnvironment(root, environment, name, args...)
}

// openStore retains one replayed handle; writes still acquire transaction locks.
func (g *gateContext) openStore() (*overgodb.Store, error) {
	if g.storePath == "" || !g.environment.ID.Valid() {
		return nil, errors.New("gate store: canonical path and environment are required")
	}
	if g.store == nil {
		// Another writer, a lane recording its receipts beside the test
		// groups, is waited out under the host batch budget instead of
		// refused on the store's OS exclusion.
		ctx, cancel := context.WithTimeoutCause(context.Background(), testAdmissionBudget, errStoreAdmissionBudget)
		defer cancel()
		began := time.Now()
		store, err := overgodb.OpenContext(ctx, filepath.Join(g.repo, g.storePath))
		if err != nil {
			return nil, err
		}
		if waited := time.Since(began); waited > time.Second {
			g.note(fmt.Sprintf("store admission waited %s for another writer", waited.Round(time.Millisecond)))
		}
		g.store = store
	}
	return g.store, nil
}

// names the exhausted store admission budget as the cancellation cause
var errStoreAdmissionBudget = errors.New("gate store admission budget exhausted")

func (g *gateContext) closeStore() error {
	if g == nil || g.store == nil {
		return nil
	}
	err := g.store.Close()
	g.store = nil
	return err
}

const (
	historicalRankingMarker = "<!-- overgo-document: historical-ranking -->"
	currentWorkMarker       = "<!-- overgo-current-work: docs/plan.json -->"
)

var (
	errGateIndexMoved            = errors.New("Git index moved beyond the write-ahead states")
	gateIndexCASBeforeLockHook   func(string)
	gateIndexCASLockedHook       func(string)
	gateMergeCASBeforeLockHook   func(string)
	gateMergeCASLockedHook       func(string)
	gateMergeCASAfterStepHook    func(string, string, string)
	gateCommitIndexInstalledHook func(string, gateCommitIntent)
	gateKeepaliveBoundHook       func(string, gateCommitIntent)
	gateRecoveryBeforeLockHook   func(string)
	gateRecoveryLockedHook       func(string)
	gateRecoveryAfterStepHook    func(string, string)
	gateCheckPersistedHook       func(string)
)

func (transaction *gatePreparedReferenceTransaction) exchange(command, expected string) error {
	if transaction == nil || transaction.finished {
		return errors.New("gate: prepared ref transaction is not active")
	}
	if _, err := io.WriteString(transaction.input, command+"\n"); err != nil {
		return err
	}
	response, err := transaction.output.ReadString('\n')
	if err != nil {
		return err
	}
	if strings.TrimSpace(response) != expected {
		return fmt.Errorf("unexpected Git ref transaction response %q", strings.TrimSpace(response))
	}
	return nil
}

func (transaction *gatePreparedReferenceTransaction) fail(phase string, cause error) error {
	if transaction == nil {
		return cause
	}
	_ = transaction.input.Close()
	waitErr := transaction.process.Wait()
	transaction.finished = true
	return fmt.Errorf(
		"gate: prepared ref transaction %s: %w: %s",
		phase, errors.Join(cause, waitErr), strings.TrimSpace(transaction.stderr.String()),
	)
}

func (transaction *gatePreparedReferenceTransaction) commit() error {
	if transaction == nil || transaction.finished {
		return errors.New("gate: prepared ref transaction is not active")
	}
	if err := transaction.exchange("commit", "commit: ok"); err != nil {
		return transaction.fail("commit", err)
	}
	closeErr := transaction.input.Close()
	waitErr := transaction.process.Wait()
	transaction.finished = true
	if err := errors.Join(closeErr, waitErr); err != nil {
		return fmt.Errorf(
			"gate: finish prepared ref transaction: %w: %s",
			err, strings.TrimSpace(transaction.stderr.String()),
		)
	}
	return nil
}

func (transaction *gatePreparedReferenceTransaction) abort() error {
	if transaction == nil || transaction.finished {
		return nil
	}
	if err := transaction.exchange("abort", "abort: ok"); err != nil {
		return transaction.fail("abort", err)
	}
	closeErr := transaction.input.Close()
	waitErr := transaction.process.Wait()
	transaction.finished = true
	if err := errors.Join(closeErr, waitErr); err != nil {
		return fmt.Errorf(
			"gate: abort prepared ref transaction: %w: %s",
			err, strings.TrimSpace(transaction.stderr.String()),
		)
	}
	return nil
}

func (intent gateCommitIntent) validate() error {
	projection, projectionErr := plan.ParseMergeProjection(string(intent.PlanProjection))
	if projectionErr != nil || projection != intent.PlanProjection ||
		projection == plan.MergeProjectionFirstParentTarget && intent.Merge == nil {
		return errors.New("gate: invalid interrupted commit plan projection")
	}
	if projection == plan.MergeProjectionFirstParentTarget {
		receipt, err := plan.ParseFirstParentTargetMergeAuthority(intent.MergeAuthority)
		if err != nil || receipt.ID.Kind() != artifact.KindEvidence {
			return errors.New("gate: invalid interrupted projected-merge authority")
		}
	} else if len(intent.MergeAuthority) != 0 {
		return errors.New("gate: semantic-union intent carries projected-merge authority")
	}
	if intent.Version != artifact.InitialDocumentVersion ||
		intent.Preparation.Kind() != artifact.KindEvidence || !intent.PreparationCommit.Valid() ||
		intent.Recipe.Kind() != artifact.KindRecipe ||
		intent.CandidateManifest.Kind() != artifact.KindProfile ||
		!validGitObjectID(intent.Parent) || !validGitObjectID(intent.Commit) ||
		!validGitObjectID(intent.IndexTree) || !validGitObjectID(intent.Tree) ||
		!validGitObjectID(intent.KeepaliveTree) || !validGitObjectID(intent.KeepaliveCommit) ||
		len(intent.IndexBefore) == 0 || len(intent.IndexAfter) == 0 || len(intent.IndexRestore) == 0 ||
		intent.IndexMode == 0 || intent.IndexMode > uint32(fs.ModePerm) ||
		len(intent.Plan) == 0 || len(intent.AdvancedPlan) == 0 ||
		strings.TrimSpace(intent.HeadReference) != intent.HeadReference || intent.HeadReference == "" ||
		strings.ContainsAny(intent.HeadReference, "\x00\r\n") || intent.PlanMode > uint32(fs.ModePerm) ||
		intent.PlanMode == 0 {
		return errors.New("gate: invalid interrupted commit intent")
	}
	if intent.KeepaliveRef != gateIntentKeepaliveReference(intent.Preparation) {
		return errors.New("gate: interrupted commit intent has invalid keepalive reference authority")
	}
	if intent.Merge != nil {
		if err := intent.Merge.validate(intent.Parent); err != nil {
			return err
		}
		if intent.Merge.IndexTree != intent.IndexTree {
			return errors.New("gate: interrupted merge and shared-index authorities differ")
		}
	}
	if len(intent.Paths) == 0 {
		return errors.New("gate: interrupted commit intent has no scoped paths")
	}
	containsPlan := false
	seenPaths := make(map[string]bool, len(intent.Paths))
	for _, candidate := range intent.Paths {
		clean := filepath.Clean(filepath.FromSlash(candidate))
		if candidate == "" || clean == "." || filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return errors.New("gate: interrupted commit intent has an invalid scoped path")
		}
		canonical := filepath.ToSlash(clean)
		if seenPaths[canonical] {
			return errors.New("gate: interrupted commit intent repeats a scoped path")
		}
		seenPaths[canonical] = true
		containsPlan = containsPlan || canonical == plan.Path
	}
	if !containsPlan {
		return errors.New("gate: interrupted commit intent does not scope the plan")
	}
	item, step, found := strings.Cut(intent.PlanRef, "/")
	if !found || item == "" || step == "" {
		return errors.New("gate: interrupted commit intent has invalid plan reference")
	}
	before, err := plan.Parse(intent.Plan)
	if err != nil {
		return fmt.Errorf("gate: interrupted commit intent has invalid plan bytes: %w", err)
	}
	after, err := plan.Parse(intent.AdvancedPlan)
	if err != nil {
		return fmt.Errorf("gate: interrupted commit intent has invalid advanced plan bytes: %w", err)
	}
	expected, err := plan.Advance(before, item, step)
	if err != nil {
		return fmt.Errorf("gate: interrupted commit intent cannot derive its plan transition: %w", err)
	}
	expectedJSON, err := json.Marshal(expected)
	if err != nil {
		return err
	}
	afterJSON, err := json.Marshal(after)
	if err != nil {
		return err
	}
	if !bytes.Equal(expectedJSON, afterJSON) {
		return errors.New("gate: interrupted commit intent has the wrong advanced plan")
	}
	return nil
}

func (merge gateMergeIntent) validate(parent string) error {
	if !validGitObjectID(merge.IndexTree) || len(merge.Head) == 0 || len(merge.Message) == 0 {
		return errors.New("gate: interrupted commit intent has invalid merge state")
	}
	for _, mode := range []uint32{merge.HeadFileMode, merge.ModeFileMode, merge.MessageFileMode} {
		if mode == 0 || mode > uint32(fs.ModePerm) {
			return errors.New("gate: interrupted commit intent has invalid merge metadata permissions")
		}
	}
	parents := strings.Fields(string(merge.Head))
	if len(parents) != 1 || !validGitObjectID(parents[0]) || parents[0] == parent {
		return errors.New("gate: interrupted commit intent requires one distinct merge parent")
	}
	if merge.AutoMerge != nil {
		autoMerge := strings.TrimSpace(string(merge.AutoMerge))
		if merge.AutoMergeFileMode == 0 || merge.AutoMergeFileMode > uint32(fs.ModePerm) ||
			!validGitObjectID(autoMerge) ||
			!bytes.Equal(merge.AutoMerge, []byte(autoMerge)) &&
				!bytes.Equal(merge.AutoMerge, []byte(autoMerge+"\n")) {
			return errors.New("gate: interrupted commit intent has invalid AUTO_MERGE authority")
		}
	} else if merge.AutoMergeFileMode != 0 {
		return errors.New("gate: interrupted commit intent has AUTO_MERGE permissions without content")
	}
	return nil
}

func (merge gateMergeIntent) parents() []string {
	return strings.Fields(string(merge.Head))
}

func gateMergeManagedMetadata(files []gateMergeMetadataFile) []gateMergeMetadataFile {
	managed := make([]gateMergeMetadataFile, 0, len(files))
	for _, file := range files {
		if file.managed {
			managed = append(managed, file)
		}
	}
	return managed
}

type gateGitLockPath struct {
	name string
	path string
	mode fs.FileMode
}

type gateGitLockMarker struct {
	Version            uint16 `json:"version"`
	Authority          string `json:"authority"`
	Preparation        string `json:"preparation"`
	KeepaliveReference string `json:"keepalive_reference"`
	KeepaliveCommit    string `json:"keepalive_commit"`
	WorktreeGitDir     string `json:"worktree_git_dir"`
	IndexPath          string `json:"index_path"`
	IntentSHA256       string `json:"intent_sha256"`
}

type gateGitLockIntentAuthority struct {
	Version           uint16               `json:"version"`
	Preparation       string               `json:"preparation"`
	PreparationCommit string               `json:"preparation_commit"`
	Recipe            string               `json:"recipe"`
	CandidateManifest string               `json:"candidate_manifest"`
	Parent            string               `json:"parent"`
	HeadReference     string               `json:"head_reference"`
	Merge             *gateMergeIntent     `json:"merge,omitempty"`
	PlanProjection    plan.MergeProjection `json:"plan_projection,omitzero"`
	MergeAuthority    []byte               `json:"merge_authority,omitempty"`
	Commit            string               `json:"commit,omitzero"`
	IndexTree         string               `json:"index_tree"`
	Tree              string               `json:"tree"`
	KeepaliveRef      string               `json:"keepalive_ref"`
	KeepaliveTree     string               `json:"keepalive_tree"`
	KeepaliveCommit   string               `json:"keepalive_commit"`
	IndexBefore       []byte               `json:"index_before"`
	IndexAfter        []byte               `json:"index_after"`
	IndexRestore      []byte               `json:"index_restore"`
	IndexMode         uint32               `json:"index_mode"`
	PlanRef           string               `json:"plan_ref"`
	Paths             []string             `json:"paths"`
	Plan              []byte               `json:"plan"`
	AdvancedPlan      []byte               `json:"advanced_plan"`
	PlanMode          uint32               `json:"plan_mode"`
}

func gateGitLockMarkerBytes(repo, indexPath string, intent gateCommitIntent) ([]byte, error) {
	if !intent.Preparation.Valid() || intent.KeepaliveRef != gateIntentKeepaliveReference(intent.Preparation) ||
		!validGitObjectID(intent.KeepaliveCommit) {
		return nil, errors.New("gate: Git lock marker has invalid interrupted-intent authority")
	}
	if err := requireGateIntentKeepalive(repo, intent); err != nil {
		return nil, err
	}
	canonicalIntent, err := json.Marshal(gateGitLockIntentAuthority{
		Version: intent.Version, Preparation: intent.Preparation.String(),
		PreparationCommit: intent.PreparationCommit.String(), Recipe: intent.Recipe.String(),
		CandidateManifest: intent.CandidateManifest.String(), Parent: intent.Parent,
		HeadReference: intent.HeadReference, Merge: intent.Merge, PlanProjection: intent.PlanProjection,
		MergeAuthority: intent.MergeAuthority,
		Commit:         intent.Commit,
		IndexTree:      intent.IndexTree, Tree: intent.Tree, KeepaliveRef: intent.KeepaliveRef,
		KeepaliveTree: intent.KeepaliveTree, KeepaliveCommit: intent.KeepaliveCommit,
		IndexBefore: intent.IndexBefore, IndexAfter: intent.IndexAfter, IndexRestore: intent.IndexRestore,
		IndexMode: intent.IndexMode, PlanRef: intent.PlanRef, Paths: intent.Paths,
		Plan: intent.Plan, AdvancedPlan: intent.AdvancedPlan, PlanMode: intent.PlanMode,
	})
	if err != nil {
		return nil, err
	}
	digest := sha256.Sum256(canonicalIntent)
	absoluteIndex, err := filepath.Abs(indexPath)
	if err != nil {
		return nil, err
	}
	absoluteIndex = filepath.Clean(absoluteIndex)
	marker := gateGitLockMarker{
		Version: artifact.InitialDocumentVersion, Authority: "overgo-gate-git-lock/v1",
		Preparation: intent.Preparation.String(), KeepaliveReference: intent.KeepaliveRef,
		KeepaliveCommit: intent.KeepaliveCommit, WorktreeGitDir: filepath.Dir(absoluteIndex),
		IndexPath: absoluteIndex, IntentSHA256: hex.EncodeToString(digest[:]),
	}
	return json.Marshal(marker)
}

func (g *gateContext) resolveAttemptStrategy(reader artifact.Reader) {
	declared := strings.TrimSpace(os.Getenv(loop.StrategyIDEnvironment))
	if declared == "" {
		return
	}
	id, err := artifact.ParseID(declared)
	if err != nil || id.Kind() != artifact.KindProfile || reader == nil {
		g.note("attempt strategy profile was declared but not resolvable; attempt remains comparison-ineligible")
		return
	}
	strategy, err := loop.RequireStrategy(context.Background(), reader, id)
	if err != nil {
		g.note("attempt strategy profile was declared but not resolvable; attempt remains comparison-ineligible")
		return
	}
	g.strategy = &strategy
}

// observeDiff reads the worktree's change size against HEAD; binary
// rows count as files with no line observation, and a failed read
// reports zero counts rather than inventing any.
func observeDiff(repo string) runrecord.AttemptDiff {
	output, err := command(repo, "git", "diff", "--numstat", "HEAD")
	if err != nil {
		return runrecord.AttemptDiff{}
	}
	var diff runrecord.AttemptDiff
	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		insertionsText, remainder, hasDeletions := strings.Cut(line, "\t")
		deletionsText, _, hasPath := strings.Cut(remainder, "\t")
		if !hasDeletions || !hasPath {
			continue
		}
		diff.Files++
		if insertions, err := strconv.Atoi(insertionsText); err == nil {
			diff.Insertions += insertions
		}
		if deletions, err := strconv.Atoi(deletionsText); err == nil {
			diff.Deletions += deletions
		}
	}
	return diff
}

func (g *gateContext) printSummary(output io.Writer, outcome runrecord.Outcome, failure string) {
	var run, reused, skipped, inapplicable []string
	for _, step := range g.steps {
		switch step.Outcome {
		case runrecord.StepSkipped:
			skipped = append(skipped, step.Name)
		case runrecord.StepReused:
			reused = append(reused, step.Name)
		case runrecord.StepInapplicable:
			inapplicable = append(inapplicable, step.Name)
		default:
			run = append(run, step.Name)
		}
	}
	fmt.Fprintf(output, "GATE %s %.1fs | ran=%s | reused=%s | skipped=%s | inapplicable=%s\n", strings.ToUpper(string(outcome)), time.Since(g.start).Seconds(), strings.Join(run, ","), strings.Join(reused, ","), strings.Join(skipped, ","), strings.Join(inapplicable, ","))
	if failure != "" {
		fmt.Fprintf(output, "blocker: %s\n", failure)
	}
	for _, line := range formatPhaseWallTable(phaseWallTable(g.steps), time.Since(g.start)) {
		fmt.Fprintln(output, line)
	}
	for _, line := range compactAudit(g.audit) {
		fmt.Fprintln(output, line)
	}
}

func appendGateAdvisoryFinding(ctx context.Context, store *overgodb.Store, batch *artifact.Batch, owners, audit []string) error {
	evidence := slices.DeleteFunc(compactAudit(audit), func(line string) bool {
		return !strings.HasPrefix(line, "advisory: review:") && !strings.HasPrefix(line, "advisory: warning:") &&
			(!strings.HasPrefix(line, "advisory: consumer:") || strings.Contains(line, "candidates=;"))
	})
	if len(evidence) == 0 {
		return nil
	}
	if len(owners) == 0 {
		owners = []string{"repository"}
	}
	document, findingBatch, err := finding.NewTextBatch(finding.GateAdvisoriesTitle, finding.SeverityMedium, owners, evidence,
		"Resolve each advisory at its owning source and retain a failable regression check.", "The gate emits no actionable advisory for the same owner surface.")
	if err != nil {
		return err
	}
	alias := artifact.AliasBinding{Name: finding.GateAdvisoriesAlias, Target: document.ID}
	if previous, found, err := artifact.ResolveAlias(ctx, store, alias.Name); err != nil {
		return err
	} else if found {
		alias.Previous = &previous
	}
	batch.Contents = append(batch.Contents, findingBatch.Contents...)
	batch.Lineage = append(batch.Lineage, findingBatch.Lineage...)
	batch.Aliases = append(batch.Aliases, alias)
	return nil
}

func compactAudit(lines []string) []string {
	var output []string
	for _, line := range lines {
		// Keep the costliest group's typed explanation parseable. It contains
		// input paths and reasons, never copied source or test output.
		if strings.HasPrefix(line, "test input attribution: ") {
			output = append(output, "advisory: dependency: "+line)
			continue
		}
		label := ""
		switch {
		case strings.HasPrefix(line, "suite cost ranking:"):
			label = "advisory: suite-cost: "
		case strings.Contains(line, "code profile delta vs HEAD"):
			label = "advisory: delta: "
		case strings.HasPrefix(line, "automation ROI"):
			label = "advisory: roi: "
		case strings.HasPrefix(line, "consumer census"):
			label = "advisory: consumer: "
		case strings.HasPrefix(line, "impact selection:"):
			label = "advisory: impact: "
		case strings.HasPrefix(line, "test scope:"):
			label = "advisory: scope: "
		case strings.HasPrefix(line, "package test evidence:"):
			label = "advisory: reuse: "
		case strings.Contains(line, " reused:"):
			label = "advisory: reuse: "
		case strings.Contains(line, "exact_clone=") && !strings.Contains(line, "exact_clone=none"):
			label = "advisory: review: "
		case strings.Contains(line, "uncatalogued") || strings.Contains(line, "unplanned dirty") ||
			strings.Contains(line, "unavailable") || strings.Contains(line, "unreadable") || strings.Contains(line, "not persisted"):
			label = "advisory: warning: "
		}
		if label == "" {
			continue
		}
		line = label + line
		if len(line) > 600 {
			line = line[:600] + "..."
		}
		output = append(output, line)
	}
	return output
}

func command(dir, name string, args ...string) (string, error) {
	return commandEnvironment(dir, nil, name, args...)
}

func gitWriterCommand(dir string, args ...string) (string, error) {
	return gitWriterCommandEnvironment(dir, nil, args...)
}

func gitWriterCommandEnvironment(dir string, environment []string, args ...string) (string, error) {
	cmd := newGateGitWriterCommand(dir, environment, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf(
			"git %s: %v: %s",
			strings.Join(args, " "),
			err,
			clioptions.Tail(string(out), clioptions.DiagnosticTailBytes),
		)
	}
	return string(out), nil
}

func newGateGitWriterCommand(dir string, environment []string, args ...string) *exec.Cmd {
	cmd := exec.Command("git", gitauthority.WriterArguments(args...)...)
	cmd.Dir = dir
	cmd.Env = gateGitEnvironment(gitauthority.RepositoryEnvironment(), environment)
	return cmd
}

func newGateGitReaderCommand(dir string, args ...string) *exec.Cmd {
	cmd := exec.Command("git", append([]string{"--no-replace-objects"}, args...)...)
	cmd.Dir = dir
	cmd.Env = gitauthority.ReaderEnvironment()
	return cmd
}

func commandEnvironment(dir string, environment []string, name string, args ...string) (string, error) {
	commandArgs := args
	if name == "git" {
		commandArgs = append([]string{"--no-replace-objects"}, args...)
	}
	var out []byte
	var err error
	// A child that Windows could not start (its loader failed under the
	// parallel test load) ran nothing; starting it again is a retry of the
	// launch, not of the work, and the bound keeps a real fault visible.
	for range processStartAttempts {
		cmd := exec.Command(name, commandArgs...)
		cmd.Dir = dir
		if name == "git" {
			cmd.Env = gateGitEnvironment(gitauthority.ReaderEnvironment(), environment)
		} else if environment != nil {
			cmd.Env = environment
		}
		out, err = cmd.CombinedOutput()
		if !processStartFailed(err) {
			break
		}
		time.Sleep(processStartRetryDelay)
	}
	if err != nil {
		return string(out), fmt.Errorf(
			"%s %s: %v: %s",
			name,
			strings.Join(args, " "),
			err,
			clioptions.Tail(string(out), clioptions.DiagnosticTailBytes),
		)
	}
	return string(out), nil
}

// Windows reports STATUS_DLL_INIT_FAILED when a process could not
// initialise; under the gate's parallel test load a nested go test
// occasionally ends this way before running anything.
const (
	windowsProcessStartFailure = 0xc0000142
	processStartAttempts       = 3
	processStartRetryDelay     = 2 * time.Second
)

// processStartFailed reports a child that Windows failed to start.
func processStartFailed(err error) bool {
	exit, ok := errors.AsType[*exec.ExitError](err)
	return ok && exit.ExitCode() == windowsProcessStartFailure
}

func gateGitEnvironment(base, environment []string) []string {
	gitEnvironment := slices.Clone(base)
	for _, entry := range environment {
		key, _, _ := strings.Cut(entry, "=")
		if strings.EqualFold(key, "GIT_INDEX_FILE") {
			gitEnvironment = append(gitEnvironment, entry)
		}
	}
	return gitEnvironment
}

func gitIndexEnvironment(value string) []string {
	return []string{"GIT_INDEX_FILE=" + value}
}

func gitLines(dir string, args ...string) ([]string, error) {
	out, err := command(dir, "git", args...)
	if err != nil {
		return nil, err
	}
	var lines []string
	for _, line := range strings.Split(out, "\n") {
		if line = strings.TrimRight(line, "\r"); strings.TrimSpace(line) != "" {
			lines = append(lines, line)
		}
	}
	return lines, nil
}

func readJSON(repo, name string, value any) error {
	return jsonfile.DecodeStrict(filepath.Join(repo, filepath.FromSlash(name)), value)
}

func writeJSON(repo, name string, value any, mode os.FileMode) error {
	return jsonfile.Write(filepath.Join(repo, filepath.FromSlash(name)), value, mode)
}

func printJSON(value any) error {
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(value)
}

// StorePath is the default canonical OvergoDB store directory the gate
// operates over; cmd/gate surfaces it as the -store default.
const StorePath = gateStorePath
