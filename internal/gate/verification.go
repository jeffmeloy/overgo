package gate

import (
	"cmp"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/clioptions"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/codeprofile"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/planverify"
	"overgo/internal/processcontrol"
	"overgo/internal/protection"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
)

func (g *gateContext) pipelineChecks(devicePackages ...string) []automationcheck.Check {
	generated := automationcheck.GeneratedChecks(g.sourceRoot(), g.runGateCommand)
	device := automationcheck.DeviceCheck(g.sourceRoot(), g.paths, devicePackages, g.runLaneCommand)
	published := automationcheck.PublishedCheck(g.sourceRoot(), g.runGateCommand)
	webui := automationcheck.WebUICheck(g.sourceRoot(), g.runLaneCommand)
	// The store writer among the static checks (the magics phase may rebind
	// the closure ledger) declares the store exclusively; the store readers
	// declare it shared, so the writer never overlaps a reader's replay.
	storeWriter := []automationcheck.Resource{{Name: "store", Exclusive: true}}
	storeReader := []automationcheck.Resource{{Name: "store"}}
	published.Descriptor.Resources = storeReader
	checks := []automationcheck.Check{
		gateCheck("protection", runrecord.PhaseValidate, g.stepProtection), gateCheck("scope", runrecord.PhaseValidate, g.stepScope),
		withResources(gateCheck("magics", runrecord.PhaseValidate, g.stepMagics), storeWriter),
		gateCheck("modern-go", runrecord.PhaseValidate, g.stepModernGoRatchet),
		gateCheck("architecture", runrecord.PhaseValidate, g.stepArchitectureRatchet),
		gateCheck("profile", runrecord.PhaseValidate, g.stepProfile), gateCheck("fmt", runrecord.PhaseValidate, g.stepFmt),
		gateCheck("style", runrecord.PhaseValidate, g.stepStyle), generated[0], generated[1], generated[2],
		withResources(gateCheck("docs", runrecord.PhaseValidate, g.stepDocumentation), storeReader), published,
		gateCheck("vet", runrecord.PhaseVet, g.stepVet), gateCheck("build", runrecord.PhaseBuild, g.stepBuild),
		gateCheck("acceptance", runrecord.PhaseTest, g.stepAcceptance), {
			// The changed source owners run first as their own check; the
			// remaining groups' device packages follow, then the lanes and
			// the host packages together.
			Descriptor: automationcheck.Descriptor{Name: "test-owners", Phase: runrecord.PhaseTest, Always: true},
			Run: func(ctx context.Context, _ automationcheck.Invocation) (bool, string, error) {
				skipped, err := g.stepTestOwners(ctx)
				return skipped, "", err
			},
		}, {
			// Dependent device-capable tests share admission. Opted-in
			// measurements remain outside this host-test batch.
			Descriptor: automationcheck.Descriptor{Name: "test-device", Phase: runrecord.PhaseTest, Always: true},
			Run: func(ctx context.Context, _ automationcheck.Invocation) (bool, string, error) {
				skipped, err := g.stepTestDevice(ctx)
				return skipped, "", err
			},
		}, {
			Descriptor: automationcheck.Descriptor{Name: "test", Phase: runrecord.PhaseTest, Always: true},
			Run: func(ctx context.Context, _ automationcheck.Invocation) (bool, string, error) {
				skipped, err := g.stepTestRest(ctx)
				return skipped, "", err
			},
		},
		device, webui, gateCheck("commit", runrecord.PhasePackage, g.stepCommit),
	}
	// Protection and scope admit the candidate first: an unplanned
	// verification input invalidates every later verdict. The static checks
	// carry no data dependency on one another and run in one wave after
	// scope; vet and build follow the whole wave as master ordered them.
	dependencies := map[string][]string{"scope": {"protection"}}
	for _, name := range validateWave {
		dependencies[name] = []string{"scope"}
	}
	dependencies["vet"] = slices.Clone(validateWave)
	dependencies["build"] = slices.Clone(validateWave)
	dependencies["acceptance"] = []string{"vet", "build"}
	// Changed owners precede shared dependency tests and browser correctness.
	// The device lane and remaining host tests follow the dependency tests.
	// Shared device admission and waiting store locks protect overlap.
	dependencies["test-owners"] = []string{"acceptance"}
	dependencies["test-device"] = []string{"test-owners"}
	dependencies["test"] = []string{"test-device"}
	dependencies["device"] = []string{"test-device"}
	dependencies[automationcheck.WebUICheckName] = []string{"test-owners"}
	dependencies["commit"] = []string{"test", "device", automationcheck.WebUICheckName}
	for index := range checks {
		checks[index].Descriptor.Dependencies = dependencies[checks[index].Descriptor.Name]
	}
	return checks
}

// validateWave lists the static checks that share no data and run together
// after the admission checks, in the order the wave declares them: the
// store writer first, so the store readers follow it in the next wave.
var validateWave = []string{"magics", "modern-go", "architecture", "profile", "fmt", "style", "manifest", "sbom", "claims", "docs", "published"}

func withResources(check automationcheck.Check, resources []automationcheck.Resource) automationcheck.Check {
	check.Descriptor.Resources = resources
	return check
}

// joins every failed check of the wave that stopped the pipeline, so one
// attempt reports all its findings
func checkFailures(results []automationcheck.DAGResult) error {
	var failures []error
	for _, result := range results {
		if result.Err != nil {
			failures = append(failures, fmt.Errorf("%s: %w", result.Invocation.Check.Name, result.Err))
		}
	}
	return errors.Join(failures...)
}

func gateCheck(name string, phase runrecord.Phase, run func() (bool, error)) automationcheck.Check {
	return automationcheck.Check{
		Descriptor: automationcheck.Descriptor{Name: name, Phase: phase, Always: true},
		Run: func(context.Context, automationcheck.Invocation) (bool, string, error) {
			skipped, err := run()
			return skipped, "", err
		},
	}
}

func (g *gateContext) pipeline() error {
	if g.planProjection == plan.MergeProjectionFirstParentTarget {
		if err := reportGateAdmissionPhase("preflight projected merge completion", g.preflightProjectedMergeCompletion); err != nil {
			return err
		}
	}
	tree, err := g.plannedTree()
	if err != nil {
		return err
	}
	return g.withCandidateWorktree(tree, func(string) error {
		planningStarted := time.Now()
		planned, err := g.planPipeline()
		if err != nil {
			return err
		}
		definitions, checks, impact := planned.definitions, planned.invocations, planned.impact
		planningDuration := time.Since(planningStarted)
		if g.terminal == nil {
			g.terminal = map[string]automationcheck.Evidence{}
		}
		cache := g.loadRetryCache()
		cache.Compact()
		g.retryCache = &cache
		inputs := make(map[artifact.ID]artifact.ID, len(checks))
		for index, check := range checks {
			input, inputErr := g.phaseInputFingerprint(check.Check.Name)
			if inputErr != nil {
				return inputErr
			}
			if planned.manifest != nil {
				bound, bindErr := automationcheck.BindManifestExecution(*planned.manifest, check, []artifact.ID{input})
				if bindErr != nil {
					return bindErr
				}
				checks[index] = bound
				check = bound
			}
			inputs[check.ID] = input
		}
		satisfied := make(map[string]bool, len(impact.Exclusions))
		for _, exclusion := range impact.Exclusions {
			satisfied[exclusion.Check] = true
		}
		var drift func() error
		if planned.manifest != nil {
			tree := planned.manifest.CandidateTree
			drift = func() error { return g.requireCandidateTree(tree) }
		}
		results, err := g.executeChecks(checks, satisfied, inputs, &cache, drift)
		if err != nil {
			return err
		}
		if saveErr := g.saveRetryCache(cache); saveErr != nil {
			g.note("retry cache not saved: " + saveErr.Error())
		}
		byName := make(map[string]automationcheck.DAGResult, len(results))
		for _, result := range results {
			if result.Invocation.ID.Valid() {
				byName[result.Invocation.Check.Name] = result
			}
		}
		cacheHits := 0
		cacheEligible := 0
		for _, result := range results {
			if _, _, eligible := g.checkCacheKey(result.Invocation, inputs); eligible {
				cacheEligible++
			}
			if result.Evidence.Reused {
				cacheHits++
			}
		}
		g.manifestMetrics = automationcheck.MeasureManifest(
			len(definitions), len(checks), len(impact.Exclusions), len(planned.surface.Unknown),
			cacheEligible, cacheHits, planningDuration, time.Since(g.start),
		)
		for _, definition := range definitions {
			name := definition.Descriptor.Name
			result, ran := byName[name]
			if !ran {
				if exclusion, excluded := impact.ExclusionReason(name); excluded {
					g.steps = append(g.steps, runrecord.GateStep{Name: name, Phase: definition.Descriptor.Phase, Outcome: runrecord.StepSkipped, DurationNS: uint64(time.Nanosecond)})
					g.note(name + " skipped: " + exclusion)
				}
				continue
			}
			if planned.manifest != nil && result.Evidence.ID.Valid() &&
				!slices.Contains(automationcheck.EvidenceLineage(result.Evidence), planned.manifest.ID) {
				return fmt.Errorf("%s: terminal evidence omits manifest plan %s", name, planned.manifest.ID)
			}
			g.terminal[name] = result.Evidence
			g.steps = append(g.steps, gateEvidenceRecord(name, result.Invocation.Check.Phase, result.Evidence, result.Err, g.stepEvidence[name]))
			if result.Evidence.Reused {
				g.note(name + " reused: derived inputs already passed this step")
			}
		}
		// Every failure of the wave that stopped the pipeline is reported at
		// once, so one attempt shows all its findings.
		return checkFailures(results)
	})
}

func ownershipSurface(impact codeprofile.FunctionImpact) automationcheck.Surface {
	packages := map[string]bool{}
	for _, packagePath := range impact.Packages {
		packages[packagePath] = true
	}
	symbols := make([]automationcheck.Symbol, 0, len(impact.Reachable))
	for _, symbol := range impact.Reachable {
		packagePath := path.Dir(symbol.File)
		packages[packagePath] = true
		symbols = append(symbols, automationcheck.Symbol{
			Package: packagePath, Receiver: symbol.Receiver, Name: symbol.Name,
		})
	}
	unknown := make([]string, 0, len(impact.Unknown))
	for _, boundary := range impact.Unknown {
		unknown = append(unknown, strings.Join([]string{boundary.Kind, boundary.Path, boundary.Symbol}, ":"))
	}
	return automationcheck.Surface{
		Identity: impact.BaseIdentity + ":" + impact.CandidateIdentity,
		Packages: slices.Sorted(maps.Keys(packages)), Symbols: symbols, Unknown: unknown,
	}
}

func gateEvidenceRecord(name string, phase runrecord.Phase, evidence automationcheck.Evidence, runErr error, storedEvidence string) runrecord.GateStep {
	record := runrecord.GateStep{
		Name: name, Phase: phase, Outcome: runrecord.StepSucceeded,
		DurationNS: max(evidence.DurationNS, uint64(1)), Evidence: storedEvidence,
	}
	if record.Evidence == "" && evidence.ID.Valid() {
		record.Evidence = evidence.ID.String()
	}
	switch {
	case runErr != nil:
		record.Outcome = runrecord.StepFailed
	case evidence.Inapplicable:
		record.Outcome = runrecord.StepInapplicable
	case evidence.Reused:
		record.Outcome = runrecord.StepReused
	}
	return record
}

// stepDocumentation keeps prose assessments from masquerading as current
// work authority. Ranked current work belongs to the failable plan; prose may
// retain its historical ordering only with an explicit warning and redirect.
func (g *gateContext) stepDocumentation() (bool, error) {
	if err := repoanalysis.ValidateDocsInventory(g.repo); err != nil {
		return false, err
	}
	if slices.Contains(g.paths, plan.Path) || g.pathsTouchAny("internal/plan/") {
		tests := []string{"TestSingleCanonicalCampaignPlan", "TestRSICampaignRatchetAndParallelStructure", "TestOptimizedValidationCampaign"}
		output, err := g.runGateCommand(g.sourceRoot(), "go", "test", "./internal/plan", "-json", "-run", "^("+strings.Join(tests, "|")+")$", "-count=1")
		if err != nil {
			return false, fmt.Errorf("campaign structure: %w\n%s", err, output)
		}
		for _, name := range tests {
			if err := testevidence.VerifyGoTestEvidence("go test ./internal/plan -run '^"+name+"$'", output); err != nil {
				return false, fmt.Errorf("campaign structure: %w", err)
			}
		}
		g.note("campaign structure: required checks passed before long verification")
	}
	document, err := g.loadPlan()
	if err != nil {
		return false, err
	}
	if err := plan.ValidateCampaignCensusAuthority(document); err != nil {
		return false, err
	}
	store, err := overgodb.OpenReadOnly(filepath.Join(g.repo, g.storePath))
	if err != nil {
		return false, err
	}
	_, found, readErr := closurescan.ReadCensusEvidence(context.Background(), store, *document.Census)
	closeErr := store.Close()
	if readErr != nil || closeErr != nil {
		return false, errors.Join(readErr, closeErr)
	}
	if !found {
		return false, errors.New("gate: campaign census evidence is absent from OvergoDB")
	}
	return false, documentationFreshness(g.repo)
}

func documentationFreshness(root string) error {
	return filepath.WalkDir(filepath.Join(root, "docs"), func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".md") {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		text := string(data)
		if !strings.Contains(text, "## Execution Program") {
			return nil
		}
		header := text
		if len(header) > 1024 {
			header = header[:1024]
		}
		if !strings.Contains(header, historicalRankingMarker) || !strings.Contains(header, currentWorkMarker) {
			return fmt.Errorf("documentation ranking %s lacks a prominent historical marker and docs/plan.json redirect", filepath.ToSlash(path))
		}
		return nil
	})
}

func (g *gateContext) sourceSnapshot() (repoanalysis.SourceSnapshot, error) {
	g.sourceMutex.Lock()
	defer g.sourceMutex.Unlock()
	if g.source != nil {
		return *g.source, nil
	}
	snapshot, err := repoanalysis.DiscoverGo(g.sourceRoot(), "internal", "cmd")
	if err == nil {
		g.source = &snapshot
	}
	return snapshot, err
}

// returns the HEAD source snapshot, computed once and shared by every check
// of the validate wave that compares the candidate against HEAD
func (g *gateContext) baseSnapshot(candidate repoanalysis.SourceSnapshot) (repoanalysis.SourceSnapshot, error) {
	g.sourceMutex.Lock()
	defer g.sourceMutex.Unlock()
	if g.baseSource != nil {
		return *g.baseSource, nil
	}
	base, err := sourceAtHEAD(g.repo, candidate)
	if err == nil {
		g.baseSource = &base
	}
	return base, err
}

// stepArchitectureRatchet audits the complete candidate source on every gate
// run, including documentation-only changes. It is deliberately in-process:
// the manifest already binds this check to the candidate tree, and a second
// test process would only repeat source discovery while weakening ordering.
func (g *gateContext) stepArchitectureRatchet() (bool, error) {
	if _, err := g.stepArchitecture(); err != nil {
		return false, err
	}
	staged, err := codeprofile.LoadStagedSurface(filepath.Join(g.repo, "docs", "staged_surface.json"))
	if err != nil {
		return false, err
	}
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	report, err := repoanalysis.AuditProductionAuthorityBoundaries(snapshot)
	if err != nil {
		return false, err
	}
	budgets, err := repoanalysis.LoadStructureBudgets(filepath.Join(g.repo, filepath.FromSlash(repoanalysis.StructureBudgetsFile)))
	if err != nil {
		return false, err
	}
	var violations []error
	for _, finding := range repoanalysis.MeasureStructureBudgets(snapshot, budgets) {
		violations = append(violations, fmt.Errorf("architecture: %s %s = %d over limit %d", finding.Budget, finding.Subject, finding.Value, finding.Limit))
	}
	if err := errors.Join(violations...); err != nil {
		return false, err
	}
	if err := runrecord.ValidateTriggerRegistry(); err != nil {
		return false, err
	}
	// The staged-surface declaration is gate authority even when no Go file
	// changed: every deferred export must still resolve to an open canonical
	// step or an evidence-bound retained classification.
	report.Rules += len(staged.Staged) + 1
	report.Sites += len(staged.Staged)
	g.note(fmt.Sprintf(
		"architecture ratchet: source=%s paths=%d rules=%d sites=%d findings=%d",
		report.SourceIdentity, len(g.paths), report.Rules, report.Sites, len(report.Findings),
	))
	return false, report.Error()
}

func (g *gateContext) stepProfile() (bool, error) {
	changed := g.plannedGoFiles()
	if len(changed) == 0 {
		return true, nil
	}
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	profile, err := codeprofile.Build(snapshot)
	if err != nil {
		return false, err
	}
	profile.Impact = codeprofile.ImpactSelection{
		Identity: g.selectionID, Owned: g.selection.Owned, Triggered: g.selection.Triggered,
		Excluded: g.selection.Excluded, Unresolved: g.selection.Unresolved,
	}
	baseSource, err := g.baseSnapshot(snapshot)
	if err != nil {
		return false, err
	}
	base, err := codeprofile.Build(baseSource)
	if err != nil {
		return false, err
	}
	g.note(fmt.Sprintf(
		"code profile: runtime=%d files/%d nodes automation=%d/%d generated=%d/%d test=%d/%d duplicate_excess=%d clones=%d functions=%d exported=%d imports=%d",
		profile.Runtime.Files, profile.Runtime.Nodes, profile.Automation.Files, profile.Automation.Nodes,
		profile.Generated.Files, profile.Generated.Nodes, profile.Test.Files, profile.Test.Nodes, profile.DuplicateExcessNodes,
		len(profile.Clones), len(profile.Functions), profile.ExportedDeclarations, profile.PackageImportEdges,
	))
	g.note(surfaceDeltaAudit(base, profile))
	baseline, ratcheted, err := codeprofile.LoadCloneBaseline(filepath.Join(g.repo, filepath.FromSlash(codeprofile.CloneBaselineFile)))
	if err != nil {
		return false, err
	}
	if ratcheted {
		if err := codeprofile.AdmitCloneBaseline(baseline, profile.DuplicateExcessNodes); err != nil {
			return false, err
		}
		g.note(fmt.Sprintf(
			"clone ratchet: duplicate_excess=%d ceiling=%d headroom=%d",
			profile.DuplicateExcessNodes, baseline.DuplicateExcessNodes,
			baseline.DuplicateExcessNodes-profile.DuplicateExcessNodes,
		))
	}
	if g.automationPlan() {
		movement, err := codeprofile.MeasureProductionMovement(baseSource, snapshot)
		if err != nil {
			return false, err
		}
		summary, err := automationROIAdmission("commit", movement)
		g.note(summary)
		if err != nil {
			return false, err
		}
	}
	g.note(profileReviewFocus(profile, g.changedGoFiles()))
	if err := g.appendConsumerCensus(snapshot, baseSource, changed, &profile); err != nil {
		return false, err
	}
	g.profile = &profile
	return false, nil
}

func (g *gateContext) appendConsumerCensus(candidate, head repoanalysis.SourceSnapshot, changed []string, profile *codeprofile.Profile) error {
	selection, err := repoanalysis.HostBuildSelection(g.sourceRoot(), "./cmd/...", "./internal/...")
	if err != nil {
		return err
	}
	impact := codeprofile.FunctionImpact{}
	if g.structural != nil {
		impact = *g.structural
	} else {
		impact, err = codeprofile.DeriveFunctionImpact(head, candidate, selection, selection, g.paths)
		if err != nil {
			return err
		}
	}
	g.note(fmt.Sprintf(
		"function impact: base=%s candidate=%s seeds=%d reachable=%d unknown=%d",
		impact.BaseIdentity, impact.CandidateIdentity, len(impact.Seeds), len(impact.Reachable), len(impact.Unknown),
	))
	g.note(impactSelectionAudit(profile.Impact))
	if _, profile.Consumers, err = codeprofile.ProductionConsumerCensus(candidate, selection, nil); err != nil {
		return err
	}
	paths := pathSet(changed)
	declarations, current, err := codeprofile.ProductionConsumerCensus(candidate, selection, paths)
	if err != nil {
		return err
	}
	baseDeclarations, base, err := codeprofile.ProductionConsumerCensus(head, selection, paths)
	if err != nil {
		return err
	}
	g.note(consumerCensusAudit("commit", selection.Context, declarations, base, current))
	if unconsumed := codeprofile.NewUnconsumedSurface(baseDeclarations, declarations); len(unconsumed) > 0 {
		// docs/staged_surface.json is the reviewed acceptance for new
		// surface whose consumer is deliberately deferred (owner ruling
		// 2026-08-27: valuable new elements land declared, not blocked);
		// undeclared new surface still refuses.
		staged, err := codeprofile.LoadStagedSurface(filepath.Join(g.repo, "docs", "staged_surface.json"))
		if err != nil {
			return err
		}
		accepted, blocking := codeprofile.PartitionStagedSurface(unconsumed, staged)
		if len(accepted) > 0 {
			g.note(fmt.Sprintf(
				"staged surface accepted per docs/staged_surface.json: %s", consumerCandidates(accepted)))
		}
		if len(blocking) > 0 {
			return fmt.Errorf("new unconsumed production surface: %s", consumerCandidates(blocking))
		}
	}

	mergeBase, err := command(g.repo, "git", "merge-base", "master", "HEAD")
	if err != nil {
		g.note("consumer census plan-slice unavailable: " + err.Error())
		return nil
	}
	mergeBase = strings.TrimSpace(mergeBase)
	slicePaths, err := changedGoPathsAtRevision(g.repo, mergeBase, changed)
	if err != nil {
		return err
	}
	sliceBase, err := sourceAtRevision(g.repo, candidate, mergeBase, slicePaths)
	if err != nil {
		return err
	}
	if g.automationPlan() {
		movement, err := codeprofile.MeasureProductionMovement(sliceBase, candidate)
		if err != nil {
			return err
		}
		summary, err := automationROIAdmission("plan-slice@"+mergeBase[:12], movement)
		g.note(summary)
		if err != nil {
			return err
		}
	}
	paths = pathSet(slicePaths)
	declarations, current, err = codeprofile.ProductionConsumerCensus(candidate, selection, paths)
	if err != nil {
		return err
	}
	_, base, err = codeprofile.ProductionConsumerCensus(sliceBase, selection, paths)
	if err != nil {
		return err
	}
	g.note(consumerCensusAudit("plan-slice@"+mergeBase[:12], selection.Context, declarations, base, current))
	return nil
}

func impactSelectionAudit(selection codeprofile.ImpactSelection) string {
	return fmt.Sprintf(
		"impact selection: excluded=%d/%d triggered=%d unresolved=%d snapshot=%s",
		selection.Excluded, selection.Owned, selection.Triggered, selection.Unresolved, selection.Identity,
	)
}

func (g *gateContext) automationPlan() bool {
	item, _, _ := strings.Cut(g.planRef, "/")
	return strings.HasPrefix(item, "automation-")
}

func automationROIAdmission(scope string, movement codeprofile.ProductionMovement) (string, error) {
	summary := fmt.Sprintf(
		"automation ROI %s: production_ast=%d-%d net=%+d go_lines=%d-%d net=%+d; admission=net-negative",
		scope, movement.Added, movement.Deleted, movement.Added-movement.Deleted,
		movement.GoLinesAdded, movement.GoLinesDeleted, movement.GoLinesAdded-movement.GoLinesDeleted,
	)
	if movement.Added > 0 && movement.Added >= movement.Deleted ||
		movement.GoLinesAdded > 0 && movement.GoLinesAdded >= movement.GoLinesDeleted {
		return summary, fmt.Errorf("automation surface is not net-negative: %s", summary)
	}
	return summary, nil
}

func consumerCensusAudit(scope, context string, declarations []codeprofile.ConsumerDeclaration, base, current codeprofile.ConsumerSummary) string {
	declarations = slices.DeleteFunc(slices.Clone(declarations), func(value codeprofile.ConsumerDeclaration) bool {
		return value.ProductionReferences > 0 || value.Boundary != ""
	})
	return fmt.Sprintf(
		"consumer census %s context=%s delta: production=%+d test_only=%+d boundary=%+d zero=%+d; current=%d/%d/%d/%d; candidates=%s; advisory_only=ambiguous dispatch is a boundary, tests are not production consumers",
		scope, context, current.Production-base.Production, current.TestOnly-base.TestOnly,
		current.Boundary-base.Boundary, current.Zero-base.Zero,
		current.Production, current.TestOnly, current.Boundary, current.Zero, consumerCandidates(declarations),
	)
}

func consumerCandidates(declarations []codeprofile.ConsumerDeclaration) string {
	candidates := make([]string, len(declarations))
	for index, declaration := range declarations {
		class := "zero"
		if declaration.TestReferences > 0 {
			class = "test-only"
		}
		candidates[index] = consumerCandidate(declaration, class)
	}
	slices.Sort(candidates)
	return strings.Join(candidates, ",")
}

func consumerCandidate(declaration codeprofile.ConsumerDeclaration, class string) string {
	return declaration.File + ":" + declaration.Name + "=" + class
}

func changedGoPathsAtRevision(repo, revision string, pending []string) ([]string, error) {
	raw, err := command(repo, "git", "diff", "--no-renames", "--name-only", "-z", revision, "--", "cmd", "internal")
	if err != nil {
		return nil, err
	}
	paths := pathSet(pending)
	for _, name := range strings.Split(raw, "\x00") {
		if strings.HasSuffix(name, ".go") {
			paths[filepath.ToSlash(name)] = true
		}
	}
	return slices.Sorted(maps.Keys(paths)), nil
}

func sourceAtRevision(repo string, candidate repoanalysis.SourceSnapshot, revision string, paths []string) (repoanalysis.SourceSnapshot, error) {
	overlay := make(map[string][]byte, len(paths))
	for _, path := range paths {
		cmd := newGateGitReaderCommand(repo, "show", revision+":"+path)
		data, err := cmd.Output()
		if err != nil {
			if _, missing := err.(*exec.ExitError); missing {
				overlay[path] = nil
				continue
			}
			return repoanalysis.SourceSnapshot{}, err
		}
		overlay[path] = data
	}
	return candidate.Overlay(overlay)
}

func pathSet(paths []string) map[string]bool {
	set := make(map[string]bool, len(paths))
	for _, path := range paths {
		set[filepath.ToSlash(path)] = true
	}
	return set
}

func surfaceDeltaAudit(base, candidate codeprofile.Profile) string {
	runtimeFiles, runtimeNodes := candidate.Runtime.Files-base.Runtime.Files, candidate.Runtime.Nodes-base.Runtime.Nodes
	automationFiles, automationNodes := candidate.Automation.Files-base.Automation.Files, candidate.Automation.Nodes-base.Automation.Nodes
	duplicateExcess := candidate.DuplicateExcessNodes - base.DuplicateExcessNodes
	adverse := "none"
	if duplicateExcess < 0 && (runtimeFiles+automationFiles > 0 || runtimeNodes+automationNodes > 0) {
		adverse = "duplication fell while production grew; reduction does not offset surface growth"
	}
	return fmt.Sprintf(
		"code profile delta vs HEAD: runtime=%+d files/%+d nodes automation=%+d/%+d generated=%+d/%+d test=%+d/%+d duplicate_excess=%+d clones=%+d function_count=%+d exported=%+d imports=%+d; adverse_pattern=%s",
		runtimeFiles, runtimeNodes, automationFiles, automationNodes,
		candidate.Generated.Files-base.Generated.Files, candidate.Generated.Nodes-base.Generated.Nodes,
		candidate.Test.Files-base.Test.Files, candidate.Test.Nodes-base.Test.Nodes, duplicateExcess,
		len(candidate.Clones)-len(base.Clones), len(candidate.Functions)-len(base.Functions),
		candidate.ExportedDeclarations-base.ExportedDeclarations, candidate.PackageImportEdges-base.PackageImportEdges, adverse,
	)
}

func profileReviewFocus(profile codeprofile.Profile, changed []string) string {
	paths := pathSet(changed)
	cloneFocus := "none"
	for _, candidate := range profile.Clones {
		if !cliMainClone(candidate) && slices.ContainsFunc(candidate.Functions, func(function string) bool {
			path, _, ok := strings.Cut(function, ":")
			return ok && paths[path]
		}) {
			cloneFocus = fmt.Sprintf("nodes=%d functions=%s", candidate.Nodes, strings.Join(candidate.Functions, ","))
			break
		}
	}
	return fmt.Sprintf("code review candidate: exact_clone=%s; advisory_only=inspect semantic ownership and numerical contracts, migrate callers and delete displaced paths, require parity evidence", cloneFocus)
}

func cliMainClone(clone codeprofile.Clone) bool {
	return len(clone.Functions) > 1 && !slices.ContainsFunc(clone.Functions, func(function string) bool {
		return !strings.HasSuffix(function, ":main")
	})
}

func sourceAtHEAD(repo string, candidate repoanalysis.SourceSnapshot) (repoanalysis.SourceSnapshot, error) {
	raw, err := command(repo, "git", "status", "--porcelain=v1", "-z", "--untracked-files=all")
	if err != nil {
		return repoanalysis.SourceSnapshot{}, err
	}
	dirty, err := repoanalysis.ParseDirtyStatus([]byte(raw))
	if err != nil {
		return repoanalysis.SourceSnapshot{}, err
	}
	paths := map[string]bool{}
	for _, entry := range dirty {
		for _, path := range []string{entry.Path, entry.OriginalPath} {
			if !strings.HasSuffix(path, ".go") || !strings.HasPrefix(path, "internal/") && !strings.HasPrefix(path, "cmd/") {
				continue
			}
			paths[path] = true
		}
	}
	return sourceAtRevision(repo, candidate, "HEAD", slices.Sorted(maps.Keys(paths)))
}

func (g *gateContext) loadRetryCache() automationcheck.EvidenceCache {
	empty := automationcheck.NewEvidenceCache(g.environment.ID)
	var cache automationcheck.EvidenceCache
	if readJSON(g.repo, gateRetryFile, &cache) != nil || !cache.Reusable(g.environment.ID) {
		return empty
	}
	return cache
}

// saveRetryCache writes the cache atomically; the tmp directory is created so
// a fresh candidate tree persists too; an unbound environment writes nothing.
func (g *gateContext) saveRetryCache(cache automationcheck.EvidenceCache) error {
	if !cache.Environment.Valid() {
		return nil
	}
	if err := os.MkdirAll(filepath.Join(g.repo, filepath.Dir(filepath.FromSlash(gateRetryFile))), gatePrivateDirectoryMode); err != nil {
		return err
	}
	return writeJSON(g.repo, gateRetryFile, cache, clioptions.OutputFileMode)
}

func (g *gateContext) phaseInputFingerprint(phase string) (artifact.ID, error) {
	paths := g.cachePaths
	var err error
	if paths == nil {
		paths, err = gitLines(g.repo, "ls-files", "-co", "--exclude-standard")
		if err != nil {
			return artifact.ID{}, err
		}
		g.cachePaths = paths
	}
	return fingerprintPhaseInputs(g.repo, phase, paths)
}

func fingerprintPhaseInputs(root, phase string, paths []string) (artifact.ID, error) {
	var selected []string
	for _, path := range paths {
		path = filepath.ToSlash(path)
		if phaseOwnsPath(phase, path) {
			selected = append(selected, path)
		}
	}
	slices.Sort(selected)
	hasher := sha256.New()
	hasher.Write([]byte(phase + "\x00"))
	for _, path := range selected {
		hasher.Write([]byte(path + "\x00"))
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
		if errors.Is(err, os.ErrNotExist) {
			hasher.Write([]byte("<deleted>\x00"))
			continue
		}
		if err != nil {
			return artifact.ID{}, err
		}
		hasher.Write(data)
		hasher.Write([]byte{0})
	}
	return artifact.IdentifyBytes(artifact.KindEvidence, hasher.Sum(nil))
}

func phaseOwnsPath(phase, path string) bool {
	goSource := path == "go.mod" || path == "go.sum" || strings.HasSuffix(path, ".go")
	goInput := goSource ||
		(strings.HasPrefix(path, "internal/") || strings.HasPrefix(path, "cmd/")) && !strings.HasSuffix(path, ".md")
	documentation := strings.HasSuffix(path, ".md") || strings.HasPrefix(path, "docs/") ||
		path == "compatibility.json" || path == "SBOM.cdx.json"
	switch phase {
	case "vet", "build", "fmt", "style", "profile", "architecture":
		return goInput
	case "scope", "protection":
		return goInput || strings.HasPrefix(path, "scripts/") || strings.HasPrefix(path, protection.HarnessConfigDirectory)
	case "manifest", "sbom", "claims", "docs":
		return goInput || documentation
	case "test", "test-owners", "test-device":
		return goInput || path == "README.md" || strings.HasPrefix(path, "docs/") && path != plan.Path
	default:
		return false
	}
}

// phaseReusesEvidence reports whether a check's terminal evidence may be
// reused across consecutive gate attempts when its exact manifest-bound input
// fingerprint is byte-identical. Store-coupled checks (magics, acceptance,
// published) stay excluded because every attempt's preparation commit moves
// the store their verdicts read; device stays excluded because hardware is
// not a fingerprintable input; the commit check is never reused.
func phaseReusesEvidence(phase string) bool {
	switch phase {
	case "vet", "build", "fmt", "style", "profile", "architecture", "scope", "protection",
		"manifest", "sbom", "claims", "docs", "test", "test-owners", "test-device":
		return true
	default:
		return false
	}
}

func (g *gateContext) changedGoFiles() []string {
	var out []string
	for _, p := range g.paths {
		if !strings.HasSuffix(p, ".go") {
			continue
		}
		// Skip planned .go paths no longer on disk (a rename/delete staged in
		// this commit): gofmt cannot stat a removed file, and the scope step
		// already validated the deletion is tracked.
		if _, err := os.Stat(filepath.Join(g.repo, filepath.FromSlash(p))); err != nil {
			continue
		}
		out = append(out, p)
	}
	return out
}

func (g *gateContext) plannedGoFiles() []string {
	var out []string
	for _, path := range g.paths {
		if strings.HasSuffix(path, ".go") && (strings.HasPrefix(path, "cmd/") || strings.HasPrefix(path, "internal/")) {
			out = append(out, path)
		}
	}
	return out
}

// argBudget bounds the cumulative path-argument bytes per spawned
// process: Windows caps the CreateProcess command line at 32767
// characters, and a repo-wide commit can plan more paths than one
// invocation carries; 28000 leaves headroom for the executable path
// and the leading flags.
const argBudget = 28_000

// chunkByArgBudget splits files into runs whose joined length fits one
// command line under argBudget; every run is non-empty.
func chunkByArgBudget(files []string) [][]string {
	var chunks [][]string
	for start := 0; start < len(files); {
		end, used := start, 0
		for end < len(files) && (end == start || used+len(files[end])+1 <= argBudget) {
			used += len(files[end]) + 1
			end++
		}
		chunks = append(chunks, files[start:end])
		start = end
	}
	return chunks
}

func (g *gateContext) stepFmt() (bool, error) {
	files := g.changedGoFiles()
	if len(files) == 0 {
		return true, nil
	}
	var unformatted []string
	for _, chunk := range chunkByArgBudget(files) {
		out, err := command(g.repo, "gofmt", append([]string{"-l"}, chunk...)...)
		if err != nil {
			return false, err
		}
		if s := strings.TrimSpace(out); s != "" {
			unformatted = append(unformatted, s)
		}
	}
	if len(unformatted) > 0 {
		files := strings.Fields(strings.Join(unformatted, "\n"))
		return false, fmt.Errorf(
			"unformatted: %s; remediate with `gofmt -w %s`",
			strings.Join(files, " "), strings.Join(files, " "),
		)
	}
	return false, nil
}

func (g *gateContext) stepStyle() (bool, error) {
	if len(g.changedGoFiles()) == 0 {
		return true, nil
	}
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	baseline, err := g.baseSnapshot(snapshot)
	if err != nil {
		return false, err
	}
	return false, repoanalysis.ValidateGoStyleDelta(snapshot, baseline)
}

func (g *gateContext) stepModernGoRatchet() (bool, error) {
	baseline, err := repoanalysis.LoadModernGoBaseline(filepath.Join(g.sourceRoot(), filepath.FromSlash(repoanalysis.ModernGoBaselineFile)))
	if err != nil {
		return false, err
	}
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	selection, err := repoanalysis.HostBuildSelection(g.sourceRoot(), "./cmd/...", "./internal/...")
	if err != nil {
		return false, err
	}
	candidate, err := repoanalysis.ModernGoCensusSnapshot(snapshot, selection, baseline.TargetGo)
	if err != nil {
		return false, err
	}
	if baseline.SourceIdentity != candidate.SourceIdentity {
		return false, errors.New("modern-Go baseline source is stale; run `go run ./cmd/modern-census -lower-baseline` and `go run ./cmd/modern-census -publish-census`")
	}
	if err := repoanalysis.AdmitModernGoRatchet(baseline, candidate, time.Now().UTC()); err != nil {
		return false, err
	}
	if err := admitModernGoExactExceptions(baseline, candidate); err != nil {
		return false, err
	}
	expected, err := repoanalysis.BuildModernGoPublishedCensus(candidate, baseline)
	if err != nil {
		return false, err
	}
	var published repoanalysis.ModernGoPublishedCensus
	if err := jsonfile.DecodeStrict(filepath.Join(g.sourceRoot(), filepath.FromSlash(repoanalysis.ModernGoPublishedCensusFile)), &published); err != nil {
		return false, err
	}
	if err := repoanalysis.ValidateModernGoPublishedCensus(published, expected); err != nil {
		return false, err
	}
	if _, err := command(g.repo, "git", "cat-file", "-e", "HEAD:"+repoanalysis.ModernGoBaselineFile); err == nil {
		base, err := g.baseSnapshot(snapshot)
		if err != nil {
			return false, err
		}
		previous := candidate
		if base.Identity() != snapshot.Identity() {
			previous, err = repoanalysis.ModernGoCensusSnapshot(base, selection, baseline.TargetGo)
			if err != nil {
				return false, err
			}
		}
		if err := repoanalysis.AdmitModernGoDelta(baseline, previous, candidate, g.paths); err != nil {
			return false, err
		}
	}
	g.note(fmt.Sprintf(
		"modern-Go ratchet: findings=%d candidates=%d inspected=%d typed=%d catalog=%s",
		len(candidate.Findings), candidate.CandidateCount(), baseline.Coverage.InspectedFiles,
		baseline.Coverage.TypedFiles, baseline.CatalogCommit,
	))
	return false, nil
}

func (g *gateContext) stepVet() (bool, error) {
	if len(g.changedGoFiles()) == 0 {
		return true, nil
	}
	_, err := g.runGateCommand(g.sourceRoot(), "go", "vet", "./...")
	return false, err
}

func (g *gateContext) pathsTouchGo() bool {
	for _, p := range g.paths {
		if strings.HasSuffix(p, ".go") || p == "go.mod" || p == "go.sum" {
			return true
		}
	}
	return false
}

func (g *gateContext) pathsTouchAny(prefixes ...string) bool {
	for _, p := range g.paths {
		for _, prefix := range prefixes {
			if p == prefix || strings.HasPrefix(p, prefix) {
				return true
			}
		}
	}
	return false
}

func (g *gateContext) stepBuild() (bool, error) {
	if !g.pathsTouchGo() && !g.pathsTouchAny("cmd/", "internal/") {
		g.note("build skipped: no Go-owned source or asset paths in -paths")
		return true, nil
	}
	_, err := g.runGateCommand(g.sourceRoot(), "go", "build", "./...")
	return false, err
}

// stepTest follows the compiled import graph for production changes and keeps
// test-only edits with their owner. Selection and evidence reuse share one input graph.
// stepTest runs the whole test phase in order: the changed owners, then the
// remaining groups; the pipeline runs the two halves as their own checks so
// the lanes can start between them.
func (g *gateContext) stepTest(ctx context.Context) (bool, error) {
	skipped, err := g.stepTestOwners(ctx)
	if err != nil || skipped {
		return skipped, err
	}
	return g.stepTestRest(ctx)
}

// stepTestOwners derives the test scope, prepares the package evidence
// ledger and runs the changed source owners first.
func (g *gateContext) stepTestOwners(ctx context.Context) (bool, error) {
	scope, err := g.deriveTestScope()
	if err != nil {
		return false, err
	}
	if len(scope.direct)+len(scope.dependent) == 0 {
		g.note("tests skipped: no Go package owns a compiler or repository input in -paths")
		return true, nil
	}
	direct, dependent := scope.direct, scope.dependent
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	boundaryCoverage, err := automationcheck.AgentHarnessBoundaryCoverage(snapshot, scope.productionPaths)
	if err != nil {
		return false, err
	}
	selectedTests := append(slices.Clone(direct), dependent...)
	if err := automationcheck.RequireAgentHarnessBoundaries(boundaryCoverage, selectedTests); err != nil {
		return false, err
	}
	if len(boundaryCoverage.Boundaries) != 0 {
		g.note("assembled agent boundaries: " + strings.Join(boundaryCoverage.Boundaries, ","))
	}
	if len(direct)+len(dependent) == 0 {
		g.note("tests skipped: changed packages have no importers and no tests resolved")
		return true, nil
	}
	g.note(fmt.Sprintf("test scope: %d direct + %d dependent packages (derived from import graph)", len(direct), len(dependent)))
	g.note(fmt.Sprintf("test exclusions: %d packages without affected compiled production or test inputs", scope.excluded))
	if len(scope.opaqueRuntimeInputs) != 0 {
		g.note("test scope: runtime consumers bind named commands, named paths or every repository input: " + strings.Join(scope.opaqueRuntimeInputs, ","))
	}
	if len(scope.opaqueReaders) != 0 {
		g.note(fmt.Sprintf("test scope: %d opaque reader(s) bound to every root; first=%s", len(scope.opaqueReaders), scope.opaqueReaders[0]))
	}
	if len(scope.unresolved) != 0 {
		g.note("test scope widened for global or unresolved Go inputs: " + strings.Join(scope.unresolved, ","))
	}
	inputGraph, err := g.inputGraph()
	if err != nil {
		return false, err
	}
	directInputs, err := packageInputIdentities(inputGraph, direct)
	if err != nil {
		return false, err
	}
	dependentInputs, err := packageInputIdentities(inputGraph, dependent)
	if err != nil {
		return false, err
	}
	ledger, err := g.openPackageEvidence()
	if err != nil {
		return false, err
	}
	if err := ledger.prepare(ctx, direct, "short", directInputs, g.retryCache); err != nil {
		return false, err
	}
	if err := ledger.prepare(ctx, dependent, "complete", dependentInputs, g.retryCache); err != nil {
		return false, err
	}
	directPending, directReused, err := g.packageCachePartition(direct, "short", directInputs)
	if err != nil {
		return false, err
	}
	var edited, remaining []string
	for _, pkg := range directPending {
		if slices.Contains(scope.edited, pkg) {
			edited = append(edited, pkg)
		} else {
			remaining = append(remaining, pkg)
		}
	}
	if len(edited) > 0 {
		g.note(fmt.Sprintf("test order: changed source owners first [%s]; %d other direct packages remain", strings.Join(edited, ","), len(remaining)))
	}
	dependentPending, dependentReused, err := g.packageCachePartition(dependent, "complete", dependentInputs)
	if err != nil {
		return false, err
	}
	g.testPlan = &testGroups{
		ledger: ledger, directInputs: directInputs, dependentInputs: dependentInputs,
		edited: edited, remaining: remaining, dependent: dependentPending,
		reused: directReused + dependentReused, pending: len(directPending) + len(dependentPending),
	}
	return false, g.runTestGroup(ctx, edited, true)
}

// testGroups carries the prepared test scope from the changed-owners check
// to the check that runs the rest of the groups, so the lanes can start once
// the changed owners pass while the remaining packages still run.
type testGroups struct {
	ledger                        *packageEvidenceLedger
	directInputs, dependentInputs map[string]artifact.ID
	edited, remaining, dependent  []string
	reused, pending, executed     int
}

// stepTestDevice runs the device packages of the remaining direct group and
// of the dependent group, prepared by the changed-owners check, alone on
// the device ahead of the lanes; a skipped owners check skips it too.
func (g *gateContext) stepTestDevice(ctx context.Context) (bool, error) {
	if g.testPlan == nil {
		return true, nil
	}
	return false, g.runRestParts(ctx, true)
}

// stepTestRest runs the host packages of the remaining direct group and of
// the dependent group beside the lanes.
func (g *gateContext) stepTestRest(ctx context.Context) (bool, error) {
	if g.testPlan == nil {
		return true, nil
	}
	return false, g.runRestParts(ctx, false)
}

// runRestParts runs the device or the host part of the remaining direct
// group in the short mode, then of the dependent group in the complete mode.
func (g *gateContext) runRestParts(ctx context.Context, device bool) error {
	part := func(group []string) ([]string, error) {
		devices, host, err := g.splitDevice(group)
		if device {
			return devices, err
		}
		return host, err
	}
	remaining, err := part(g.testPlan.remaining)
	if err != nil {
		return err
	}
	if err := g.runTestGroup(ctx, remaining, true); err != nil {
		return err
	}
	dependent, err := part(g.testPlan.dependent)
	if err != nil {
		return err
	}
	if len(dependent) == 0 {
		g.packageCacheAudit(g.testPlan.reused, g.testPlan.pending)
		return nil
	}
	observe := g.packagePassObserver(ctx, g.testPlan.ledger, "complete", g.testPlan.dependentInputs)
	var report testevidence.GoTestReport
	if device {
		report, err = g.runDeviceBatch(ctx, dependent, false, observe, g.runGoTestsAdmitted)
	} else {
		report, err = g.runGoTests(ctx, dependent, false, observe)
	}
	if len(report.Skipped)+len(report.Unavailable) > 0 {
		g.note(fmt.Sprintf(
			"dependent fixture evidence not credited: %d skipped [%s], %d unavailable [%s]",
			len(report.Skipped), strings.Join(report.Skipped, "; "), len(report.Unavailable), strings.Join(report.Unavailable, "; "),
		))
	}
	g.packageCacheAudit(g.testPlan.reused, g.testPlan.pending)
	return err
}

// runTestGroup runs one direct group in device-first batches and, when a
// batch fails, audits the packages executed so far.
func (g *gateContext) runTestGroup(ctx context.Context, group []string, short bool) error {
	devices, host, err := g.splitDevice(group)
	if err != nil {
		return err
	}
	observe := g.packagePassObserver(ctx, g.testPlan.ledger, "short", g.testPlan.directInputs)
	for _, batch := range []struct {
		packages []string
		device   bool
	}{{devices, true}, {host, false}} {
		if len(batch.packages) == 0 {
			continue
		}
		g.testPlan.executed += len(batch.packages)
		if batch.device {
			_, err = g.runDeviceBatch(ctx, batch.packages, short, observe, g.runGoTestsAdmitted)
		} else {
			_, err = g.runGoTests(ctx, batch.packages, short, observe)
		}
		if err != nil {
			g.packageCacheAudit(g.testPlan.reused, g.testPlan.executed)
			return err
		}
	}
	return nil
}

func (g *gateContext) packageCachePartition(packages []string, mode string, inputs map[string]artifact.ID) ([]string, int, error) {
	if g.retryCache == nil {
		cache := g.loadRetryCache()
		cache.Compact()
		g.retryCache = &cache
	}
	pending := make([]string, 0, len(packages))
	reused := 0
	for _, packagePath := range packages {
		hit, err := g.retryCache.PackageReusable(packagePath, mode, inputs[packagePath])
		if err != nil {
			return nil, 0, err
		}
		if hit {
			reused++
		} else {
			pending = append(pending, packagePath)
		}
	}
	return pending, reused, nil
}

func (g *gateContext) packageCacheAudit(reused, executed int) {
	if reused+executed > 0 {
		g.note(fmt.Sprintf("package test evidence: %d reused + %d executed", reused, executed))
	}
}

func (g *gateContext) runGoTests(ctx context.Context, packages []string, short bool, observe func(string, bool) error) (testevidence.GoTestReport, error) {
	return g.runGoTestsAdmitted(ctx, packages, short, observe, true)
}

// testRunner runs packages as runGoTestsAdmitted does; the device batch
// takes it so a test can stand in for go test.
type testRunner func(ctx context.Context, packages []string, short bool, observe func(string, bool) error, leased bool) (testevidence.GoTestReport, error)

// errContentionBudget names the exhausted wait for a refused exclusive claim.
var errContentionBudget = errors.New("device contention budget exhausted")

// runDeviceBatch runs one device batch under the shared lease and, when its
// only failures were refused exclusive claims, runs each refused package
// again alone and outside the lease: a package whose code claims the device
// exclusively is refused beneath any holder's lease, its siblings' and the
// gate's own, and admits once no other process holds the device. The wait
// for a foreign holder is bounded by the admission budget, which bounds only
// the wait; each run itself is bounded by the caller's context.
func (g *gateContext) runDeviceBatch(ctx context.Context, batch []string, short bool, observe func(string, bool) error, run testRunner) (testevidence.GoTestReport, error) {
	report, err := run(ctx, batch, short, observe, true)
	if err == nil || !report.ContentionOnly() {
		return report, err
	}
	contended := report.Contended
	g.note(fmt.Sprintf("device contention: %d package(s) refused an exclusive claim under the shared lease [%s]; each runs again alone outside it under the %s admission budget",
		len(contended), strings.Join(contended, ","), testAdmissionBudget))
	wait, cancel := context.WithTimeoutCause(ctx, testAdmissionBudget, errContentionBudget)
	defer cancel()
	for _, pkg := range contended {
		attempts := 0
		err := processcontrol.AwaitResource(wait, func() error {
			attempts++
			report, err = run(ctx, []string{pkg}, short, observe, false)
			if err != nil && report.ContentionOnly() {
				return processcontrol.ErrResourceBusy
			}
			return err
		})
		if err != nil {
			if errors.Is(err, processcontrol.ErrResourceBusy) || wait.Err() != nil {
				return report, fmt.Errorf("device contention: %s refused its exclusive claim %d time(s): %w", pkg, attempts, errors.Join(processcontrol.ErrResourceBusy, err, context.Cause(wait)))
			}
			return report, err
		}
		g.note(fmt.Sprintf("device contention: %s admitted alone after %d attempt(s)", pkg, attempts))
	}
	return report, nil
}

// runGoTestsAdmitted runs the packages under the gate's shared device lease
// when leased, else with no lease, so a package claiming the device
// exclusively is not refused by the gate's own lease.
func (g *gateContext) runGoTestsAdmitted(ctx context.Context, packages []string, short bool, observe func(string, bool) error, leased bool) (testevidence.GoTestReport, error) {
	args := []string{"test", "-json", "-count=1", "-failfast"}
	if short {
		args = append(args, "-short")
	}
	environment, err := g.sourceEnvironment()
	if err != nil {
		return testevidence.GoTestReport{}, err
	}
	// Preserve explicit GOFLAGS, including values persisted by go env -w.
	flags, err := commandEnvironment(g.sourceRoot(), environment, "go", "env", "GOFLAGS")
	if err != nil {
		return testevidence.GoTestReport{}, err
	}
	options := testevidence.GoTestOptions{Short: short, DiagnosticBytes: clioptions.DiagnosticTailBytes, Observe: observe}
	if !strings.Contains(flags, "-timeout") && !strings.Contains(flags, "-test.timeout") {
		// Apply Go's default ten-minute bound to active tests, not accumulated
		// suite time. Lifecycle silence remains bounded; output cannot renew it.
		const activeTestBudget = 10 * time.Minute
		args = append(args, "-timeout=0")
		options.ActiveTimeout = activeTestBudget
	}
	release := func() error { return nil }
	if leased {
		release, err = g.admitTestResources(ctx, packages, environment)
		if err != nil {
			return testevidence.GoTestReport{}, err
		}
	}
	report, err := testevidence.RunGoTestCommand(ctx, processcontrol.Command{
		Path: "go", Args: append(args, packages...), Dir: g.sourceRoot(), Env: environment,
	}, options)
	if len(report.ClassifiedSkipped) != 0 {
		g.note(fmt.Sprintf("short-profile exclusions (no full-test credit): %d [%s]", len(report.ClassifiedSkipped), strings.Join(report.ClassifiedSkipped, ",")))
	}
	err = errors.Join(err, release())
	if short || len(report.Failed)+len(report.Unfinished) > 0 {
		err = errors.Join(err, testevidence.RequireComplete(report))
	}
	if err != nil {
		return report, fmt.Errorf("go test evidence: %w\n%s", err, strings.Join(report.Diagnostics, "\n"))
	}
	return report, nil
}

// stepMagics enforces repository-wide zero debt.
func (g *gateContext) stepMagics() (bool, error) {
	goSource := false
	for _, path := range g.paths {
		if strings.HasSuffix(path, ".go") {
			goSource = true
		}
	}
	if !goSource {
		return true, nil
	}
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	documents, aliases, err := activeMagicBindings(g.repo, g.storePath)
	if err != nil {
		return false, err
	}
	report, err := closurescan.ValidatePermanentActiveAuthority(snapshot, documents, aliases)
	if err != nil && staleClosureAuthorityFailure(err) {
		// The safe deterministic remediation: rebind unambiguous history to
		// current offsets in the same store, then revalidate exactly once.
		// Admitted content never changes; only stale alias offsets move.
		if remediationErr := g.remediateStaleClosureBindings(); remediationErr != nil {
			return false, errors.Join(err, remediationErr)
		}
		documents, aliases, err = activeMagicBindings(g.repo, g.storePath)
		if err != nil {
			return false, err
		}
		report, err = closurescan.ValidatePermanentActiveAuthority(snapshot, documents, aliases)
	}
	if err != nil {
		return false, fmt.Errorf(
			"%w; remediate with `go run ./cmd/closure-scan -import-store %s` and catalog what remains, then re-run the gate",
			err, gateStorePath,
		)
	}
	g.note(fmt.Sprintf(
		"permanent magic authority: production=%d classified=%d tests=%d open=0 stale=0 policy_copies=0",
		report.ProductionSites, report.ClassifiedSites, report.TestSites,
	))
	return false, nil
}

// staleClosureAuthorityFailure classifies magic-authority refusals whose fix
// is the deterministic same-store rebind: catalogued sites whose byte offsets
// moved under the active aliases, either reported stale directly or surfacing
// as an uncatalogued site that exact prior history still covers.
func staleClosureAuthorityFailure(err error) bool {
	message := err.Error()
	return strings.Contains(message, "stale active binding") ||
		strings.Contains(message, "uncatalogued production policy")
}

func (g *gateContext) remediateStaleClosureBindings() error {
	if g.preflight {
		return fmt.Errorf("preflight: repair required; run `go run ./cmd/closure-scan -import-store %s` outside preflight", g.storePath)
	}
	started := time.Now()
	out, err := g.runGateCommand(g.sourceRoot(), "go", "run", "./cmd/closure-scan", "-import-store", filepath.Join(g.repo, g.storePath))
	if err != nil {
		return fmt.Errorf("gate: closure rebind remediation: %w", err)
	}
	g.note(fmt.Sprintf(
		"remediation: closure rebind applied (%s) wall=%dms",
		strings.TrimSpace(out), time.Since(started).Milliseconds(),
	))
	return nil
}

func activeMagicBindings(repo, storePath string) ([]closureledger.Document, map[string]artifact.ID, error) {
	store, err := overgodb.OpenReadOnly(filepath.Join(repo, storePath))
	if err != nil {
		return nil, nil, err
	}
	defer store.Close()
	var documents []closureledger.Document
	aliases := map[string]artifact.ID{}
	_, err = overgodb.VisitDecodedDocuments(context.Background(), store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: closureledger.MediaType, Schema: closureledger.Schema,
		}}, AliasPrefixes: []string{closureledger.ActiveAliasPrefix}, Order: overgodb.DocumentOldestFirst,
	}, closureledger.Parse, func(view overgodb.DocumentView, document closureledger.Document) error {
		if document.ID != view.Content.Descriptor.ID {
			return errors.New("magic scan: active document identity mismatch")
		}
		documents = append(documents, document)
		for _, alias := range view.Aliases {
			aliases[alias] = document.ID
		}
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	return documents, aliases, nil
}

// stepArchitecture is the permanent entry-authority ratchet. Unlike magics it
// never skips: a documentation-only commit still validates the complete
// candidate source inventory, so no commit shape can land a bypass around the
// established storage, process, capability, tool, trigger, and promotion
// entry authorities. It reuses the gate's cached source snapshot in-process
// and launches no second verification process; the check retires only when
// the authorities it guards disappear.
func (g *gateContext) stepArchitecture() (bool, error) {
	started := time.Now()
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	required := []closurescan.EntryAuthorityDomain{
		closurescan.EntryAuthorityStorage, closurescan.EntryAuthorityProcess,
		closurescan.EntryAuthorityCapability, closurescan.EntryAuthorityTool,
		closurescan.EntryAuthorityTrigger, closurescan.EntryAuthorityPromotion,
		closurescan.EntryAuthorityGoOnly,
	}
	rules := make([]closurescan.EntryAuthorityRule, 0, len(required))
	for _, domain := range required {
		rule, found := closurescan.EntryAuthorityRuleFor(domain)
		if !found {
			return false, fmt.Errorf("architecture: entry authority table lost the %s domain", domain)
		}
		rules = append(rules, rule)
	}
	report, err := closurescan.ValidateEntryAuthorities(snapshot, rules)
	if err != nil {
		return false, fmt.Errorf("architecture: %w", err)
	}
	domains := make([]string, 0, len(report.Domains))
	for _, domain := range report.Domains {
		domains = append(domains, string(domain.Domain))
	}
	g.note(fmt.Sprintf(
		"entry authority ratchet: domains=%s wall=%dms",
		strings.Join(domains, ","), time.Since(started).Milliseconds(),
	))
	return false, nil
}

func (g *gateContext) stepAcceptance() (bool, error) {
	contract, err := completionAcceptanceContract(g.repo, g.planRef, g.completionAuthority)
	if err != nil {
		return false, err
	}
	if err := g.verifyAcceptedCandidate(contract.verify, true); err != nil {
		return false, err
	}
	g.stepEvidence["acceptance"] = contract.evidence
	return false, nil
}

func (g *gateContext) verifyAcceptedCandidate(verify string, complete bool) error {
	tree, err := g.plannedTree()
	if err != nil {
		return err
	}
	if g.manifestPlan == nil || candidateTreeKey(tree) != g.manifestPlan.CandidateTree {
		return errors.New("acceptance: immutable candidate tree differs from the manifest plan")
	}
	if err := g.requirePreparedCandidate(g.manifestPlan.CandidateTree); err != nil {
		return err
	}
	verdict, err := g.executeCandidateVerifier(tree, verify)
	if err != nil {
		return err
	}
	if verdict != testevidence.ClassifyVerifyCommand(verify) {
		return errors.New("acceptance: verifier returned the wrong classifier verdict")
	}
	if complete {
		g.acceptedTree = tree
	}
	return nil
}

type acceptanceContract struct {
	evidence string
	verify   string
}

func completionAcceptanceContract(repo, reference string, completions plan.CompletionAuthority) (acceptanceContract, error) {
	document, err := plan.Load(filepath.Join(repo, plan.Path))
	if err != nil {
		return acceptanceContract{}, err
	}
	return completionAcceptanceContractForPlan(document, reference, completions)
}

func completionAcceptanceEvidenceForPlan(
	document plan.Plan,
	reference string,
	completions plan.CompletionAuthority,
) (string, error) {
	contract, err := completionAcceptanceContractForPlan(document, reference, completions)
	return contract.evidence, err
}

func completionAcceptanceContractForPlan(
	document plan.Plan,
	reference string,
	completions plan.CompletionAuthority,
) (acceptanceContract, error) {
	role, roleErr := plan.AutomationRole("")
	if roleErr != nil {
		return acceptanceContract{}, roleErr
	}
	item, step, open := plan.Current(document, role, completions)
	if !open {
		return acceptanceContract{}, errors.New("acceptance: plan has no current open step")
	}
	if current := item.ID + "/" + step.ID; current != reference {
		return acceptanceContract{}, fmt.Errorf("acceptance: current plan step %s differs from gate authority %s", current, reference)
	}
	evidence, err := runrecord.FormatCompletionAcceptanceEvidence(
		testevidence.CurrentVerifyPolicy, reference, step.Verify,
	)
	if err != nil {
		return acceptanceContract{}, err
	}
	return acceptanceContract{evidence: evidence, verify: step.Verify}, nil
}

func (g *gateContext) executeCandidateVerifier(
	tree, verify string,
) (verdict testevidence.VerdictClass, err error) {
	err = g.withCandidateWorktree(tree, func(worktree string) error {
		environment, executeErr := g.sourceEnvironment()
		if executeErr != nil {
			return executeErr
		}
		verdict, executeErr = planverify.Execute(context.Background(), worktree, verify, environment)
		if executeErr != nil {
			return executeErr
		}
		return requireCandidateWorktreeUnchanged(worktree, tree)
	})
	return verdict, err
}

func requireCandidateWorktreeUnchanged(worktree, tree string) error {
	for _, arguments := range [][]string{
		{"diff", "--name-only", "-z", "--"},
		{"ls-files", "--others", "--exclude-standard", "-z"},
		{"ls-files", "--others", "--ignored", "--exclude-standard", "-z"},
	} {
		changed, err := command(worktree, "git", arguments...)
		if err != nil {
			return err
		}
		if changed != "" {
			return errors.New("acceptance: verifier mutated its immutable candidate worktree")
		}
	}
	indexTree, err := gitWriterCommand(worktree, "write-tree")
	if err != nil {
		return err
	}
	if strings.TrimSpace(indexTree) != tree {
		return errors.New("acceptance: verifier changed the accepted candidate index")
	}
	return nil
}

func (g *gateContext) withCandidateWorktree(tree string, use func(string) error) (err error) {
	if !validGitObjectID(tree) || use == nil {
		return errors.New("gate: candidate worktree requires an exact tree and callback")
	}
	if g.candidateRoot != "" {
		if tree != g.candidateTree {
			return errors.New("gate: active candidate differs from requested tree")
		}
		return use(g.candidateRoot)
	}
	temporary, err := os.MkdirTemp("", "overgo-gate-acceptance-*")
	if err != nil {
		return err
	}
	worktree := filepath.Join(temporary, "candidate")
	added := false
	defer func() {
		if added {
			_, cleanupErr := gitWriterCommand(g.repo, "worktree", "remove", "--force", worktree)
			err = errors.Join(err, cleanupErr)
		}
		err = errors.Join(err, os.RemoveAll(temporary))
	}()
	if _, err := gitWriterCommand(
		g.repo, "worktree", "add", "--quiet", "--detach", "--no-checkout", worktree, "HEAD",
	); err != nil {
		return err
	}
	added = true
	if _, err := gitWriterCommand(worktree, "read-tree", tree); err != nil {
		return err
	}
	if _, err := gitWriterCommand(worktree, "checkout-index", "--all", "--force", "--index"); err != nil {
		return err
	}
	g.candidateRoot, g.candidateTree = worktree, tree
	g.packageGraph = nil
	defer func() {
		g.dependencyCostAudit(g.steps)
		g.candidateRoot, g.candidateTree = "", ""
		g.packageGraph = nil
	}()
	return use(worktree)
}

// sourceRoot separates immutable verification inputs from Git and store ownership.
func (g *gateContext) sourceRoot() string {
	return cmp.Or(g.candidateRoot, g.repo)
}

// Bind external data explicitly; never project ignored source into the candidate.
func (g *gateContext) sourceEnvironment() ([]string, error) {
	roots, err := dataroot.Resolve(g.repo)
	if err != nil {
		return nil, err
	}
	environment := gitauthority.RepositoryEnvironment()
	if strings.TrimSpace(os.Getenv(dataroot.Env)) == "" {
		environment = append(environment, dataroot.Env+"="+g.repo)
	}
	if os.Getenv(audioReferenceEnv) == "" {
		environment = append(environment, audioReferenceEnv+"="+audioReferenceStore(roots))
	}
	return environment, nil
}

// audioReferenceEnv names the operator's override of the audio reference
// store; the declared roots supply it otherwise.
const audioReferenceEnv = "OVERGO_AUDIO_REFERENCE_STORE"

// audioReferenceStore derives the audio parity acceptances' reference store
// the way the acceptances derive theirs: the operator's override, else the
// store the declared data roots name for it, which is the checkout's own
// store unless local-models.json declares another checkout's.
func audioReferenceStore(roots dataroot.Roots) string {
	return cmp.Or(os.Getenv(audioReferenceEnv), roots.AudioReference)
}

// splitDevice parts one group into the packages whose tests need the device
// and the host-only rest, each in the group's order.
func (g *gateContext) splitDevice(group []string) (devices, host []string, err error) {
	if len(group) == 0 {
		return nil, nil, nil
	}
	graph, err := g.inputGraph()
	if err != nil {
		return nil, nil, err
	}
	devices, err = graph.devicePackages(group)
	if err != nil {
		return nil, nil, err
	}
	if len(devices) == 0 {
		return nil, group, nil
	}
	for _, pkg := range group {
		if !slices.Contains(devices, pkg) {
			host = append(host, pkg)
		}
	}
	g.note(fmt.Sprintf("test order: %d device packages first under the shared lease [%s]; %d host packages follow without it", len(devices), strings.Join(devices, ","), len(host)))
	return devices, host, nil
}
