package gate

import (
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
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"overgo/internal/agentworkflow"
	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
	"overgo/internal/codeprofile"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
)

type plannedPipeline struct {
	definitions []automationcheck.Check
	invocations []automationcheck.Invocation
	impact      automationcheck.Impact
	surface     automationcheck.Surface
	manifest    *automationcheck.ManifestPlan
	structural  codemanifest.Impact
}

type planDisposition struct {
	Name       string                 `json:"name"`
	Phase      runrecord.Phase        `json:"phase"`
	Reason     string                 `json:"reason"`
	Matched    []automationcheck.Fact `json:"matched,omitempty"`
	RequiredBy []string               `json:"required_by,omitempty"`
}

type gatePlanReport struct {
	Kind              string                         `json:"kind"`
	Schema            int                            `json:"schema"`
	PlanID            string                         `json:"plan_id,omitzero"`
	BaseManifest      string                         `json:"base_manifest,omitzero"`
	CandidateManifest string                         `json:"candidate_manifest,omitzero"`
	CandidateSource   string                         `json:"candidate_source,omitzero"`
	CandidateTree     string                         `json:"candidate_tree,omitzero"`
	Selected          []planDisposition              `json:"selected"`
	Excluded          []planDisposition              `json:"excluded"`
	Unresolved        []planDisposition              `json:"unresolved"`
	AgentContext      *agentworkflow.ManifestContext `json:"agent_context,omitempty"`
}

func buildGatePlanReport(planned plannedPipeline) gatePlanReport {
	report := gatePlanReport{
		Kind: "overgo.gate-plan-inspection", Schema: 2,
		Selected: []planDisposition{}, Excluded: []planDisposition{},
		Unresolved: []planDisposition{},
	}
	if planned.manifest != nil {
		report.PlanID = planned.manifest.ID.String()
		report.BaseManifest = planned.manifest.BaseManifest.String()
		report.CandidateManifest = planned.manifest.CandidateManifest.String()
		report.CandidateSource = planned.manifest.CandidateSource
		report.CandidateTree = planned.manifest.CandidateTree
		if context, err := agentworkflow.NewManifestContext(
			planned.structural, *planned.manifest, nil, agentworkflow.DefaultManifestContextLimits(),
		); err == nil {
			report.AgentContext = &context
		}
	}
	definitions := make(map[string]automationcheck.Descriptor, len(planned.definitions))
	for _, check := range planned.definitions {
		definitions[check.Descriptor.Name] = check.Descriptor
	}
	excluded := make(map[string]string, len(planned.impact.Exclusions))
	for _, exclusion := range planned.impact.Exclusions {
		excluded[exclusion.Check] = exclusion.Reason
		descriptor := definitions[exclusion.Check]
		report.Excluded = append(report.Excluded, planDisposition{
			Name: exclusion.Check, Phase: descriptor.Phase, Reason: exclusion.Reason,
		})
	}
	unknownReason := "no impact producer proved this check independent"
	if len(planned.surface.Unknown) != 0 {
		unknownReason = "impact analysis unresolved: " + strings.Join(planned.surface.Unknown, "; ")
	}
	requiredBy := make(map[string][]string, len(planned.invocations))
	for _, invocation := range planned.invocations {
		for _, dependency := range invocation.Check.Dependencies {
			requiredBy[dependency] = append(requiredBy[dependency], invocation.Check.Name)
		}
	}
	for _, invocation := range planned.invocations {
		disposition := planDisposition{
			Name: invocation.Check.Name, Phase: invocation.Check.Phase,
			Matched: slices.Clone(invocation.Matched), RequiredBy: slices.Clone(requiredBy[invocation.Check.Name]),
		}
		switch {
		case invocation.Check.Always:
			disposition.Reason = "always required by check definition"
		case len(invocation.Matched) != 0:
			disposition.Reason = "matched derived impact facts"
		default:
			disposition.Reason = unknownReason
			report.Unresolved = append(report.Unresolved, disposition)
		}
		report.Selected = append(report.Selected, disposition)
	}
	slices.SortFunc(report.Excluded, func(left, right planDisposition) int { return strings.Compare(left.Name, right.Name) })
	slices.SortFunc(report.Unresolved, func(left, right planDisposition) int { return strings.Compare(left.Name, right.Name) })
	return report
}

func writeGatePlanReport(output io.Writer, planned plannedPipeline) error {
	encoder := json.NewEncoder(output)
	encoder.SetIndent("", "  ")
	return encoder.Encode(buildGatePlanReport(planned))
}

func (g *gateContext) planPipeline() (plannedPipeline, error) {
	tree, err := g.plannedTree()
	if err != nil {
		return plannedPipeline{}, err
	}
	var graphErr error
	var devicePackages []string
	var structural codemanifest.Impact
	var baseManifest, candidateManifest codemanifest.Manifest
	var structuralErr error
	var legacy codeprofile.FunctionImpact
	var legacyErr error
	var verificationBatch *plan.VerificationBatch
	analysisErr := g.withCandidateWorktree(tree, func(root string) error {
		var err error
		verificationBatch, err = planVerificationBatch(root, g.planRef)
		if err != nil {
			return err
		}
		_, graphErr = g.inputGraph()
		// The device lane owns the packages it tests, not every dependent
		// of the device runtime: its full plan plus the packages the changed
		// kernels' functions own.
		if graphErr == nil {
			devicePackages = slices.Clone(automationcheck.DeviceLanePackages[1:])
			if devicePlan, planErr := automationcheck.DevicePlan(root, g.paths); planErr == nil {
				for _, packagePath := range devicePlan.Packages {
					devicePackages = append(devicePackages, strings.TrimPrefix(packagePath, "./"))
				}
			}
			slices.Sort(devicePackages)
			devicePackages = slices.Compact(devicePackages)
		}
		structural, baseManifest, candidateManifest, structuralErr = g.deriveManifestImpact()
		if structuralErr != nil {
			legacy, legacyErr = g.deriveStructuralImpact()
		}
		return nil
	})
	if analysisErr != nil {
		return plannedPipeline{}, analysisErr
	}
	definitions := g.pipelineChecks(devicePackages...)
	definitions, err = g.batchAcceptanceChecks(definitions, verificationBatch)
	if err != nil {
		return plannedPipeline{}, err
	}
	surface := automationcheck.Surface{}
	if structuralErr == nil {
		surface = automationcheck.ManifestSurface(structural)
		if requiresManifestBootstrap(g.paths) {
			surface.Unknown = append(surface.Unknown, "manifest analyzer or planner implementation changed")
			g.note("manifest bootstrap: analyzer-owned change forced the complete selectable plan")
		}
	} else {
		if legacyErr == nil {
			surface = ownershipSurface(legacy)
		}
		surface.Unknown = append(surface.Unknown, "code manifest unavailable: "+structuralErr.Error())
		g.note("code manifest unavailable; owned checks defaulted to run: " + structuralErr.Error())
	}
	if graphErr != nil {
		surface.Unknown = append(surface.Unknown, "package ownership: "+graphErr.Error())
		g.note("package ownership unavailable; owned checks defaulted to run: " + graphErr.Error())
	}
	definitions, surface, coverage, completenessErr := automationcheck.CompleteOwnership(definitions, surface, nil)
	if completenessErr != nil {
		return plannedPipeline{}, completenessErr
	}
	if len(coverage.UncoveredPackages)+len(coverage.UncoveredSymbols) != 0 {
		g.note(fmt.Sprintf(
			"ownership incomplete; owned checks defaulted to run: packages=%d symbols=%d",
			len(coverage.UncoveredPackages), len(coverage.UncoveredSymbols),
		))
	}
	impact := automationcheck.OwnershipImpact(definitions, surface)
	// Symbol reachability uncertain (interface dispatch, reflection, cgo):
	// the linker's rule still decides package-owned checks, since a check's
	// tests observe only packages in their dependency closure.
	if len(surface.Unknown) != 0 && structuralErr == nil && graphErr == nil && !requiresManifestBootstrap(g.paths) && reachabilityOnlyUncertainty(structural) {
		changed := changedPackages(structural)
		if resolver, resolverErr := g.dependencyResolver(); resolverErr == nil {
			impact = automationcheck.OwnershipByDependency(definitions, changed, resolver)
			g.note(fmt.Sprintf("impact fallback: dependency closure over changed packages %s excluded %d owned check(s) under %d uncertainties",
				strings.Join(changed, ","), len(impact.Exclusions), len(surface.Unknown)))
		} else {
			g.note("impact fallback unavailable; owned checks defaulted to run: " + resolverErr.Error())
		}
	}
	// The shell's assets are not Go symbols: a changed web UI path triggers
	// the browser lane that the symbol closure could not select.
	if automationcheck.WebUIPaths(g.paths) {
		impact = impact.Trigger(automationcheck.WebUIImpact, automationcheck.WebUICheckName)
	}
	g.selection = automationcheck.MeasureSelection(definitions, impact)
	g.selectionID = surface.Identity
	checks, err := automationcheck.Plan(definitions, impact)
	if err != nil {
		return plannedPipeline{}, err
	}
	var boundPlan *automationcheck.ManifestPlan
	if structuralErr == nil {
		candidateKey := candidateTreeKey(tree)
		if err := g.requirePreparedCandidate(candidateKey); err != nil {
			return plannedPipeline{}, err
		}
		bound, err := automationcheck.BindManifestPlan(
			baseManifest.ID, candidateManifest.ID, candidateManifest.SourceIdentity, candidateKey,
			surface, impact, checks,
		)
		if err != nil {
			return plannedPipeline{}, err
		}
		g.manifestPlan = &bound
		g.baseManifest, g.candidateManifest = &baseManifest, &candidateManifest
		boundPlan = &bound
		g.selectionID = bound.ID.String()
		g.note("manifest plan: " + bound.ID.String())
	}
	return plannedPipeline{definitions: definitions, invocations: checks, impact: impact, surface: surface, manifest: boundPlan, structural: structural}, nil
}

func requiresManifestBootstrap(paths []string) bool {
	for _, name := range paths {
		name = filepath.ToSlash(name)
		for _, owner := range []string{
			"internal/codemanifest/", "internal/codeprofile/", "internal/repoanalysis/",
			"internal/automationcheck/", "cmd/code-manifest/", "cmd/gate/",
		} {
			if strings.HasPrefix(name, owner) {
				return true
			}
		}
	}
	return false
}

func (g *gateContext) inputGraph() (packageInputGraph, error) {
	if g.packageGraph != nil {
		return *g.packageGraph, nil
	}
	graph, err := loadPackageInputGraph(g.sourceRoot())
	if err == nil {
		g.packageGraph = &graph
	}
	return graph, err
}

func (g *gateContext) deriveStructuralImpact() (codeprofile.FunctionImpact, error) {
	if g.structural != nil {
		return *g.structural, nil
	}
	candidate, err := g.sourceSnapshot()
	if err != nil {
		return codeprofile.FunctionImpact{}, err
	}
	base, err := sourceAtHEAD(g.repo, candidate)
	if err != nil {
		return codeprofile.FunctionImpact{}, err
	}
	selection, err := repoanalysis.HostBuildSelection(g.sourceRoot(), "./cmd/...", "./internal/...")
	if err != nil {
		return codeprofile.FunctionImpact{}, err
	}
	impact, err := codeprofile.DeriveFunctionImpact(base, candidate, selection, selection, g.paths)
	if err == nil {
		g.structural, g.baseSource = &impact, &base
	}
	return impact, err
}

func (g *gateContext) deriveManifestImpact() (codemanifest.Impact, codemanifest.Manifest, codemanifest.Manifest, error) {
	candidate, err := g.sourceSnapshot()
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	base, err := sourceAtHEAD(g.repo, candidate)
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	selection, err := repoanalysis.HostBuildSelection(g.sourceRoot(), "./cmd/...", "./internal/...")
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	baseInputs, err := g.manifestExternalInputs(false)
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	candidateInputs, err := g.manifestExternalInputs(true)
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	manifestCache, err := codemanifest.NewCache(len([]repoanalysis.SourceSnapshot{base, candidate}))
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	baseManifest, _, err := manifestCache.Generate(base, []repoanalysis.BuildSelection{selection}, baseInputs)
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	candidateManifest, reused, err := manifestCache.Generate(candidate, []repoanalysis.BuildSelection{selection}, candidateInputs)
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	if reused {
		g.note("candidate code manifest reused by exact analysis authority")
	}
	delta, err := codemanifest.Diff(baseManifest, candidateManifest)
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	impact, err := codemanifest.Close(baseManifest, candidateManifest, delta)
	if err == nil {
		g.manifestDelta, g.manifestImpact = &delta, &impact
	}
	return impact, baseManifest, candidateManifest, err
}

func (g *gateContext) manifestExternalInputs(candidate bool) ([]codemanifest.ExternalInput, error) {
	var inputs []codemanifest.ExternalInput
	for _, name := range g.paths {
		if strings.HasSuffix(name, ".go") {
			continue
		}
		var digest string
		var found bool
		var err error
		if candidate {
			digest, found, err = worktreeFileDigest(g.repo, name)
		} else {
			digest, found, err = revisionFileDigest(g.repo, "HEAD", name)
		}
		if err != nil {
			return nil, err
		}
		if found {
			inputs = append(inputs, codemanifest.ExternalInput{
				Path: name, ContentID: digest, Kind: "repository-file", Owner: path.Dir(name),
			})
		}
	}
	return inputs, nil
}

func worktreeFileDigest(root, relative string) (string, bool, error) {
	file, err := os.Open(filepath.Join(root, filepath.FromSlash(relative)))
	if errors.Is(err, os.ErrNotExist) {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	defer file.Close()
	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return "", false, err
	}
	return hex.EncodeToString(hasher.Sum(nil)), true, nil
}

func revisionFileDigest(root, revision, relative string) (string, bool, error) {
	object := revision + ":" + filepath.ToSlash(relative)
	probe := newGateGitReaderCommand(root, "cat-file", "-e", object)
	if err := probe.Run(); err != nil {
		if _, missing := err.(*exec.ExitError); missing {
			return "", false, nil
		}
		return "", false, err
	}
	hasher := sha256.New()
	show := newGateGitReaderCommand(root, "show", object)
	show.Stdout = hasher
	if err := show.Run(); err != nil {
		return "", false, err
	}
	return hex.EncodeToString(hasher.Sum(nil)), true, nil
}

// treeStateKey identifies the exact full Git tree produced by overlaying only
// planned paths onto HEAD. Git's tree encoding provides the path/type/length
// framing; the outer domain-separated SHA-256 remains the durable gate key.
func (g *gateContext) treeStateKey() (string, error) {
	tree, err := g.plannedTree()
	if err != nil {
		return "", err
	}
	return candidateTreeKey(tree), nil
}

func candidateTreeKey(tree string) string {
	digest := sha256.Sum256([]byte("overgo-candidate-tree/v1\x00" + tree))
	return hex.EncodeToString(digest[:])
}

func (g *gateContext) requirePreparedCandidate(candidateKey string) error {
	if g.preparation.TreeKey != "" && g.preparation.TreeKey != candidateKey {
		return errors.New("gate: candidate tree changed after durable preparation")
	}
	return nil
}

// plannedTree snapshots the commit candidate into an isolated temporary index.
// It never reads unplanned worktree paths and never mutates the caller's index.
// The snapshot is built once per candidate state: a fingerprint of HEAD and
// the planned paths' bytes names the state, so the drift check before every
// verification check reads the files instead of rebuilding the index, and
// any change to a planned path or to HEAD rebuilds it.
func (g *gateContext) plannedTree() (string, error) {
	fingerprint, err := g.plannedFingerprint()
	if err != nil {
		return "", err
	}
	g.plannedTreeMutex.Lock()
	defer g.plannedTreeMutex.Unlock()
	if g.plannedTreeID != "" && g.plannedTreeFingerprint == fingerprint {
		return g.plannedTreeID, nil
	}
	tree, err := g.buildPlannedTree()
	if err != nil {
		return "", err
	}
	g.plannedTreeID, g.plannedTreeFingerprint = tree, fingerprint
	g.plannedTreeBuilds++
	return tree, nil
}

// plannedFingerprint hashes HEAD and every planned path's bytes; an absent
// planned path hashes as its own marker so a deletion changes the state.
func (g *gateContext) plannedFingerprint() (string, error) {
	head, err := command(g.repo, "git", "rev-parse", "HEAD")
	if err != nil {
		return "", err
	}
	hasher := sha256.New()
	hasher.Write([]byte(strings.TrimSpace(head)))
	for _, path := range g.paths {
		hasher.Write([]byte{0})
		hasher.Write([]byte(path))
		hasher.Write([]byte{0})
		full := filepath.Join(g.repo, filepath.FromSlash(path))
		if info, err := os.Stat(full); err == nil && info.IsDir() {
			// A directory is expanded to its files before the tree is built;
			// until then it names a state of its own.
			hasher.Write([]byte("directory"))
			continue
		}
		data, err := os.ReadFile(full)
		if err != nil {
			if !errors.Is(err, os.ErrNotExist) {
				return "", err
			}
			hasher.Write([]byte("absent"))
			continue
		}
		hasher.Write(data)
	}
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func (g *gateContext) buildPlannedTree() (string, error) {
	temporary, err := os.MkdirTemp("", "overgo-gate-index-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(temporary)
	environment := gitIndexEnvironment(filepath.Join(temporary, "index"))
	if _, err := gitWriterCommandEnvironment(g.repo, environment, "read-tree", "HEAD"); err != nil {
		return "", err
	}
	for _, chunk := range chunkByArgBudget(g.paths) {
		arguments := append([]string{"--literal-pathspecs", "add", "-A", "--"}, chunk...)
		if _, err := gitWriterCommandEnvironment(g.repo, environment, arguments...); err != nil {
			return "", err
		}
	}
	tree, err := gitWriterCommandEnvironment(g.repo, environment, "write-tree")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(tree), nil
}

func (g *gateContext) requireCandidateTree(expected string) error {
	current, err := g.treeStateKey()
	if err != nil {
		return err
	}
	if current != expected {
		return fmt.Errorf("gate: candidate drifted after manifest planning: planned=%s current=%s", expected, current)
	}
	return nil
}

func discoverEnvironment(repo string) (runrecord.Environment, error) {
	// Package tests can read external models and catalogs through either root
	// override. Bind effective locations so retries cannot cross that boundary.
	// This identifies locations, not the contents of external artifacts.
	roots, err := dataroot.Resolve(repo)
	if err != nil {
		return runrecord.Environment{}, err
	}
	rootBytes, err := json.Marshal(struct {
		Roots          dataroot.Roots
		AudioReference string
	}{roots, os.Getenv("OVERGO_AUDIO_REFERENCE_STORE")})
	if err != nil {
		return runrecord.Environment{}, err
	}
	rootIdentity := sha256.Sum256(rootBytes)
	out, err := command(repo, "go", "env", "CGO_ENABLED", "GOFLAGS", "GOEXPERIMENT", "GOTOOLCHAIN")
	if err != nil {
		return runrecord.Environment{}, err
	}
	values := strings.Split(strings.ReplaceAll(out, "\r\n", "\n"), "\n")
	for len(values) < 4 {
		values = append(values, "")
	}
	host, err := os.Hostname()
	if err != nil {
		host = "unknown"
	}
	return runrecord.NewEnvironment(runrecord.Environment{
		Host: host, OS: runtime.GOOS, Arch: runtime.GOARCH, Device: "host", Backend: "go",
		Driver: "cgo=" + strings.TrimSpace(values[0]),
		Runtime: fmt.Sprintf("%s;goflags=%s;goexperiment=%s;gotoolchain=%s;data-roots=%x", runtime.Version(),
			strings.TrimSpace(values[1]), strings.TrimSpace(values[2]), strings.TrimSpace(values[3]), rootIdentity),
	})
}

// expandDirectoryPaths rewrites a -paths entry naming a directory into that
// directory's files (tracked + untracked, gitignore honored). Incident: a
// directory entry never matched porcelain's trailing-slash "?? dir/" form, so
// the add silently no-opped and a commit shipped a message claiming files it
// did not carry. Empty expansion refuses rather than dropping the entry.
func (g *gateContext) expandDirectoryPaths() error {
	var expanded []string
	seen := map[string]bool{}
	for _, p := range g.paths {
		info, err := os.Stat(filepath.Join(g.repo, filepath.FromSlash(p)))
		if err != nil || !info.IsDir() {
			if !seen[p] {
				seen[p] = true
				expanded = append(expanded, p)
			}
			continue
		}
		files, err := gitLines(g.repo, "--literal-pathspecs", "ls-files", "-co", "--exclude-standard", "--", p)
		if err != nil {
			return err
		}
		if len(files) == 0 {
			return fmt.Errorf("-paths entry %s is a directory with no eligible files", p)
		}
		for _, f := range files {
			f = filepath.ToSlash(f)
			if !seen[f] {
				seen[f] = true
				expanded = append(expanded, f)
			}
		}
	}
	g.paths = expanded
	return nil
}

func validatePlannedPaths(paths []string) error {
	if len(paths) == 0 {
		return errors.New("gate: planned paths are empty")
	}
	seen := make(map[string]bool, len(paths))
	for _, candidate := range paths {
		trimmed := strings.TrimSuffix(candidate, "/")
		if trimmed == "" || trimmed == "." || path.Clean(trimmed) != trimmed || filepath.IsAbs(trimmed) ||
			strings.HasPrefix(trimmed, "../") || strings.HasPrefix(trimmed, ":") ||
			strings.ContainsAny(trimmed, "*?[\x00\r\n") {
			return fmt.Errorf("gate: planned path is not one canonical literal repository path: %q", candidate)
		}
		if seen[trimmed] {
			return fmt.Errorf("gate: planned path is duplicated: %q", candidate)
		}
		seen[trimmed] = true
	}
	return nil
}

func gatePlanScratchPath(repo string, preparation artifact.ID) (string, error) {
	if preparation.Kind() != artifact.KindEvidence || preparation.DigestHex() == "" {
		return "", errors.New("gate: plan transaction scratch requires an exact preparation")
	}
	root := filepath.Join(repo, filepath.FromSlash(gatePlanScratchDir))
	path := filepath.Join(root, preparation.DigestHex())
	if err := os.MkdirAll(path, gatePrivateDirectoryMode); err != nil {
		return "", fmt.Errorf("gate: create private plan transaction scratch: %w", err)
	}
	for _, directory := range []string{root, path} {
		info, err := os.Lstat(directory)
		if err != nil {
			return "", err
		}
		if !info.IsDir() || info.Mode()&fs.ModeSymlink != 0 ||
			runtime.GOOS != "windows" && info.Mode().Perm() != gatePrivateDirectoryMode.Perm() {
			return "", errors.New("gate: plan transaction scratch is not an exact private directory")
		}
	}
	return path, nil
}

func (g *gateContext) prepare() error {
	store, err := overgodb.Open(filepath.Join(g.repo, g.storePath))
	if err != nil {
		return fmt.Errorf("prepare gate lifecycle before Git commit: %w", err)
	}
	defer store.Close()
	if err := requireNoPendingGateStateWithStore(g.repo, store); err != nil {
		return err
	}
	return g.prepareWithStore(store)
}

// prepareWithStore publishes the lifecycle preparation through the store that
// already proved pending-state and plan authority. The authority lock keeps
// other gate writers out between those checks, so reopening and replaying the
// same journal here added cost without adding a distinct observation.
func (g *gateContext) prepareWithStore(store *overgodb.Store) error {
	if store == nil {
		return errors.New("gate: lifecycle preparation requires the admission store")
	}
	treeKey, err := g.treeStateKey()
	if err != nil {
		return err
	}
	g.preparation, err = runrecord.NewGatePreparation(treeKey, g.environment.ID, g.start)
	if err != nil {
		return err
	}
	preparationContent, err := g.preparation.Content()
	if err != nil {
		return err
	}
	environmentContent, err := g.environment.Content()
	if err != nil {
		return err
	}
	batch, err := artifact.NewDocumentBatch(
		"gate/prepared/"+g.preparation.ID.String(),
		[]artifact.Content{environmentContent, preparationContent},
		g.preparation.Lineage(), nil,
	)
	if err != nil {
		return err
	}
	g.resolveAttemptStrategy(store)
	alias, err := gatePreparationAlias(context.Background(), store, g.preparation.ID)
	if err != nil {
		return fmt.Errorf("prepare gate lifecycle authority: %w", err)
	}
	batch.Aliases = append(batch.Aliases, alias)
	// The locator is write-ahead: a process loss after the append-only
	// preparation commit must not leave debt that the next gate cannot name.
	if err := g.writeHeartbeat(runrecord.HeartbeatRunning); err != nil {
		return fmt.Errorf("prepare gate lifecycle locator: %w", err)
	}
	preparationCommit, err := store.Commit(context.Background(), batch)
	if err != nil {
		return fmt.Errorf("prepare gate lifecycle before Git commit: %w", err)
	}
	introduction, found, err := store.ArtifactIntroduction(context.Background(), g.preparation.ID)
	if err != nil {
		return fmt.Errorf("prepare gate lifecycle introduction: %w", err)
	}
	if !found || introduction.Commit != preparationCommit {
		return errors.New("prepare gate lifecycle introduction differs from its durable commit")
	}
	g.preparationCommit = introduction.Commit
	return nil
}

// admitPendingGateState runs only while the caller owns the gate mutation lock.
func admitPendingGateState(repo, storePath string, store *overgodb.Store) (string, error) {
	if store == nil {
		return "", errors.New("gate: pending-state admission requires the canonical store")
	}
	var intent gateCommitIntent
	intentErr := readJSON(repo, gateCommitIntentFile, &intent)
	if intentErr != nil && !errors.Is(intentErr, os.ErrNotExist) {
		return "", intentErr
	}
	if intentErr == nil {
		if err := intent.validate(); err != nil {
			return "", err
		}
	}
	var debt gateDebtEnvelope
	debtErr := readJSON(repo, gateDebtFile, &debt)
	if debtErr != nil && !errors.Is(debtErr, os.ErrNotExist) {
		return "", debtErr
	}
	if debtErr == nil || intentErr == nil {
		preparation := intent.Preparation
		if debtErr == nil {
			final, err := gateDebtFinalization(debt)
			if err != nil {
				return "", err
			}
			if intentErr == nil && intent.Preparation != debt.Preparation {
				return "", errors.New("gate: debt and commit intent name different preparations")
			}
			head, err := command(repo, "git", "rev-parse", "HEAD")
			if err != nil {
				return "", err
			}
			if strings.TrimSpace(head) != final.CodeCommit {
				return "", errors.New("gate: recording debt does not name the current revision")
			}
			if final.Outcome == runrecord.OutcomeSucceeded && intentErr == nil {
				candidate := gateContext{repo: repo, paths: intent.Paths}
				tree, err := candidate.plannedTree()
				if err != nil {
					return "", err
				}
				if tree != intent.Tree {
					return "", errors.New("gate: candidate changed before recording recovery")
				}
			}
			preparation = debt.Preparation
		}
		var heartbeat runrecord.GateHeartbeat
		if err := readJSON(repo, gateHeartbeatFile, &heartbeat); err == nil {
			if err := heartbeat.Validate(); err != nil {
				return "", err
			}
			if heartbeat.Preparation != preparation {
				return "", errors.New("gate: recovery locator names another preparation")
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			return "", err
		}
		var err error
		if debtErr == nil {
			_, err = reconcileGateDebt(repo, storePath)
		} else {
			_, err = recoverInterruptedCommit(repo, storePath)
		}
		if err != nil {
			return "", err
		}
		if err := store.Refresh(context.Background()); err != nil {
			return "", err
		}
		if err := requireNoPendingGateStateWithStore(repo, store); err != nil {
			return "", err
		}
		if intentErr == nil && intent.Commit != "" {
			final, found, err := runrecord.GateFinalizationForPreparation(context.Background(), store, intent.Preparation)
			if err != nil {
				return "", err
			}
			if found && final.Outcome == runrecord.OutcomeSucceeded {
				head, err := command(repo, "git", "rev-parse", "HEAD")
				if err != nil {
					return "", err
				}
				if strings.TrimSpace(head) == intent.Commit {
					fmt.Fprintf(os.Stderr, "gate: admission recovered completed commit %s for %s\n", intent.Commit, intent.PlanRef)
					return intent.PlanRef, nil
				}
			}
		}
		fmt.Fprintf(os.Stderr, "gate: admission recovered transaction %s; acceptance remains required\n", preparation)
		return "", nil
	}
	if err := requireNoPendingGateStateWithStore(repo, store); !errors.Is(err, errGateLifecyclePending) {
		return "", err
	}
	recovered, err := recordSelectedUnbatchableFailure(repo, storePath, "")
	if err != nil {
		return "", fmt.Errorf("gate: recover abandoned lifecycle: %w", err)
	}
	if err := store.Refresh(context.Background()); err != nil {
		return "", err
	}
	if err := requireNoPendingGateStateWithStore(repo, store); err != nil {
		return "", err
	}
	fmt.Fprintf(os.Stderr, "gate: admission recovered lifecycle %s; acceptance remains required\n", recovered)
	return "", nil
}

var errGateLifecyclePending = errors.New("gate: unresolved lifecycle")

// requireNoPendingGateStateWithStore checks recovery authority without
// reopening the canonical store. Callers that continue into plan binding or
// preparation retain this exact store view under the gate authority lock.
func requireNoPendingGateStateWithStore(repo string, store *overgodb.Store) error {
	if store == nil {
		return errors.New("gate: pending-state admission requires the canonical store")
	}
	if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(gateDebtFile))); err == nil {
		return errors.New("gate: unresolved tmp/gate_debt.json; run `go run ./cmd/gate -reconcile` before another gate")
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(gateCommitIntentFile))); err == nil {
		return errors.New("gate: interrupted commit intent remains; run `go run ./cmd/gate -recover-interrupted`")
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err := requireNoOutstandingGateLifecycle(context.Background(), store); err != nil {
		return err
	}
	var heartbeat runrecord.GateHeartbeat
	if err := readJSON(repo, gateHeartbeatFile, &heartbeat); errors.Is(err, os.ErrNotExist) {
		return nil
	} else if err != nil {
		return fmt.Errorf("gate: inspect lifecycle locator: %w", err)
	}
	if err := heartbeat.Validate(); err != nil {
		return fmt.Errorf("gate: invalid lifecycle locator: %w", err)
	}
	if heartbeat.State == runrecord.HeartbeatRunning || heartbeat.State == runrecord.HeartbeatRecordDebt {
		return fmt.Errorf("%w: unresolved gate lifecycle locator; run `go run ./cmd/gate -record-failure` before another gate", errGateLifecyclePending)
	}
	return nil
}

// ensureGateStoreAcceleration turns a verified cold replay into the derived
// checkpoint set future opens consume. Checkpoints are acceleration rather
// than authority: Open already validated the journal, Snapshot binds the
// derived state to its exact head and sequence, and a later defect falls back
// to the same canonical log. An empty store has no snapshot boundary to write.
func ensureGateStoreAcceleration(ctx context.Context, store *overgodb.Store) (bool, error) {
	if ctx == nil || store == nil {
		return false, errors.New("gate: store acceleration requires a store and context")
	}
	if store.SnapshotReplay().Loaded {
		return false, nil
	}
	_, sequence := store.Head()
	if sequence == 0 {
		return false, nil
	}
	if _, err := store.Snapshot(ctx); err != nil {
		return false, fmt.Errorf("gate: checkpoint cold admission store: %w", err)
	}
	return true, nil
}

func requireNoOutstandingGateLifecycle(ctx context.Context, store *overgodb.Store) error {
	current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		return fmt.Errorf("gate: resolve lifecycle authority: %w", err)
	}
	lifecycles, err := runrecord.GateLifecyclesInStore(ctx, store)
	if err != nil {
		return fmt.Errorf("gate: inspect lifecycle debt: %w", err)
	}
	complete := make(map[artifact.ID]bool)
	var unresolved []artifact.ID
	for _, preparation := range lifecycles {
		if preparation.State != runrecord.GatePrepared {
			continue
		}
		finalization, finalized, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
		if err != nil {
			return fmt.Errorf("gate: validate lifecycle %s: %w", preparation.ID, err)
		}
		if !finalized {
			unresolved = append(unresolved, preparation.ID)
			continue
		}
		if err := validateCompleteGateFinalization(ctx, store, preparation, finalization); err != nil {
			return fmt.Errorf("gate: validate lifecycle %s: %w", preparation.ID, err)
		}
		complete[finalization.ID] = true
	}
	if found {
		lifecycle, err := runrecord.RequireGateLifecycle(ctx, store, current)
		if err != nil {
			return fmt.Errorf("gate: load lifecycle authority: %w", err)
		}
		if lifecycle.State != runrecord.GateFinalized {
			return fmt.Errorf("%w: OvergoDB records an unresolved prepared lifecycle; run `go run ./cmd/gate -record-failure`", errGateLifecyclePending)
		}
		if !complete[lifecycle.ID] {
			return errors.New("gate: lifecycle authority is not one complete typed gate finalization")
		}
	}
	if len(unresolved) == 1 {
		return fmt.Errorf("%w: OvergoDB records 1 unresolved prepared lifecycle; run `go run ./cmd/gate -record-failure`", errGateLifecyclePending)
	}
	if len(unresolved) != 0 {
		// The refusal carries its own deterministic remediation: one exact
		// argument-bound recovery command per stale preparation.
		commands := make([]string, 0, len(unresolved))
		for _, preparation := range unresolved {
			commands = append(commands, "go run ./cmd/gate -record-failure -preparation "+preparation.String())
		}
		return fmt.Errorf(
			"gate: OvergoDB records %d unresolved prepared lifecycle(s); remediate each with: %s",
			len(unresolved), strings.Join(commands, " ; "),
		)
	}
	return nil
}

func requireCompleteGateFinalization(
	ctx context.Context,
	store *overgodb.Store,
	finalization runrecord.GateLifecycle,
) error {
	if finalization.State != runrecord.GateFinalized || finalization.Preparation == nil || finalization.Result == nil {
		return errors.New("current lifecycle alias does not target a finalization")
	}
	preparation, err := runrecord.RequireGateLifecycle(ctx, store, *finalization.Preparation)
	if err != nil {
		return err
	}
	canonical, found, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
	if err != nil {
		return err
	}
	if !found || canonical.ID != finalization.ID {
		return errors.New("preparation does not have the current alias target as its sole canonical finalization")
	}
	return validateCompleteGateFinalization(ctx, store, preparation, finalization)
}

func validateCompleteGateFinalization(
	ctx context.Context,
	store *overgodb.Store,
	preparation, finalization runrecord.GateLifecycle,
) error {
	if preparation.State != runrecord.GatePrepared || finalization.State != runrecord.GateFinalized ||
		finalization.Preparation == nil || *finalization.Preparation != preparation.ID || finalization.Result == nil ||
		finalization.TreeKey != preparation.TreeKey || finalization.Environment != preparation.Environment ||
		finalization.Started != preparation.Started {
		return errors.New("gate finalization contradicts its preparation")
	}
	result, err := runrecord.RequireGateResult(ctx, store, *finalization.Result)
	if err != nil {
		return fmt.Errorf("load typed gate result: %w", err)
	}
	if result.Environment != finalization.Environment || result.CodeCommit != finalization.CodeCommit ||
		result.Outcome != finalization.Outcome {
		return errors.New("gate result contradicts its finalization")
	}
	if _, err := runrecord.RequireEnvironment(ctx, store, preparation.Environment); err != nil {
		return fmt.Errorf("load typed gate environment: %w", err)
	}
	introduction := func(id artifact.ID, label string) (overgodb.ArtifactIntroduction, error) {
		value, found, err := store.ArtifactIntroduction(ctx, id)
		if err != nil {
			return overgodb.ArtifactIntroduction{}, err
		}
		if !found {
			return overgodb.ArtifactIntroduction{}, fmt.Errorf("%s introduction is absent", label)
		}
		return value, nil
	}
	environmentIntroduction, err := introduction(preparation.Environment, "gate environment")
	if err != nil {
		return err
	}
	preparationIntroduction, err := introduction(preparation.ID, "gate preparation")
	if err != nil {
		return err
	}
	resultIntroduction, err := introduction(result.ID, "gate result")
	if err != nil {
		return err
	}
	finalizationIntroduction, err := introduction(finalization.ID, "gate finalization")
	if err != nil {
		return err
	}
	if environmentIntroduction.Sequence > preparationIntroduction.Sequence {
		return errors.New("gate environment was published after its preparation")
	}
	if preparationIntroduction.Sequence > finalizationIntroduction.Sequence {
		return errors.New("gate preparation was published after its finalization")
	}
	if resultIntroduction.Commit != finalizationIntroduction.Commit ||
		resultIntroduction.Sequence != finalizationIntroduction.Sequence {
		return errors.New("gate result and finalization were not published atomically")
	}
	edges, err := store.Children(ctx, result.ID)
	if err != nil {
		return err
	}
	for _, edge := range edges {
		if edge.Child == finalization.ID && edge.Parent == result.ID && edge.Relation == artifact.RelationDependsOn {
			return nil
		}
	}
	return errors.New("finalization lacks exact gate-result lineage")
}

func gatePreparationAlias(
	ctx context.Context,
	store *overgodb.Store,
	preparation artifact.ID,
) (artifact.AliasBinding, error) {
	// The writable store handle owns a stable view until this preparation is
	// committed. Repeat the complete census here so a preparation appended
	// after the gate-start precheck cannot hide behind a finalized alias.
	if err := requireNoOutstandingGateLifecycle(ctx, store); err != nil {
		return artifact.AliasBinding{}, err
	}
	binding := artifact.AliasBinding{Name: runrecord.GateLifecycleCurrentAlias, Target: preparation}
	current, found, err := artifact.ResolveAlias(ctx, store, binding.Name)
	if err != nil {
		return artifact.AliasBinding{}, err
	}
	if !found {
		return binding, nil
	}
	lifecycle, err := runrecord.RequireGateLifecycle(ctx, store, current)
	if err != nil {
		return artifact.AliasBinding{}, err
	}
	if lifecycle.State != runrecord.GateFinalized {
		return artifact.AliasBinding{}, errors.New("current gate lifecycle is not finalized")
	}
	binding.Previous = &current
	return binding, nil
}

// requireSoleCurrentGatePreparation is the locked-store admission boundary for
// a gate's final Git and record transaction. The current alias must still name
// this exact preparation, its durable introduction must be the receipt carried
// by Git, it must have no finalization yet, and every older preparation must
// already have one complete typed finalization. Refresh the retained handle;
// the gate authority lock excludes competing gate lifecycle writers.
func requireSoleCurrentGatePreparation(
	ctx context.Context,
	store *overgodb.Store,
	preparation runrecord.GateLifecycle,
	preparationCommit artifact.CommitID,
) error {
	if ctx == nil || store == nil || preparation.State != runrecord.GatePrepared ||
		!preparationCommit.Valid() {
		return errors.New("gate: exact current preparation authority is absent")
	}
	if err := preparation.ValidateIdentity(); err != nil {
		return err
	}
	if err := store.Refresh(ctx); err != nil {
		return err
	}
	current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		return err
	}
	if !found || current != preparation.ID {
		return errors.New("gate: current lifecycle alias no longer names this preparation")
	}
	stored, err := runrecord.RequireGateLifecycle(ctx, store, preparation.ID)
	if err != nil {
		return err
	}
	if stored.ID != preparation.ID || stored.State != runrecord.GatePrepared {
		return errors.New("gate: current preparation differs from its typed store authority")
	}
	introduction, found, err := store.ArtifactIntroduction(ctx, preparation.ID)
	if err != nil {
		return err
	}
	if !found || introduction.Commit != preparationCommit {
		return errors.New("gate: current preparation differs from its durable introduction receipt")
	}
	lifecycles, err := runrecord.GateLifecyclesInStore(ctx, store)
	if err != nil {
		return err
	}
	seenCurrent := false
	for _, candidate := range lifecycles {
		if candidate.State != runrecord.GatePrepared {
			continue
		}
		finalization, finalized, err := runrecord.GateFinalizationForPreparation(ctx, store, candidate.ID)
		if err != nil {
			return err
		}
		if candidate.ID == preparation.ID {
			seenCurrent = true
			if finalized {
				return errors.New("gate: current preparation already has a finalization")
			}
			continue
		}
		if !finalized {
			return fmt.Errorf("gate: prior lifecycle %s has no finalization", candidate.ID)
		}
		if err := validateCompleteGateFinalization(ctx, store, candidate, finalization); err != nil {
			return fmt.Errorf("gate: validate prior lifecycle %s: %w", candidate.ID, err)
		}
	}
	if !seenCurrent {
		return errors.New("gate: current preparation is absent from the lifecycle census")
	}
	return nil
}

func appendGateFinalizationAlias(
	batch *artifact.Batch,
	preparation, finalization artifact.ID,
) {
	previous := preparation
	batch.Aliases = append(batch.Aliases, artifact.AliasBinding{
		Name: runrecord.GateLifecycleCurrentAlias, Target: finalization, Previous: &previous,
	})
}
