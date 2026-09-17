package gate

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"overgo/internal/closurescan"
	"overgo/internal/jsonfile"
	"overgo/internal/repoanalysis"
)

// The reviewed harness surface ceiling the architecture ratchet reads, the
// API manifest the served routes and the browser lane census read, and the
// compatibility manifest with the matrix generated from it.
const (
	harnessSurfaceBaselineFile = "docs/harness_surface_baseline.json"
	apiManifestFile            = "docs/api_manifest.json"
	compatibilityManifestFile  = "compatibility.json"
	compatibilityMatrixFile    = "docs/COMPATIBILITY.md"
)

// mechanicalRepair is one derived-file repair the gate applies before the
// candidate freezes: the phase that consumes its result, the repository
// files it may rewrite, and the writer. A writer that would move a
// reviewed threshold refuses instead of writing.
type mechanicalRepair struct {
	name  string
	phase string
	files func(g *gateContext) []string
	apply func(g *gateContext) error
	// applies reports whether the planned paths reach the repair's inputs;
	// nil means a changed Go source.
	applies func(g *gateContext) bool
	// store marks a repair that writes the OvergoDB store; the preflight,
	// which diagnoses without store repair, leaves it to the gate.
	store bool
	// Retain outputs authored on every successful apply, including retries.
	retainOutputs bool
}

// mechanicalRepairs lists the registry: the formatter first, since every
// other repair reads the Go sources it rewrites, then the repairs that
// write disjoint outputs; the closure rebind the magics phase already
// applied on its own is a registry entry, not a second path.
func (g *gateContext) mechanicalRepairs() []mechanicalRepair {
	none := func(*gateContext) []string { return nil }
	return []mechanicalRepair{
		{name: "gofmt", phase: "fmt", files: (*gateContext).changedGoFiles, apply: (*gateContext).repairFormatting},
		{name: "closure rebind", phase: "magics", files: none, apply: (*gateContext).remediateStaleClosureBindings, store: true},
		{
			name: "modern-Go census", phase: "modern-go",
			retainOutputs: true,
			files: func(*gateContext) []string {
				return []string{repoanalysis.ModernGoPublishedCensusFile, repoanalysis.ModernGoBaselineFile}
			},
			apply: (*gateContext).repairModernGoCensus,
		},
		{
			name: "harness surface baseline", phase: "architecture",
			files: func(*gateContext) []string { return []string{harnessSurfaceBaselineFile} },
			apply: (*gateContext).repairHarnessSurface,
		},
		{
			name: "API manifest", phase: "test",
			retainOutputs: true,
			files:         func(*gateContext) []string { return []string{apiManifestFile} },
			apply:         (*gateContext).repairAPIManifest,
		},
		{
			name: "compatibility identities", phase: "claims",
			files:   func(*gateContext) []string { return []string{compatibilityManifestFile, compatibilityMatrixFile} },
			apply:   (*gateContext).repairCompatibilityIdentities,
			applies: (*gateContext).compatibilityEvidenceChanged,
		},
	}
}

// stageMechanicalRepairs applies every registered repair whose inputs the
// planned paths reach before the candidate freezes, records each as a
// staged repair in the audit with the files it rewrote, and binds every
// rewritten file into the planned paths so verification runs over the
// repaired candidate. A refused repair stops the gate with its reason. The
// preflight applies the same registry to the working tree, less the store
// repairs.
func (g *gateContext) stageMechanicalRepairs() error {
	goChanged := len(g.changedGoFiles()) != 0
	var repairs []mechanicalRepair
	for _, repair := range g.mechanicalRepairs() {
		applies := goChanged
		if repair.applies != nil {
			applies = repair.applies(g)
		}
		if applies && !(g.preflight && repair.store) {
			repairs = append(repairs, repair)
		}
	}
	if len(repairs) == 0 {
		return nil
	}
	// The formatter rewrites the Go sources the other repairs read, so it
	// runs first; the rest write disjoint files and run in one wave.
	if repairs[0].name == "gofmt" {
		if err := g.stageRepairs(repairs[:1]); err != nil {
			return err
		}
		repairs = repairs[1:]
	}
	return g.stageRepairs(repairs)
}

// repairAPIManifest rewrites the API manifest from the candidate's sources.
func (g *gateContext) repairAPIManifest() error {
	return g.repairCommand("API manifest update", "./cmd/api-manifest", "-update")
}

// repairCompatibilityIdentities refreshes the evidence identities the
// compatibility manifest pins and the matrix generated from it.
func (g *gateContext) repairCompatibilityIdentities() error {
	return g.repairCommand("compatibility identity refresh", "./cmd/compatibility", "-refresh-identities")
}

func (g *gateContext) repairCommand(label, commandPath, argument string) error {
	if out, err := g.runGateCommand(g.repo, "go", "run", commandPath, argument); err != nil {
		return fmt.Errorf("%s: %w: %s", label, err, strings.TrimSpace(out))
	}
	return nil
}

// compatibilityEvidenceChanged reports a planned path that is the
// compatibility manifest or an evidence file one of its claims pins.
func (g *gateContext) compatibilityEvidenceChanged() bool {
	if slices.Contains(g.paths, compatibilityManifestFile) {
		return true
	}
	raw, err := os.ReadFile(filepath.Join(g.repo, compatibilityManifestFile))
	if err != nil {
		return false
	}
	var manifest struct {
		Claims []struct {
			Evidence []struct {
				Path string `json:"path"`
			} `json:"evidence"`
		} `json:"claims"`
	}
	if json.Unmarshal(raw, &manifest) != nil {
		return false
	}
	for _, claim := range manifest.Claims {
		for _, proof := range claim.Evidence {
			if slices.Contains(g.paths, filepath.ToSlash(proof.Path)) {
				return true
			}
		}
	}
	return false
}

// stageRepairs applies one wave of repairs concurrently and records each
// once every member has finished, in registry order.
func (g *gateContext) stageRepairs(repairs []mechanicalRepair) error {
	type outcome struct {
		files   []string
		before  map[string]string
		started time.Time
		wall    time.Duration
		err     error
	}
	outcomes := make([]outcome, len(repairs))
	for index, repair := range repairs {
		outcomes[index].files = repair.files(g)
		before, err := g.fileDigests(outcomes[index].files)
		if err != nil {
			return err
		}
		outcomes[index].before = before
	}
	var wait sync.WaitGroup
	for index, repair := range repairs {
		wait.Go(func() {
			outcomes[index].started = time.Now()
			outcomes[index].err = repair.apply(g)
			outcomes[index].wall = time.Since(outcomes[index].started)
		})
	}
	wait.Wait()
	for index, repair := range repairs {
		result := outcomes[index]
		if result.err != nil {
			return fmt.Errorf("gate: staged repair %s refused: %w", repair.name, result.err)
		}
		after, err := g.fileDigests(result.files)
		if err != nil {
			return err
		}
		var changed []string
		for _, file := range result.files {
			if result.before[file] != after[file] {
				changed = append(changed, file)
			} else if !repair.retainOutputs {
				continue
			}
			if !slices.Contains(g.paths, file) {
				g.paths = append(g.paths, file)
			}
		}
		g.note(fmt.Sprintf("staged repair: %s for the %s phase wall=%dms rewrote=[%s]", repair.name, repair.phase, result.wall.Milliseconds(), strings.Join(changed, ",")))
	}
	return nil
}

// fileDigests hashes repository files by path; an absent file hashes empty.
func (g *gateContext) fileDigests(files []string) (map[string]string, error) {
	digests := make(map[string]string, len(files))
	for _, file := range files {
		data, err := os.ReadFile(filepath.Join(g.repo, filepath.FromSlash(file)))
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return nil, err
		}
		sum := sha256.Sum256(data)
		digests[file] = hex.EncodeToString(sum[:])
	}
	return digests, nil
}

// repairFormatting formats the planned Go files gofmt lists as unformatted.
func (g *gateContext) repairFormatting() error {
	files := g.changedGoFiles()
	if len(files) == 0 {
		return nil
	}
	var unformatted []string
	for _, chunk := range chunkByArgBudget(files) {
		out, err := command(g.repo, "gofmt", append([]string{"-l"}, chunk...)...)
		if err != nil {
			return err
		}
		unformatted = append(unformatted, strings.Fields(out)...)
	}
	for _, chunk := range chunkByArgBudget(unformatted) {
		if len(chunk) == 0 {
			continue
		}
		if _, err := command(g.repo, "gofmt", append([]string{"-w"}, chunk...)...); err != nil {
			return err
		}
	}
	return nil
}

// Publication and verification share one exact candidate computation. The
// publisher still refuses debt increases and expired authority on every call.
func (g *gateContext) repairModernGoCensus() error {
	input, err := g.refreshModernGoInput()
	if err != nil {
		return err
	}
	census, err := input.candidate()
	if err != nil {
		return err
	}
	info, err := os.Stat(filepath.Join(g.repo, repoanalysis.ModernGoBaselineFile))
	if err != nil {
		return err
	}
	_, err = repoanalysis.PublishModernGoCensus(g.repo, census, true, info.Mode().Perm())
	return err
}

// repairHarnessSurface republishes the harness surface baseline when the
// candidate tightened it and refuses with the delta when the candidate
// would raise a reviewed ceiling or add a layer violation.
func (g *gateContext) repairHarnessSurface() error {
	encoded, err := g.harnessSurfaceUpdate()
	if err != nil || encoded == nil {
		return err
	}
	path := filepath.Join(g.repo, filepath.FromSlash(harnessSurfaceBaselineFile))
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(encoded, '\n'), info.Mode().Perm())
}

// Compute and validate once for diagnostics and the derived-file writer.
func (g *gateContext) harnessSurfaceUpdate() ([]byte, error) {
	path := filepath.Join(g.repo, filepath.FromSlash(harnessSurfaceBaselineFile))
	var baseline closurescan.HarnessSurface
	if err := jsonfile.DecodeStrict(path, &baseline); err != nil {
		if errors.Is(err, os.ErrNotExist) && !g.preflight {
			return nil, nil
		}
		return nil, err
	}
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return nil, err
	}
	surface, err := closurescan.BuildAgentHarnessSurface(snapshot)
	if err != nil {
		return nil, err
	}
	if regressions := closurescan.AgentHarnessSurfaceRegressions(baseline, surface); len(regressions) != 0 {
		var deltas []string
		for _, regression := range regressions {
			deltas = append(deltas, fmt.Sprintf("%s %d -> %d %s", regression.Metric, regression.Base, regression.Value, regression.Detail))
		}
		return nil, fmt.Errorf("the repair would move a reviewed threshold; republish %s with the reason recorded: %s", harnessSurfaceBaselineFile, strings.Join(deltas, "; "))
	}
	current, err := json.MarshalIndent(baseline, "", " ")
	if err != nil {
		return nil, err
	}
	encoded, err := json.MarshalIndent(surface, "", " ")
	if err != nil {
		return nil, err
	}
	if bytes.Equal(current, encoded) {
		return nil, nil
	}
	return encoded, nil
}
