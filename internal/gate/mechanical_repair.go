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

// The reviewed harness surface ceiling the architecture ratchet reads.
const harnessSurfaceBaselineFile = "docs/harness_surface_baseline.json"

// mechanicalRepair is one derived-file repair the gate applies before the
// candidate freezes: the phase that consumes its result, the repository
// files it may rewrite, and the writer. A writer that would move a
// reviewed threshold refuses instead of writing.
type mechanicalRepair struct {
	name  string
	phase string
	files func(g *gateContext) []string
	apply func(g *gateContext) error
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
		{name: "closure rebind", phase: "magics", files: none, apply: (*gateContext).remediateStaleClosureBindings},
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
	}
}

// stageMechanicalRepairs applies every registered repair before the
// candidate freezes when the commit changes a Go input, records each as a
// staged repair in the audit with the files it rewrote, and binds every
// rewritten file into the planned paths so verification runs over the
// repaired candidate. A refused repair stops the gate with its reason.
func (g *gateContext) stageMechanicalRepairs() error {
	if len(g.changedGoFiles()) == 0 {
		return nil
	}
	// The formatter rewrites the Go sources the other repairs read, so it
	// runs first; the rest write disjoint files and run in one wave.
	repairs := g.mechanicalRepairs()
	if err := g.stageRepairs(repairs[:1]); err != nil {
		return err
	}
	return g.stageRepairs(repairs[1:])
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

// repairModernGoCensus republishes the modern-Go census and lowers the
// baseline through their own command; the command refuses a debt increase,
// so no ceiling moves upward here.
func (g *gateContext) repairModernGoCensus() error {
	for _, mode := range []string{"-publish-census", "-lower-baseline"} {
		if out, err := g.runGateCommand(g.repo, "go", "run", "./cmd/modern-census", mode); err != nil {
			return fmt.Errorf("modern-census %s: %w: %s", mode, err, strings.TrimSpace(out))
		}
	}
	return nil
}

// repairHarnessSurface republishes the harness surface baseline when the
// candidate tightened it and refuses with the delta when the candidate
// would raise a reviewed ceiling or add a layer violation.
func (g *gateContext) repairHarnessSurface() error {
	path := filepath.Join(g.repo, filepath.FromSlash(harnessSurfaceBaselineFile))
	var baseline closurescan.HarnessSurface
	if err := jsonfile.DecodeStrict(path, &baseline); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	snapshot, err := repoanalysis.DiscoverGo(g.repo, "internal", "cmd")
	if err != nil {
		return err
	}
	surface, err := closurescan.BuildAgentHarnessSurface(snapshot)
	if err != nil {
		return err
	}
	if regressions := closurescan.AgentHarnessSurfaceRegressions(baseline, surface); len(regressions) != 0 {
		var deltas []string
		for _, regression := range regressions {
			deltas = append(deltas, fmt.Sprintf("%s %d -> %d %s", regression.Metric, regression.Base, regression.Value, regression.Detail))
		}
		return fmt.Errorf("the repair would move a reviewed threshold; republish %s with the reason recorded: %s", harnessSurfaceBaselineFile, strings.Join(deltas, "; "))
	}
	current, err := json.MarshalIndent(baseline, "", " ")
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(surface, "", " ")
	if err != nil {
		return err
	}
	if bytes.Equal(current, encoded) {
		return nil
	}
	// The republished baseline keeps the committed file's own mode.
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(encoded, '\n'), info.Mode().Perm())
}
