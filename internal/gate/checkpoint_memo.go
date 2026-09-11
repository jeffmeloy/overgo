package gate

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/plan"
)

// checkpointMemoEntry: stable cache slot + memo input of one checkpoint.
type checkpointMemoEntry struct {
	slot  artifact.ID
	input artifact.ID
}

// checkpointMemoKey: declared flush key, else the dispatched plan ref.
func checkpointMemoKey(planRef string, batch *plan.VerificationBatch) string {
	if batch != nil && batch.Flush != nil {
		return batch.Flush.Key
	}
	return planRef
}

// verifierInputs: every command of a verifier -> package arguments of each
// go test segment and the paths of each `test -f` guard; any other segment
// makes the verifier uncacheable with the reason.
func verifierInputs(verify string) (packages, guards []string, err error) {
	for segment := range strings.SplitSeq(verify, "&&") {
		fields := strings.Fields(segment)
		switch {
		case len(fields) == 0:
			return nil, nil, errors.New("empty command segment")
		case len(fields) >= 3 && fields[0] == "test" && fields[1] == "-f":
			guards = append(guards, fields[2])
		case len(fields) >= 2 && fields[0] == "go" && fields[1] == "test":
			count := 0
			for _, field := range fields[2:] {
				if strings.HasPrefix(field, "-") {
					break
				}
				packages = append(packages, field)
				count++
			}
			if count == 0 {
				return nil, nil, errors.New("go test segment names no package")
			}
		default:
			return nil, nil, fmt.Errorf("segment %q is not go test or a file guard", strings.Join(fields, " "))
		}
	}
	if len(packages) == 0 {
		return nil, nil, errors.New("verifier names no package")
	}
	return packages, guards, nil
}

// importPathForDir: repo-relative package directory -> import path of its
// production node; test variants are skipped.
func (graph packageInputGraph) importPathForDir(root, relative string) (string, bool) {
	target := filepath.Clean(filepath.Join(root, filepath.FromSlash(relative)))
	for _, node := range graph.nodes {
		if node.ForTest != "" || node.Dir == "" {
			continue
		}
		if filepath.Clean(node.Dir) == target {
			return node.ImportPath, true
		}
	}
	return "", false
}

// guardCovered: a `test -f` path lies inside one of the named package
// directories, whose input identity then covers it.
func guardCovered(guard string, packages []string) bool {
	guard = filepath.ToSlash(filepath.Clean(guard))
	for _, relative := range packages {
		directory := filepath.ToSlash(filepath.Clean(relative))
		// Root package covers root-level files only; every package is tried.
		if directory == "." && !strings.Contains(guard, "/") || directory != "." && strings.HasPrefix(guard, directory+"/") {
			return true
		}
	}
	return false
}

// checkpointMemoInputs: per checkpoint, a stable slot and a memo input from
// the key, the verify text and the transitive input identity of every
// package the verify names; a verifier with an unsupported segment, an
// uncovered guard, or a package absent from the graph is not memoised and the
// refusal is audited.
func (g *gateContext) checkpointMemoInputs(batch *plan.VerificationBatch) (map[string]checkpointMemoEntry, error) {
	if batch == nil {
		return nil, nil
	}
	key := checkpointMemoKey(g.planRef, batch)
	graph, err := g.inputGraph()
	if err != nil {
		return nil, err
	}
	root, err := filepath.Abs(graph.root)
	if err != nil {
		return nil, err
	}
	memos := make(map[string]checkpointMemoEntry, len(batch.Checkpoints))
	for _, checkpoint := range batch.Checkpoints {
		if reason := g.memoiseCheckpoint(memos, graph, root, key, checkpoint); reason != "" {
			g.note(fmt.Sprintf("checkpoint memo refused: %s: %s", checkpoint.ID, reason))
		}
	}
	return memos, nil
}

// memoiseCheckpoint: one checkpoint -> memo entry, or the refusal reason.
func (g *gateContext) memoiseCheckpoint(
	memos map[string]checkpointMemoEntry, graph packageInputGraph, root, key string, checkpoint plan.VerificationCheckpoint,
) string {
	packages, guards, err := verifierInputs(checkpoint.Verify)
	if err != nil {
		return err.Error()
	}
	for _, guard := range guards {
		if !guardCovered(guard, packages) {
			return fmt.Sprintf("guard %q is outside the named packages", guard)
		}
	}
	imports := make([]string, 0, len(packages))
	for _, relative := range packages {
		importPath, found := graph.importPathForDir(root, relative)
		if !found {
			return fmt.Sprintf("package %q is absent from the input graph", relative)
		}
		imports = append(imports, importPath)
	}
	identities, err := packageInputIdentities(graph, imports)
	if err != nil {
		return err.Error()
	}
	input, err := artifact.JSONID(artifact.KindProfile, struct {
		Policy     string                 `json:"policy"`
		Key        string                 `json:"key"`
		Checkpoint string                 `json:"checkpoint"`
		Verify     string                 `json:"verify"`
		Inputs     map[string]artifact.ID `json:"inputs"`
	}{"checkpoint-memo/v2", key, checkpoint.ID, checkpoint.Verify, identities})
	if err != nil {
		return err.Error()
	}
	// Slot: stable per key and checkpoint, independent of the candidate
	// that fills it; the input above carries the candidate identity.
	slot, err := artifact.JSONID(artifact.KindRecipe, struct {
		Policy     string `json:"policy"`
		Key        string `json:"key"`
		Checkpoint string `json:"checkpoint"`
	}{"checkpoint-memo/v1", key, checkpoint.ID})
	if err != nil {
		return err.Error()
	}
	memos[checkpoint.GateCheckName()] = checkpointMemoEntry{slot: slot, input: input}
	return ""
}

// memoSlot: the stable invocation and memo input the cache keys a checkpoint
// check by; (check, zero, false) for every other check.
func (g *gateContext) memoSlot(check automationcheck.Invocation) (automationcheck.Invocation, artifact.ID, bool) {
	entry, found := g.checkpointMemos[check.Check.Name]
	if !found {
		return check, artifact.ID{}, false
	}
	slot := check
	slot.ID = entry.slot
	return slot, entry.input, true
}
