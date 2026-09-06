package gate

import (
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

// verifyPackages: package arguments of the first go test in verify, up to the
// first flag or command separator; "." names the module root.
func verifyPackages(verify string) []string {
	fields := strings.Fields(verify)
	var packages []string
	for index := 0; index+1 < len(fields); index++ {
		if fields[index] != "go" || fields[index+1] != "test" {
			continue
		}
		for _, field := range fields[index+2:] {
			if strings.HasPrefix(field, "-") || field == "&&" || field == "||" || field == ";" {
				break
			}
			packages = append(packages, field)
		}
		break
	}
	return packages
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

// checkpointMemoInputs: per checkpoint, a stable slot and a memo input from
// the key, the verify text and the transitive input identity of the verify's
// packages; a verify naming no package or one absent from the graph fails.
func (g *gateContext) checkpointMemoInputs(batch *plan.VerificationBatch) (map[string]checkpointMemoEntry, error) {
	if batch == nil {
		return nil, nil
	}
	key := checkpointMemoKey(g.planRef, batch)
	graph, err := g.inputGraph()
	if err != nil {
		return nil, err
	}
	root, err := filepath.Abs(g.repo)
	if err != nil {
		return nil, err
	}
	memos := make(map[string]checkpointMemoEntry, len(batch.Checkpoints))
	for _, checkpoint := range batch.Checkpoints {
		packages := verifyPackages(checkpoint.Verify)
		if len(packages) == 0 {
			return nil, fmt.Errorf("checkpoint %s names no package", checkpoint.ID)
		}
		imports := make([]string, 0, len(packages))
		for _, relative := range packages {
			importPath, found := graph.importPathForDir(root, relative)
			if !found {
				return nil, fmt.Errorf("checkpoint %s: package %q is absent from the input graph", checkpoint.ID, relative)
			}
			imports = append(imports, importPath)
		}
		identities, err := packageInputIdentities(graph, imports)
		if err != nil {
			return nil, err
		}
		input, err := artifact.JSONID(artifact.KindProfile, struct {
			Policy     string                 `json:"policy"`
			Key        string                 `json:"key"`
			Checkpoint string                 `json:"checkpoint"`
			Verify     string                 `json:"verify"`
			Inputs     map[string]artifact.ID `json:"inputs"`
		}{"checkpoint-memo/v1", key, checkpoint.ID, checkpoint.Verify, identities})
		if err != nil {
			return nil, err
		}
		// Slot: stable per key and checkpoint, independent of the candidate
		// that fills it; the input above carries the candidate identity.
		slot, err := artifact.JSONID(artifact.KindRecipe, struct {
			Policy     string `json:"policy"`
			Key        string `json:"key"`
			Checkpoint string `json:"checkpoint"`
		}{"checkpoint-memo/v1", key, checkpoint.ID})
		if err != nil {
			return nil, err
		}
		memos[checkpoint.GateCheckName()] = checkpointMemoEntry{slot: slot, input: input}
	}
	return memos, nil
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
