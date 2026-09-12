package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/codeprofile"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
)

// catalogText is the reviewed understanding one catalogue command binds to
// every candidate it names; the coordinates come from the scan.
type catalogText struct {
	tier, understanding, closurePath, rerankTrigger string
}

func (text catalogText) validate() error {
	for _, field := range []struct{ name, value string }{
		{"tier", text.tier}, {"understanding", text.understanding}, {"closure-path", text.closurePath}, {"rerank-trigger", text.rerankTrigger},
	} {
		if strings.TrimSpace(field.value) == "" {
			return fmt.Errorf("catalog: -%s is required", field.name)
		}
	}
	return nil
}

// catalogCandidates catalogues the named uncatalogued candidates in one
// store transaction: their coordinates are proposed from the scan, their
// documents bound under the reviewed text, and the commit carries them
// beside whatever drifted binding the same scan rebinds. The count is the
// documents catalogued; rebound counts the drifted documents committed
// with them.
func catalogCandidates(root, storePath string, snapshot repoanalysis.SourceSnapshot, names []string, text catalogText) (catalogued, rebound int, err error) {
	if err := text.validate(); err != nil {
		return 0, 0, err
	}
	if len(names) == 0 {
		return 0, 0, errors.New("catalog: no candidate named")
	}
	// A writer open creates the store a first catalogue lands in.
	ctx := context.Background()
	store, err := overgodb.OpenContext(ctx, filepath.Join(root, storePath))
	if err != nil {
		return 0, 0, err
	}
	report, proposeErr := buildUnclassifiedPolicyReport(ctx, snapshot, store)
	if err := errors.Join(proposeErr, store.Close()); err != nil {
		return 0, 0, err
	}
	var fresh []string
	for _, name := range names {
		index := slices.IndexFunc(report.Candidates, func(candidate unclassifiedCandidate) bool { return candidate.Current.Name == name })
		if index >= 0 && report.Candidates[index].ActiveRebind {
			continue
		}
		fresh = append(fresh, name)
	}
	proposal, err := proposeTriageRowsFromReport(report, fresh)
	if err != nil {
		return 0, 0, err
	}
	for index := range proposal.Rows {
		row := &proposal.Rows[index]
		row.Tier, row.Status = text.tier, string(closureledger.StatusClosed)
		row.Understanding, row.ClosurePath, row.RerankTrigger = text.understanding, text.closurePath, text.rerankTrigger
	}
	candidates := report.review.candidates
	var documents []closureledger.Document
	if len(proposal.Rows) != 0 {
		documents, err = triageDocuments(proposal, candidates)
		if err != nil {
			return 0, 0, err
		}
	}
	rebound, _, _, _, err = importClosureDocumentsWith(root, storePath, filepath.Join(root, storePath), snapshot, false, false, documents, report.review)
	if err != nil {
		return 0, 0, err
	}
	return len(documents), rebound - len(documents), nil
}

// triageDocuments binds every triage row to the candidate the scan found at
// its exact coordinates; a row the scan no longer finds is stale.
func triageDocuments(triage triageFile, candidates []closurescan.Candidate) ([]closureledger.Document, error) {
	if len(triage.Rows) == 0 {
		return nil, errors.New("triage file has no rows")
	}
	byKey := map[string]closurescan.Candidate{}
	for _, row := range candidates {
		byKey[row.DeclarationKey()] = row
		byKey[row.LegacyDeclarationKey()] = row
	}
	documents := make([]closureledger.Document, 0, len(triage.Rows))
	for _, row := range triage.Rows {
		found, ok := byKey[(closurescan.Candidate{
			Kind: row.Kind, File: row.File, Scope: row.Scope, Line: row.Line, Name: row.Name,
		}).DeclarationKey()]
		if !ok {
			return nil, fmt.Errorf("triage row %s not found by scan in %s (stale triage?)", row.Name, row.File)
		}
		binding, err := found.Binding()
		if err != nil {
			return nil, err
		}
		document, err := closureledger.New(
			row.Name, found.ValueJSON(),
			closureledger.Tier(row.Tier), closureledger.Status(row.Status),
			row.Understanding, []closureledger.SourceBinding{binding},
			row.ClosurePath, row.RerankTrigger, binding.Owner,
		)
		if err != nil {
			return nil, fmt.Errorf("row %s: %w", row.Name, err)
		}
		documents = append(documents, document)
	}
	return documents, nil
}

// stageSurface declares new unconsumed surface from the gate's own report
// of it, "file:Name=class" entries, as staged entries retired with one open
// plan step; entries already staged are kept. The written declaration is
// reloaded under the same validation the gate applies.
func stageSurface(root, candidates, reason, retireWith string) (added int, err error) {
	if strings.TrimSpace(reason) == "" || strings.TrimSpace(retireWith) == "" {
		return 0, errors.New("stage: -stage-reason and -retire-with are required")
	}
	module, err := modulePath(root)
	if err != nil {
		return 0, err
	}
	path := filepath.Join(root, "docs", "staged_surface.json")
	declaration, err := codeprofile.LoadStagedSurface(path)
	if err != nil {
		return 0, err
	}
	for token := range strings.SplitSeq(candidates, ",") {
		token = strings.TrimSpace(token)
		if token == "" {
			continue
		}
		file, rest, found := strings.Cut(token, ":")
		name, _, _ := strings.Cut(rest, "=")
		if !found || file == "" || name == "" {
			return 0, fmt.Errorf("stage: %q is not a file:Name=class candidate", token)
		}
		pkg := module + "/" + filepath.ToSlash(filepath.Dir(file))
		if slices.ContainsFunc(declaration.Staged, func(entry codeprofile.StagedSurfaceEntry) bool {
			return entry.Package == pkg && entry.Name == name
		}) {
			continue
		}
		declaration.Staged = append(declaration.Staged, codeprofile.StagedSurfaceEntry{Package: pkg, Name: name, Reason: reason, RetireWith: retireWith})
		added++
	}
	if added == 0 {
		return 0, nil
	}
	if declaration.Version == 0 {
		declaration.Version = artifact.SecondDocumentVersion
	}
	// The declaration keeps its own permissions; a repository holds one.
	info, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	if err := jsonfile.Write(path, declaration, info.Mode().Perm()); err != nil {
		return 0, err
	}
	if _, err := codeprofile.LoadStagedSurface(path); err != nil {
		return 0, fmt.Errorf("stage: the written declaration does not validate: %w", err)
	}
	return added, nil
}

// modulePath reads the module path from the root's go.mod.
func modulePath(root string) (string, error) {
	file, err := os.Open(filepath.Join(root, "go.mod"))
	if err != nil {
		return "", err
	}
	defer file.Close()
	lines := bufio.NewScanner(file)
	for lines.Scan() {
		if module, found := strings.CutPrefix(strings.TrimSpace(lines.Text()), "module "); found {
			return strings.TrimSpace(module), nil
		}
	}
	return "", errors.Join(errors.New("go.mod names no module"), lines.Err())
}
