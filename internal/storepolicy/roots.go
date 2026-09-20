package storepolicy

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/plan"
)

// identityPattern matches a content-addressed identity as committed
// documents and commit messages spell it.
var identityPattern = regexp.MustCompile(`[a-z][a-z0-9-]*:sha256:[0-9a-f]{64}`)

// CheckoutRoots collects every store identity a checkout pins without an
// alias: the ones its committed JSON documents name (docs/**/*.json and the
// root compatibility.json) and the ones named by its commit messages after
// the proof horizon -- the gate's Overgo-* trailers and any identity a body
// cites. The plan completion authority proves every landing after the
// horizon from the latter, so both root a store's live set. The counts
// report each source before the union.
func CheckoutRoots(ctx context.Context, checkout string) (roots []artifact.ID, documents, messages int, err error) {
	var files []string
	if err := filepath.WalkDir(filepath.Join(checkout, "docs"), func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && strings.HasSuffix(path, ".json") {
			files = append(files, path)
		}
		return nil
	}); err != nil {
		return nil, 0, 0, err
	}
	var committed []byte
	for _, path := range append(files, filepath.Join(checkout, "compatibility.json")) {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, 0, 0, err
		}
		committed = append(append(committed, data...), '\n')
	}
	// At or before a committed proof horizon the completion authority takes
	// landings from Git alone, so their messages pin nothing: only the
	// history after it roots evidence. The receipt is read from HEAD's tree,
	// as the authority reads it, so an uncommitted one releases nothing; no
	// receipt means the whole history.
	revisions := "HEAD"
	var horizon plan.ProofHorizon
	if receipt, err := gitauthority.Query(ctx, checkout, "show", "HEAD:"+plan.ProofHorizonPath); err == nil && json.Unmarshal(receipt, &horizon) == nil {
		revisions = horizon.Commit + "..HEAD"
	}
	history, err := gitauthority.Query(ctx, checkout, "log", "--format=%B", revisions)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("git log for commit-message roots: %w", err)
	}
	fromDocuments, fromMessages := identities(committed), identities(history)
	roots = slices.Concat(fromDocuments, fromMessages)
	slices.SortFunc(roots, artifact.CompareID)
	return slices.Compact(roots), len(fromDocuments), len(fromMessages), nil
}

// identities returns every identity the text names, once each, sorted. A
// match is at least one byte, so the text's length bounds their number.
func identities(text []byte) []artifact.ID {
	seen := map[artifact.ID]bool{}
	var found []artifact.ID
	for _, match := range identityPattern.FindAll(text, len(text)) {
		if id, err := artifact.ParseID(string(match)); err == nil && !seen[id] {
			seen[id] = true
			found = append(found, id)
		}
	}
	slices.SortFunc(found, artifact.CompareID)
	return found
}
