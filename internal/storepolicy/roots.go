package storepolicy

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"unicode"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/plan"
)

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
	fromDocuments, fromMessages := Identities(committed), Identities(history)
	roots = slices.Concat(fromDocuments, fromMessages)
	slices.SortFunc(roots, artifact.CompareID)
	return slices.Compact(roots), len(fromDocuments), len(fromMessages), nil
}

// Identities returns every content-addressed identity the text names, as
// documents and commit messages spell them, once each, sorted: the algorithm
// marker locates a candidate, its kind runs back to the leftmost letter, and
// artifact.ParseID, which owns the syntax, decides it.
func Identities(text []byte) []artifact.ID {
	const marker = ":sha256:"
	kindRune := func(r rune) bool { return r == '-' || unicode.IsLower(r) || unicode.IsDigit(r) }
	var found []artifact.ID
	var before []byte
	for part := range bytes.SplitSeq(text, []byte(marker)) {
		kind := bytes.TrimLeftFunc(before[len(bytes.TrimRightFunc(before, kindRune)):], func(r rune) bool { return !unicode.IsLower(r) })
		digest := part[:min(len(part), hex.EncodedLen(sha256.Size))]
		if id, err := artifact.ParseID(string(kind) + marker + string(digest)); err == nil {
			found = append(found, id)
		}
		before = part
	}
	slices.SortFunc(found, artifact.CompareID)
	return slices.Compact(found)
}
