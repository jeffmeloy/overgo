package landing

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/repoanalysis"
)

// gateWrote says the gate's own stages wrote path into the checkout after the landing's first capture: gofmt or go fix
// (recordWrites), a rename's citations or a lowered ceiling (noteWrite), or a generator (derived).
func (l *landing) gateWrote(path string) bool {
	return l.wrote[path] || slices.Contains(l.derived, path)
}

// noteWrite keeps that the gate wrote path, from the checkout's root, into the checkout.
func (l *landing) noteWrite(path string) {
	if l.wrote == nil {
		l.wrote = map[string]bool{}
	}
	l.wrote[filepath.ToSlash(path)] = true
}

// sinceFirst keeps of a later capture's changed paths those the first capture held or the gate's own stages wrote.
// Any other was changed after the first capture by something else: it is left in the checkout for its own step and
// named once (landing-snapshot, SQA-494).
func (l *landing) sinceFirst(entries []repoanalysis.DirtyPath) []repoanalysis.DirtyPath {
	owned := func(path string) bool { return path != "" && (slices.Contains(l.written, path) || l.gateWrote(path)) }
	var kept []repoanalysis.DirtyPath
	for _, entry := range entries {
		if owned(entry.Path) || owned(entry.OriginalPath) {
			kept = append(kept, entry)
			continue
		}
		if !slices.Contains(l.later, entry.Path) {
			l.later = append(l.later, entry.Path)
		}
	}
	if len(l.later) > 0 && !l.laterNamed {
		fmt.Fprintf(l.Log, "land: changed after this landing's first capture, so left in the checkout for its own step: %s\n", strings.Join(l.later, ", "))
		l.laterNamed = true
	}
	return kept
}

// recordWrites runs stage, which rewrites whole packages in the checkout (gofmt, go fix), and keeps each Go file of
// directories, as goPackages names them, whose content it changed as one the gate wrote (noteWrite).
func (l *landing) recordWrites(directories []string, stage func() error) error {
	before := l.goFiles(directories)
	if err := stage(); err != nil {
		return err
	}
	for name, sum := range l.goFiles(directories) {
		if held, found := before[name]; !found || held != sum {
			l.noteWrite(name)
		}
	}
	return nil
}

// goFiles are the digests of the Go files directories hold in the checkout, by path from its root; a file that cannot
// be read is left out.
func (l *landing) goFiles(directories []string) map[string][sha256.Size]byte {
	sums := map[string][sha256.Size]byte{}
	for _, directory := range directories {
		names, err := filepath.Glob(filepath.Join(l.Repo, filepath.FromSlash(strings.TrimPrefix(directory, "./")), "*.go"))
		if err != nil {
			continue
		}
		for _, name := range names {
			data, err := os.ReadFile(name)
			if err != nil {
				continue
			}
			if relative, err := filepath.Rel(l.Repo, name); err == nil {
				sums[filepath.ToSlash(relative)] = sha256.Sum256(data)
			}
		}
	}
	return sums
}
