// Package gosyntax is the parsed Go source the analyzers share: a snapshot
// of repository files whose identical contents parse once. Provenance and
// build selection stay in gosource, which reads without parsing.
package gosyntax

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"overgo/internal/gosource"
)

// File binds a repository Go path to its content. Files with identical
// content share one parse.
type File struct {
	Path, ContentID string
	Test            bool
	Data            []byte `json:"-"`
	parsed          *parsed
}

// parsed holds one content's syntax and the file set whose line table the
// parser filled while scanning it.
type parsed struct {
	files  *token.FileSet
	syntax func() (*ast.File, error)
}

func parse(path string, data []byte) *parsed {
	files := token.NewFileSet()
	return &parsed{files: files, syntax: sync.OnceValues(func() (*ast.File, error) {
		return parser.ParseFile(files, path, data, parser.ParseComments)
	})}
}

func newFile(path, contentID string, data []byte, shared *parsed) File {
	return File{
		Path: path, ContentID: contentID, Data: data, parsed: shared,
		Test: strings.HasSuffix(path, "_test.go") || hasPathPart(path, "testdata"),
	}
}

// Syntax parses the file once per content, with comments.
func (f File) Syntax() (*ast.File, error) {
	return f.parsed.syntax()
}

// Generated reports a file under a generated directory or carrying the
// standard generated-code header.
func (f File) Generated() (bool, error) {
	syntax, err := f.Syntax()
	return err == nil && (hasPathPart(f.Path, "generated") || ast.IsGenerated(syntax)), err
}

// Line maps a position from Syntax to its physical line, ignoring //line
// directives; a position outside the file maps to zero.
func (f File) Line(pos token.Pos) int {
	return f.parsed.files.PositionFor(pos, false).Line
}

// Snapshot is an ordered set of parsed repository Go files.
type Snapshot struct{ Files []File }

// Identity is the gosource provenance of the same ordered paths and contents.
func (s Snapshot) Identity() string {
	files := make([]gosource.File, len(s.Files))
	for index, file := range s.Files {
		files[index] = gosource.File{Path: file.Path, ContentID: file.ContentID}
	}
	return (gosource.Snapshot{Files: files}).Identity()
}

// Overlay returns a snapshot with repository-relative source replacements.
// A nil value deletes the path; input snapshots remain immutable.
func (s Snapshot) Overlay(contents map[string][]byte) (Snapshot, error) {
	files := make(map[string]File, len(s.Files)+len(contents))
	for _, file := range s.Files {
		files[file.Path] = file
	}
	for name, data := range contents {
		path := filepath.Clean(filepath.FromSlash(name))
		if !filepath.IsLocal(path) || !strings.HasSuffix(path, ".go") {
			return Snapshot{}, fmt.Errorf("repository source path %q escapes root or is not Go", name)
		}
		path = filepath.ToSlash(path)
		if data == nil {
			delete(files, path)
			continue
		}
		digest := sha256.Sum256(data)
		data = slices.Clone(data)
		files[path] = newFile(path, hex.EncodeToString(digest[:]), data, parse(path, data))
	}
	out := Snapshot{Files: make([]File, 0, len(files))}
	for _, file := range files {
		out.Files = append(out.Files, file)
	}
	slices.SortFunc(out.Files, func(a, b File) int { return strings.Compare(a.Path, b.Path) })
	return out, nil
}

// LoadGo reads the requested Go files through gosource; absent files are
// omitted for overlays.
func LoadGo(root string, relatives []string) (Snapshot, error) {
	source, err := gosource.LoadGo(root, relatives)
	if err != nil {
		return Snapshot{}, err
	}
	shared := map[string]*parsed{}
	files := make([]File, 0, len(source.Files))
	for _, file := range source.Files {
		cached, found := shared[file.ContentID]
		if !found {
			cached = parse(file.Path, file.Data)
			shared[file.ContentID] = cached
		}
		files = append(files, newFile(file.Path, file.ContentID, file.Data, cached))
	}
	return Snapshot{Files: files}, nil
}

// DiscoverGo loads every Go file under the repository-relative tops; an
// absent top contributes nothing.
func DiscoverGo(root string, tops ...string) (Snapshot, error) {
	var paths []string
	for _, top := range tops {
		err := filepath.WalkDir(filepath.Join(root, filepath.FromSlash(top)), func(path string, entry os.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(path, ".go") {
				return err
			}
			relative, err := filepath.Rel(root, path)
			if err == nil {
				paths = append(paths, filepath.ToSlash(relative))
			}
			return err
		})
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return Snapshot{}, err
		}
	}
	return LoadGo(root, paths)
}

func hasPathPart(path, want string) bool {
	return slices.Contains(strings.Split(filepath.ToSlash(path), "/"), want)
}
