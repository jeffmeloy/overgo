package repoanalysis

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

type sourceBlob struct {
	data   []byte
	once   sync.Once
	syntax *ast.File
	err    error
}

type GoFile struct {
	Path      string
	ContentID string
	Test      bool
	blob      *sourceBlob
}

func (f GoFile) Syntax() (*ast.File, error) {
	f.blob.once.Do(func() {
		f.blob.syntax, f.blob.err = parser.ParseFile(token.NewFileSet(), f.Path, f.blob.data, parser.ParseComments)
	})
	return f.blob.syntax, f.blob.err
}

func (f GoFile) Generated() (bool, error) {
	syntax, err := f.Syntax()
	return err == nil && (hasPathPart(f.Path, "generated") || ast.IsGenerated(syntax)), err
}

type SourceSnapshot struct{ Files []GoFile }

// Overlay returns a snapshot with repository-relative source replacements.
// A nil value deletes the path; input snapshots remain immutable.
func (s SourceSnapshot) Overlay(contents map[string][]byte) (SourceSnapshot, error) {
	files := make(map[string]GoFile, len(s.Files)+len(contents))
	for _, file := range s.Files {
		files[file.Path] = file
	}
	for name, data := range contents {
		path := filepath.Clean(filepath.FromSlash(name))
		if filepath.IsAbs(path) || path == ".." || strings.HasPrefix(path, ".."+string(filepath.Separator)) || !strings.HasSuffix(path, ".go") {
			return SourceSnapshot{}, fmt.Errorf("repository source path %q escapes root or is not Go", name)
		}
		path = filepath.ToSlash(path)
		if data == nil {
			delete(files, path)
			continue
		}
		digest := sha256.Sum256(data)
		files[path] = GoFile{
			Path: path, ContentID: hex.EncodeToString(digest[:]), blob: &sourceBlob{data: append([]byte(nil), data...)},
			Test: strings.HasSuffix(path, "_test.go") || hasPathPart(path, "testdata"),
		}
	}
	out := SourceSnapshot{Files: make([]GoFile, 0, len(files))}
	for _, file := range files {
		out.Files = append(out.Files, file)
	}
	sort.Slice(out.Files, func(i, j int) bool { return out.Files[i].Path < out.Files[j].Path })
	return out, nil
}

func LoadGo(root string, relatives []string) (SourceSnapshot, error) {
	paths := append([]string(nil), relatives...)
	for index, name := range paths {
		path := filepath.Clean(filepath.FromSlash(name))
		if filepath.IsAbs(path) || path == ".." || strings.HasPrefix(path, ".."+string(filepath.Separator)) {
			return SourceSnapshot{}, fmt.Errorf("repository source path %q escapes root", name)
		}
		paths[index] = filepath.ToSlash(path)
	}
	sort.Strings(paths)
	blobs := map[[sha256.Size]byte]*sourceBlob{}
	files := make([]GoFile, 0, len(paths))
	for index, relative := range paths {
		if (index > 0 && relative == paths[index-1]) || !strings.HasSuffix(relative, ".go") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(relative)))
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return SourceSnapshot{}, err
		}
		digest := sha256.Sum256(data)
		blob := blobs[digest]
		if blob == nil {
			blob = &sourceBlob{data: data}
			blobs[digest] = blob
		}
		files = append(files, GoFile{
			Path: relative, ContentID: hex.EncodeToString(digest[:]), blob: blob,
			Test: strings.HasSuffix(relative, "_test.go") || hasPathPart(relative, "testdata"),
		})
	}
	return SourceSnapshot{Files: files}, nil
}

func DiscoverGo(root string, tops ...string) (SourceSnapshot, error) {
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
			return SourceSnapshot{}, err
		}
	}
	return LoadGo(root, paths)
}

func hasPathPart(path, want string) bool {
	for _, part := range strings.Split(filepath.ToSlash(path), "/") {
		if part == want {
			return true
		}
	}
	return false
}
