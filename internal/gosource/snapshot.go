package gosource

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// File binds a repository path to its content, independent of syntax analysis.
type File struct {
	Path, ContentID string
	Data            []byte `json:"-"`
}

// Snapshot retains ordered source bytes and their provenance.
type Snapshot struct{ Files []File }

// Identity preserves the ordered path/NUL/content/NUL provenance grammar.
func (s Snapshot) Identity() string {
	hash := sha256.New()
	for _, file := range s.Files {
		hash.Write([]byte(file.Path))
		hash.Write([]byte{0})
		hash.Write([]byte(file.ContentID))
		hash.Write([]byte{0})
	}
	return hex.EncodeToString(hash.Sum(nil))
}

// LoadGo reads the requested Go files; absent files are omitted for overlays.
func LoadGo(root string, relatives []string) (Snapshot, error) {
	paths := slices.Clone(relatives)
	for index, name := range paths {
		path := filepath.Clean(filepath.FromSlash(name))
		if !filepath.IsLocal(path) {
			return Snapshot{}, fmt.Errorf("repository source path %q escapes root", name)
		}
		paths[index] = filepath.ToSlash(path)
	}
	slices.Sort(paths)
	blobs := map[[sha256.Size]byte][]byte{}
	files := make([]File, 0, len(paths))
	for index, relative := range paths {
		if (index > 0 && relative == paths[index-1]) || !strings.HasSuffix(relative, ".go") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(relative)))
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return Snapshot{}, err
		}
		digest := sha256.Sum256(data)
		if prior, found := blobs[digest]; found {
			data = prior
		} else {
			blobs[digest] = data
		}
		files = append(files, File{Path: relative, ContentID: hex.EncodeToString(digest[:]), Data: data})
	}
	return Snapshot{Files: files}, nil
}
