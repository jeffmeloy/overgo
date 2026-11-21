// Package heldfile holds the files a session reads beside a model by their bytes (adapter-hold): each opened once,
// sharing reading alone, so no handle can write, rename, replace or delete it through its name or any link of it while
// it is held, its identity hashed once through that handle, and both kept until the session lets the file go.
package heldfile

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"overgo/internal/artifact"
)

// Set is the files a session holds, by their absolute paths. Open, where set, opens a file in place of the open that
// shares reading alone, as a check counting the opens gives it.
type Set struct {
	Open  func(path string) (*os.File, error)
	mu    sync.Mutex
	files map[string]*held
}

// held is one file a Set holds: the handle that keeps it as it was named, and its identity by those bytes.
type held struct {
	file *os.File
	id   artifact.ID
}

// Session is the process's held files, which a picture request's naming (capabilityruntime.Contents) and its reading
// (latentimage) share, and which a picture session's close releases under its model.
var Session = &Set{}

// Identity is the identity by its bytes of the file at path, held from its first naming: the first opens it sharing
// reading alone and hashes it through that handle, and a later one reads nothing. An absent file holds nothing and is
// fs.ErrNotExist.
func (s *Set) Identity(path string) (artifact.ID, error) {
	key, err := filepath.Abs(path)
	if err != nil {
		return artifact.ID{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if file, ok := s.files[key]; ok {
		return file.id, nil
	}
	opener := s.Open
	if opener == nil {
		opener = open
	}
	file, err := opener(key)
	if err != nil {
		return artifact.ID{}, err
	}
	id, _, err := artifact.Identify(artifact.KindFile, file)
	if err != nil {
		return artifact.ID{}, errors.Join(err, file.Close())
	}
	if s.files == nil {
		s.files = map[string]*held{}
	}
	s.files[key] = &held{file: file, id: id}
	return id, nil
}

// Release lets go the files held under directory, every file where directory is empty, so another writer may replace
// them and the next naming reads them again.
func (s *Set) Release(directory string) error {
	var root string
	if directory != "" {
		absolute, err := filepath.Abs(directory)
		if err != nil {
			return err
		}
		root = absolute
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var err error
	for key, file := range s.files {
		if root == "" || key == root || strings.HasPrefix(key, root+string(filepath.Separator)) {
			err = errors.Join(err, file.file.Close())
			delete(s.files, key)
		}
	}
	return err
}
