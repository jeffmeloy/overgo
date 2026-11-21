// Package heldfile holds the files a session reads beside a model by their bytes (adapter-hold): each opened once,
// sharing reading alone, so no handle can write, rename, replace or delete it through its name or any link of it while
// it is held, its identity hashed once through that handle, and both kept while an owner holds the directory it lies
// in (held-release): a picture session its model's, a request the files it names until it is served. A file no owner
// holds is hashed and kept by no one, so a request that ends with no session, and a test that reads a model's files,
// leave nothing a writer cannot replace.
package heldfile

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"overgo/internal/artifact"
)

// Set is the files held for their owners, by their absolute paths. Open, where set, opens a file in place of the open
// that shares reading alone, as a check counting the opens gives it.
type Set struct {
	Open func(path string) (*os.File, error)
	mu   sync.Mutex
	// files are the files held, and owners the directories held for them, each with how many owners hold it, every
	// file's where a directory is "".
	files  map[string]*held
	owners map[string]int
}

// held is one file a Set holds: the handle that keeps it as it was named, and its identity by those bytes.
type held struct {
	file *os.File
	id   artifact.ID
}

// Session is the process's held files, which a picture request's naming (capabilityruntime.Contents) and its reading
// (latentimage) share, held for a picture session from its load to its close and for a request until it is served.
var Session = &Set{}

// Own holds the files under directory, every file where directory is "", from their next naming until the release it
// returns runs: a picture session owns its model's directory from its load to its close, and a request the files it
// names until it is served. A file stays held while any owner's directory holds it, and is let go with the last.
func (s *Set) Own(directory string) (func() error, error) {
	key, err := ownedKey(directory)
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.owners == nil {
		s.owners = map[string]int{}
	}
	s.owners[key]++
	return sync.OnceValue(func() error {
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.owners[key]--; s.owners[key] == 0 {
			delete(s.owners, key)
		}
		var err error
		for path, file := range s.files {
			if !s.owned(path) {
				err = errors.Join(err, file.file.Close())
				delete(s.files, path)
			}
		}
		return err
	}), nil
}

// ownedKey is the key an owned directory is kept under: its absolute path, "" for every file.
func ownedKey(directory string) (string, error) {
	if directory == "" {
		return "", nil
	}
	return filepath.Abs(directory)
}

// owned says an owner holds the directory path lies in, under s.mu.
func (s *Set) owned(path string) bool {
	for directory := range s.owners {
		if directory == "" || path == directory || strings.HasPrefix(path, directory+string(filepath.Separator)) {
			return true
		}
	}
	return false
}

// Identity is the identity by its bytes of the file at path. A file an owner holds (Own) is held from its first naming:
// the first opens it sharing reading alone and hashes it through that handle, and a later one reads nothing. One no
// owner holds is opened, hashed and closed. An absent file holds nothing and is fs.ErrNotExist.
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
	if err != nil || !s.owned(key) {
		return id, errors.Join(err, file.Close())
	}
	if s.files == nil {
		s.files = map[string]*held{}
	}
	s.files[key] = &held{file: file, id: id}
	return id, nil
}
