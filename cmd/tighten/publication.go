package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	publicationDirectory = ".overgo-tighten-transaction"
	publicationVersion   = 1
)

var errPublicationInterrupted = errors.New("tighten publication interrupted")

type publicationEntry struct {
	Path       string `json:"path"`
	Original   []byte `json:"original"`
	Updated    []byte `json:"updated,omitempty"`
	OriginalID string `json:"original_id"`
	UpdatedID  string `json:"updated_id,omitempty"`
	Mode       uint32 `json:"mode"`
	Delete     bool   `json:"delete,omitempty"`
}

type publicationJournal struct {
	Version int                `json:"version"`
	Entries []publicationEntry `json:"entries"`
}

type publication struct {
	root, directory string
	journal         publicationJournal
}

func beginPublication(root string, originals, updates map[string][]byte) (*publication, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	directory := filepath.Join(root, publicationDirectory)
	if err := os.Mkdir(directory, 0o700); err != nil {
		return nil, fmt.Errorf("tighten transaction: %w", err)
	}
	failed := true
	defer func() {
		if failed {
			_ = os.RemoveAll(directory)
		}
	}()
	names := make([]string, 0, len(updates))
	for name := range updates {
		names = append(names, name)
	}
	sort.Strings(names)
	journal := publicationJournal{Version: publicationVersion, Entries: make([]publicationEntry, 0, len(names))}
	for _, name := range names {
		path, err := publicationPath(root, name)
		if err != nil {
			return nil, err
		}
		original, ok := originals[name]
		if !ok {
			return nil, fmt.Errorf("tighten transaction: original %s is absent", name)
		}
		entry := publicationEntry{
			Path: name, Original: original, OriginalID: contentIdentity(original), Mode: uint32(fileMode(path)),
		}
		if updated := updates[name]; updated == nil {
			entry.Delete = true
		} else {
			entry.Updated = updated
			entry.UpdatedID = contentIdentity(updated)
		}
		journal.Entries = append(journal.Entries, entry)
	}
	data, err := json.Marshal(journal)
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(directory, "journal.json"), data, 0o600); err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(directory, "ready"), []byte("ready\n"), 0o600); err != nil {
		return nil, err
	}
	failed = false
	return &publication{root: root, directory: directory, journal: journal}, nil
}

func recoverPublication(root string) error {
	root, err := filepath.Abs(root)
	if err != nil {
		return err
	}
	directory := filepath.Join(root, publicationDirectory)
	if _, err := os.Stat(directory); os.IsNotExist(err) {
		return nil
	} else if err != nil {
		return err
	}
	if _, err := os.Stat(filepath.Join(directory, "ready")); os.IsNotExist(err) {
		return os.RemoveAll(directory)
	} else if err != nil {
		return err
	}
	data, err := os.ReadFile(filepath.Join(directory, "journal.json"))
	if err != nil {
		return err
	}
	var journal publicationJournal
	if err := json.Unmarshal(data, &journal); err != nil {
		return fmt.Errorf("tighten transaction: invalid journal: %w", err)
	}
	if journal.Version != publicationVersion {
		return errors.New("tighten transaction: unsupported journal version")
	}
	return (&publication{root: root, directory: directory, journal: journal}).rollback()
}

func (p *publication) publish(stopAfter int) error {
	for index, entry := range p.journal.Entries {
		path, err := publicationPath(p.root, entry.Path)
		if err != nil {
			return err
		}
		current, exists, err := fileIdentity(path)
		if err != nil || exists && current != entry.OriginalID && current != entry.UpdatedID {
			return fmt.Errorf("tighten transaction: %s changed after staging", entry.Path)
		}
		if entry.Delete {
			if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
				return err
			}
		} else {
			if contentIdentity(entry.Updated) != entry.UpdatedID {
				return fmt.Errorf("tighten transaction: staged update %s is corrupt", entry.Path)
			}
			if err := os.WriteFile(path, entry.Updated, os.FileMode(entry.Mode)); err != nil {
				return err
			}
		}
		if stopAfter > 0 && index+1 == stopAfter {
			return errPublicationInterrupted
		}
	}
	return nil
}

func (p *publication) rollback() error {
	var failures []error
	for _, entry := range p.journal.Entries {
		path, err := publicationPath(p.root, entry.Path)
		if err != nil {
			failures = append(failures, err)
			continue
		}
		current, exists, err := fileIdentity(path)
		if err != nil || exists && current != entry.OriginalID && current != entry.UpdatedID || !exists && !entry.Delete {
			failures = append(failures, fmt.Errorf("tighten transaction: refusing to overwrite changed %s", entry.Path))
			continue
		}
		if contentIdentity(entry.Original) != entry.OriginalID {
			err = fmt.Errorf("tighten transaction: staged original %s is corrupt", entry.Path)
		} else {
			err = os.WriteFile(path, entry.Original, os.FileMode(entry.Mode))
		}
		if err != nil {
			failures = append(failures, err)
		}
	}
	if len(failures) == 0 {
		return os.RemoveAll(p.directory)
	}
	return errors.Join(failures...)
}

func (p *publication) finish() error {
	for _, entry := range p.journal.Entries {
		path, err := publicationPath(p.root, entry.Path)
		if err != nil {
			return err
		}
		current, exists, err := fileIdentity(path)
		if err != nil || entry.Delete && exists || !entry.Delete && (!exists || current != entry.UpdatedID) {
			return fmt.Errorf("tighten transaction: published %s differs from staged update", entry.Path)
		}
	}
	return os.RemoveAll(p.directory)
}

func publicationPath(root, name string) (string, error) {
	clean := filepath.Clean(filepath.FromSlash(name))
	if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("tighten transaction: path %q escapes repository", name)
	}
	return filepath.Join(root, clean), nil
}

func fileIdentity(path string) (string, bool, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	return contentIdentity(data), true, nil
}

func contentIdentity(data []byte) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}
