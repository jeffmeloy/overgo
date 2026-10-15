package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/pathidentity"
)

// cellCreditMode keeps a cycle's credit private to the checkout's owner.
const cellCreditMode = 0o600

// cellCredit is the cells acquired for one commit and store, kept in a file so a killed run's
// work survives it. A credit for another commit or store is none: the file starts over.
type cellCredit struct {
	path   string
	Commit string   `json:"commit"`
	Store  string   `json:"store"`
	Cells  []string `json:"cells"`
	credit map[string]bool
}

// openCellCredit reads the credit at path for commit and store; an absent path keeps none, and a
// file kept for another commit or store, or one that cannot be read, holds none.
func openCellCredit(path, commit, store string) *cellCredit {
	credit := &cellCredit{path: path, Commit: commit, Store: store, credit: map[string]bool{}}
	if path == "" {
		return credit
	}
	var kept cellCredit
	if err := jsonfile.DecodeStrict(path, &kept); err == nil && kept.Commit == commit && kept.Store == store {
		credit.Cells = kept.Cells
		for _, cell := range kept.Cells {
			credit.credit[cell] = true
		}
	}
	return credit
}

// cellKey names a cell by what a run of it consumes: its model, task, validation, protocol,
// recipe, weights, the committed inputs its training and projection runs read and what the
// files its command is given were when it was bound, and not by what a run changes (the cost
// it measured, why it is scheduled), so a changed input is a different cell and a measured one is the same.
func cellKey(cell ModelValidation) string {
	consumed, err := json.Marshal([]any{
		cell.Model, cell.Kind, cell.Validation, cell.Recipe, cell.Protocol, cell.Location, cell.Directory,
		cell.Grounding, cell.Objective, cell.Conditioning, cell.Projector, cell.Consumed,
	})
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(consumed)
	return hex.EncodeToString(sum[:])
}

// consumedFiles qualifies resume credit by declared model components and explicit file inputs.
//   - Every component is read through artifact identity once per binding
//   - A directory containing undeclared files or directory links authorizes no credit
//   - Missing content or execution authority authorizes no credit
//   - Relative command inputs belong to the pinned commit
func consumedFiles(ctx context.Context, store *overgodb.Store, cell ModelValidation, listed map[string]string, execution string) string {
	_, given, err := reacquireCommand(cell, "")
	if err != nil {
		return ""
	}
	if store == nil || execution == "" {
		return rand.Text()
	}
	manifest, found, err := store.Manifest(ctx, cell.Model)
	if err != nil || !found {
		return rand.Text()
	}
	inputs := map[string]string{}
	for _, component := range manifest.Components {
		path, err := artifact.AvailablePath(ctx, store, component.Artifact, artifact.LocationFile)
		if err != nil {
			return rand.Text()
		}
		inputs[path] = component.Artifact.DigestHex()
	}
	for _, path := range append(given, cell.Location) {
		if path != cell.Location && !filepath.IsAbs(path) {
			continue
		}
		info, err := os.Stat(path)
		if err != nil {
			return rand.Text()
		}
		if info.IsDir() {
			err = filepath.Walk(path, func(file string, entry fs.FileInfo, walkErr error) error {
				if walkErr != nil {
					return walkErr
				}
				if entry.IsDir() {
					return nil
				}
				target, err := os.Stat(file)
				if err != nil || !target.Mode().IsRegular() {
					return errors.New("consumed input has no regular file authority")
				}
				canonical, err := pathidentity.Canonical(file)
				if err != nil || inputs[canonical] == "" {
					return errors.New("consumed directory has an undeclared input")
				}
				return nil
			})
			if err != nil {
				return rand.Text()
			}
		} else {
			if !info.Mode().IsRegular() {
				return rand.Text()
			}
			canonical, err := pathidentity.Canonical(path)
			if err != nil {
				return rand.Text()
			}
			if _, known := inputs[canonical]; !known {
				inputs[canonical] = ""
			}
		}
	}
	paths := make([]string, 0, len(inputs))
	for path := range inputs {
		paths = append(paths, path)
	}
	slices.Sort(paths)
	named := sha256.New()
	fmt.Fprintln(named, execution)
	for _, path := range paths {
		if ctx.Err() != nil {
			return rand.Text()
		}
		identity, read := listed[path]
		if !read {
			content, err := artifact.IdentifyFile(artifact.KindFile, path)
			if err != nil {
				identity = rand.Text()
			} else {
				identity = content.DigestHex()
			}
			listed[path] = identity
		}
		if expected := inputs[path]; expected != "" && identity != expected {
			return rand.Text()
		}
		fmt.Fprintln(named, path, identity)
	}
	return hex.EncodeToString(named.Sum(nil))
}

// has says the cell was acquired by an earlier run of this commit and store.
func (credit *cellCredit) has(cell ModelValidation) bool {
	key := cellKey(cell)
	return key != "" && credit.credit[key]
}

// add records the cell as acquired and keeps it; with no file the credit is not kept.
func (credit *cellCredit) add(cell ModelValidation) error {
	key := cellKey(cell)
	if key == "" || credit.credit[key] || credit.path == "" {
		return nil
	}
	credit.credit[key] = true
	credit.Cells = append(credit.Cells, key)
	return jsonfile.Write(credit.path, credit, cellCreditMode)
}

// cycleCellCredit is the credit of this run: the file the cycle named for the commit root holds
// and the store the cells go to, or none when no file is named or the commit cannot be read.
func cycleCellCredit(ctx context.Context, root, storeRoot string) *cellCredit {
	path := os.Getenv(clioptions.CycleCreditEnvironment)
	head, err := gitauthority.Query(ctx, root, "rev-parse", "HEAD")
	if path == "" || err != nil {
		return openCellCredit("", "", "")
	}
	store, err := filepath.Abs(storeRoot)
	if err != nil {
		return openCellCredit("", "", "")
	}
	return openCellCredit(path, strings.TrimSpace(string(head)), store)
}
