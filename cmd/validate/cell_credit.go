package main

import (
	"cmp"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
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

// consumedFiles says, without reading them, what the files a cell's command is given and its
// weights are: the path, size and time of each, and of every file under a directory. The
// catalog checks by content only what a manifest lists, so this tells a location-only model's
// replaced weights, or a changed file beside a model, from those a run measured; a change that
// keeps a file's size and time is not seen. A relative argument is a committed file, which the
// credit's commit fixes. A path that cannot be read, a remote model's address among them, gives a
// value no later binding repeats: unknown bytes are never credited. A cell with no command has none.
// listed keeps what one binding found each path to be, so a path many cells are given is read once.
func consumedFiles(cell ModelValidation, listed map[string]string) string {
	_, given, err := reacquireCommand(cell, "")
	if err != nil {
		return ""
	}
	found := sha256.New()
	for _, path := range append(given, cell.Location) {
		if path != cell.Location && !filepath.IsAbs(path) {
			continue
		}
		if _, read := listed[path]; !read {
			under := sha256.New()
			listed[path] = rand.Text()
			// Walk reports a link as the link; Stat reads the file it leads to.
			if filepath.Walk(path, func(file string, _ fs.FileInfo, walkErr error) error {
				info, err := os.Stat(file)
				if err = cmp.Or(walkErr, err); err == nil && !info.IsDir() {
					fmt.Fprintln(under, file, info.Size(), info.ModTime().UnixNano())
				}
				return err
			}) == nil {
				listed[path] = hex.EncodeToString(under.Sum(nil))
			}
		}
		fmt.Fprintln(found, listed[path])
	}
	return hex.EncodeToString(found.Sum(nil))
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
