package codeprofile

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/strictjson"
)

// This is the same item/status projection as the current plan reader, not a
// second live queue. Historical ownership is used only to retain a declaration.
type stagedSurfacePlan struct {
	Lane  string `json:"lane"`
	Scope string `json:"scope"`
	Items []struct {
		ID     string `json:"id"`
		Owner  string `json:"owner"`
		Status string `json:"status"`
		Steps  []struct {
			ID     string `json:"id"`
			Status string `json:"status"`
		} `json:"steps"`
	} `json:"items"`
}

type inheritedStagedSurface struct {
	open        map[string]string
	declaration StagedSurfaceDeclaration
}

func loadInheritedStagedSurface(currentPath, revision string) (inheritedStagedSurface, error) {
	root := filepath.Dir(filepath.Dir(currentPath))
	files, err := gitauthority.AncestorFiles(context.Background(), root, revision, "docs/plan.json", "docs/staged_surface.json")
	if err != nil {
		return inheritedStagedSurface{}, fmt.Errorf("codeprofile: inherited staged authority: %w", err)
	}
	var plan stagedSurfacePlan
	var source inheritedStagedSurface
	// Like the live reader, decode only the plan's ownership/status projection.
	// Validate all raw names first, including fields outside that projection.
	var fields map[string]json.RawMessage
	if err := strictjson.DecodeBytes(files[0], &fields); err != nil {
		return source, err
	}
	if err := json.Unmarshal(files[0], &plan); err != nil {
		return source, err
	}
	if err := strictjson.DecodeBytes(files[1], &source.declaration); err != nil {
		return source, err
	}
	if source.declaration.Version != artifact.SecondDocumentVersion {
		return source, errors.New("codeprofile: inherited staged declaration version differs")
	}
	source.open, err = plan.openSteps()
	return source, err
}

func (live stagedSurfacePlan) containsItem(reference string) bool {
	itemID, _, _ := strings.Cut(reference, "/")
	for _, item := range live.Items {
		if item.ID == itemID {
			return true
		}
	}
	return false
}

func (source inheritedStagedSurface) verify(entry StagedSurfaceEntry, lane string) error {
	owner, open := source.open[entry.RetireWith]
	if !open || owner == "" || owner == lane {
		return fmt.Errorf("codeprofile: inherited retirement %s is not a foreign open step", entry.RetireWith)
	}
	matches := 0
	for _, original := range source.declaration.Staged {
		if original.Package != entry.Package || original.Name != entry.Name {
			continue
		}
		if original.Reason != entry.Reason || original.RetireWith != entry.RetireWith || original.Retained != nil {
			return errors.New("codeprofile: inherited staged declaration differs from its source")
		}
		matches++
	}
	if matches != 1 {
		return errors.New("codeprofile: inherited staged declaration is absent or duplicated")
	}
	return nil
}
