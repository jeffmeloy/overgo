package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
)

// referenceSpec is the -declare-references input: published or
// externally measured results per servable model location.
type referenceSpec struct {
	Declarations []struct {
		Model      string                      `json:"model"`
		References []evaluation.ReferenceScore `json:"references"`
	} `json:"declarations"`
}

// declareReferenceScores binds published results to models in the
// store, resolving each model through the servable listing so the
// declaration keys the same identity the report and the picker read.
func declareReferenceScores(ctx context.Context, repository, specPath string, limit int) error {
	data, err := os.ReadFile(specPath)
	if err != nil {
		return err
	}
	var spec referenceSpec
	if err := json.Unmarshal(data, &spec); err != nil {
		return fmt.Errorf("evaluate: reference spec %q: %w", specPath, err)
	}
	if len(spec.Declarations) == 0 {
		return errors.New("evaluate: reference spec declares nothing")
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	memo := discovery.LoadMemo(ctx, store)
	entries, err := discovery.ServableWithMemo(ctx, store, limit, memo)
	if err != nil {
		return err
	}
	for _, declaration := range spec.Declarations {
		wanted := filepath.Clean(declaration.Model)
		declared := false
		for _, entry := range entries {
			if !strings.EqualFold(filepath.Clean(entry.Location), wanted) {
				continue
			}
			id, err := evaluation.DeclareReferenceScores(ctx, store, entry.Model, declaration.References)
			if err != nil {
				return err
			}
			fmt.Printf("declared %d reference(s) for %s declaration=%s\n", len(declaration.References), entry.Model, id)
			declared = true
			break
		}
		if !declared {
			return fmt.Errorf("evaluate: %q is not a servable model location", declaration.Model)
		}
	}
	return nil
}
