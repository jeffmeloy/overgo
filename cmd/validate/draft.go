package main

import (
	"context"
	"errors"
	"fmt"
	"io"

	"overgo/internal/dataroot"
	"overgo/internal/libraryintake"
	"overgo/internal/overgodb"
)

// validateDraft validates the draft layer beside a model the library has activated and reports whether the model now
// speculates through it.
func validateDraft(ctx context.Context, output io.Writer, repository, modelPath string) (err error) {
	storeRoot, err := dataroot.StoreRoot(repository)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(storeRoot)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, store.Close()) }()
	gate, active, err := libraryintake.ValidateDraft(ctx, store, modelPath)
	if err != nil {
		return err
	}
	if !active {
		fmt.Fprintf(output, "validate: the draft beside %s kept too few of its drafts and stays a candidate (gate %s)\n", modelPath, gate)
		return nil
	}
	fmt.Fprintf(output, "validate: %s speculates through the draft beside it (gate %s)\n", modelPath, gate)
	return nil
}
