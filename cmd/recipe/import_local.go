package main

import (
	"context"
	"errors"
	"flag"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
)

// importLocal is recipe import-local: it serves model files an install already holds as one model, in place of a
// download. Each file a declaration names is hard-linked from the install into the data root's model folder, under the
// declaration's name, its bytes checked against the SHA-256 and size the declaration pins, and the model's inventory is
// published with the import's receipt: each file's source path in the install, its maker's address and its licence. No
// byte is copied, and a file whose bytes changed is refused before anything is published.
func importLocal(args []string) error {
	flags := flag.NewFlagSet("recipe import-local", flag.ContinueOnError)
	repository := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	declaration := flags.String("declaration", "", "the local import's declaration (internal/modelrecipe/moss_ultra_import.json)")
	from := flags.String("from", "", "the install whose files the declaration's sources name")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *declaration == "" || *from == "" {
		return errors.New("usage: recipe import-local -declaration JSON -from INSTALL [-repo STORE]")
	}
	declared, err := modelartifact.ReadLocalImport(*declaration)
	if err != nil {
		return err
	}
	roots, err := dataroot.Resolve(".")
	if err != nil {
		return err
	}
	store := roots.Store
	if *repository != "" {
		store = *repository
	}
	ctx := context.Background()
	batch, err := modelartifact.ImportLocal(ctx, declared, *from, roots.StoreModels)
	if err != nil {
		return err
	}
	opened, err := overgodb.Open(store)
	if err != nil {
		return err
	}
	defer opened.Close()
	if err := modelartifact.PreserveRawFileDescriptors(ctx, opened, &batch); err != nil {
		return err
	}
	if _, err := artifact.Publish(ctx, opened, batch); err != nil {
		return err
	}
	fmt.Printf("imported %s: %d files hard-linked into %s, model %s; bytes copied=0\n", declared.Name, len(declared.Files), roots.StoreModels, batch.Manifests[0].ID)
	return nil
}
