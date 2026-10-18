package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/hfgguf"
	"overgo/internal/modelintake"
	"overgo/internal/overgodb"
)

func inspectModel(args []string) error {
	flags := flag.NewFlagSet("recipe inspect", flag.ContinueOnError)
	output := flags.String("output", "", "canonical inspection JSON destination; empty writes standard output")
	purpose := flags.String("purpose", "", "language output: tokens or hidden-states; empty derives from model class")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		return errors.New("usage: recipe inspect [-output JSON] [-purpose tokens|hidden-states] MODEL_DIRECTORY_OR_GGUF")
	}
	inspection, err := modelintake.InspectImport(context.Background(), flags.Arg(0), hfgguf.LanguageOutput(*purpose))
	if err != nil {
		return err
	}
	data := append(inspection.Content.Data, '\n')
	if *output != "" {
		return clioptions.WriteOutputFile(*output, data)
	}
	_, err = os.Stdout.Write(data)
	return err
}

func registerInspection(ctx context.Context, repository, path, declaration string) error {
	data, err := artifact.ReadContentFile(declaration)
	if err != nil {
		return err
	}
	inspection, err := modelintake.ReplayImport(ctx, path, data)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	if err := modelintake.RegisterImport(ctx, store, inspection); err != nil {
		return err
	}
	fmt.Printf("model=%s inspection=%s qualification=structural-only; model bytes copied=0; runtime execution and activation did not run\n", inspection.Inventory.Manifest.ID, inspection.Content.Descriptor.ID)
	return nil
}
