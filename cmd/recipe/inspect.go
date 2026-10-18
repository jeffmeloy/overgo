package main

import (
	"context"
	"errors"
	"flag"
	"os"

	"overgo/internal/clioptions"
	"overgo/internal/hfgguf"
	"overgo/internal/modelintake"
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
