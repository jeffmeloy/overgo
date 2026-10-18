package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/hfrepo"
	"overgo/internal/strictjson"
)

// assembleSpecification writes verified declarations for recipe register.
func assembleSpecification(args []string) error {
	flags := flag.NewFlagSet("recipe spec", flag.ContinueOnError)
	root := flags.String("root", "", "root containing the model directories")
	output := flags.String("output", "", "declaration file to write for recipe register -spec")
	licenses := flags.String("spdx", "", "comma-separated directory=identifier for a license the model card does not declare, after reviewing it")
	receiptPath := flags.String("source-receipt", "", "verified hub source receipt for one model directory without a Git checkout")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() == 0 || *root == "" || *output == "" {
		return errors.New("usage: recipe spec -root MODELS -output JSON [-spdx DIRECTORY=IDENTIFIER,...] [-source-receipt JSON] DIRECTORY...")
	}
	var sourceReceipt *hfrepo.SourceReceipt
	if *receiptPath != "" {
		if flags.NArg() != 1 {
			return errors.New("recipe spec: -source-receipt requires exactly one model directory")
		}
		raw, err := artifact.ReadContentFile(*receiptPath)
		if err != nil {
			return err
		}
		sourceReceipt = &hfrepo.SourceReceipt{}
		if err := strictjson.DecodeBytes(raw, sourceReceipt); err != nil {
			return err
		}
	}
	reviewed := map[string]string{}
	for pair := range strings.SplitSeq(*licenses, ",") {
		if directory, identifier, paired := strings.Cut(pair, "="); paired {
			reviewed[directory] = identifier
		}
	}
	declarations := make([]json.RawMessage, 0, flags.NArg())
	for _, directory := range flags.Args() {
		// Each directory hashes every weight it holds: say which one is running.
		fmt.Fprintf(os.Stderr, "recipe spec: identifying %s\n", directory)
		declaration, err := assembleRegistration(context.Background(), *root, directory, reviewed[directory], sourceReceipt)
		if err != nil {
			return err
		}
		declarations = append(declarations, declaration)
	}
	data, err := json.MarshalIndent(declarations, "", " ")
	if err != nil {
		return err
	}
	if err := clioptions.WriteOutputFile(*output, append(data, '\n')); err != nil {
		return err
	}
	fmt.Printf("declared=%d specification=%s; register with: go run ./cmd/recipe register -repo STORE -root %s -spec %s\n", len(declarations), *output, *root, *output)
	return nil
}
