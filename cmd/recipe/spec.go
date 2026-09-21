package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/modelartifact"
)

// assembleSpecification writes the declaration recipe register verifies, for
// the model directories named beneath a root. Registration takes exact
// identities and nothing wrote them: the one worked assembler was a test
// fixture, so a model directory could not be registered by a command.
func assembleSpecification(args []string) error {
	flags := flag.NewFlagSet("recipe spec", flag.ContinueOnError)
	root := flags.String("root", "", "root containing the model directories")
	output := flags.String("output", "", "declaration file to write for recipe register -spec")
	licenses := flags.String("spdx", "", "comma-separated directory=identifier for a license the model card does not declare, after reviewing it")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() == 0 || *root == "" || *output == "" {
		return errors.New("usage: recipe spec -root MODELS -output JSON [-spdx DIRECTORY=IDENTIFIER,...] DIRECTORY...")
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
		declaration, err := modelartifact.AssembleRegistration(context.Background(), *root, directory, reviewed[directory])
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
