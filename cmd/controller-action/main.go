// Command controller-action compiles one allowlisted controller action into
// its content-addressed artifacts and commits them. This is the single
// execution path for controller emissions: actions outside the allowlist are
// unrepresentable, and the executor is deterministic Go -- no code, no shell,
// no runtime orchestration.
//
//	go run ./cmd/controller-action -action <action.json> -record <repodb>
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/controlleraction"
	"overgo/internal/repodb"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "controller-action:", err)
		os.Exit(1)
	}
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("controller-action", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	actionPath := flags.String("action", "", "path to the typed action document (JSON)")
	recordStore := flags.String("record", "", "RepoDB root: commit the compiled artifacts")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *actionPath == "" || *recordStore == "" {
		return errors.New("usage: controller-action -action <action.json> -record <repodb>")
	}
	data, err := os.ReadFile(*actionPath)
	if err != nil {
		return err
	}
	action, err := controlleraction.ParseAction(data)
	if err != nil {
		return err
	}
	store, err := repodb.Open(*recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	batch, err := controlleraction.CompileTransaction(context.Background(), store, action)
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(context.Background(), store, batch); err != nil {
		return err
	}
	for _, content := range batch.Contents {
		fmt.Fprintf(output, "compiled artifact: %s\n", content.Descriptor.ID)
	}
	fmt.Fprintf(output, "action %s compiled %d artifact(s); honesty: deterministic Go executor, allowlist-only, never orchestrates\n",
		action.Kind, len(batch.Contents))
	return nil
}
