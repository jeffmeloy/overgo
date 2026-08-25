// Command admission is the owner-authority tool for the independent-admission
// contract: it commits per-generation authority bindings and validates
// generation succession. Deliberately NOT a controller action -- rebinding
// proposer, evaluator or decider authority is never expressible in the
// controller's language and always flows through this explicit tool.
//
//	go run ./cmd/admission -bind <binding.json> -record <overgodb>
//	go run ./cmd/admission -succeed <current-id> -prior <prior-id> -approval <decision-id> -record <overgodb>
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "admission:", err)
		os.Exit(1)
	}
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("admission", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	bind := flags.String("bind", "", "path to an admission-binding JSON specification to identify and commit")
	succeed := flags.String("succeed", "", "artifact ID of the generation-N binding to validate")
	prior := flags.String("prior", "", "artifact ID of the generation N-1 binding")
	approval := flags.String("approval", "", "artifact ID of the prior decider's approval decision")
	recordStore := flags.String("record", "", "OvergoDB root")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *recordStore == "" || (*bind == "") == (*succeed == "") {
		return errors.New("usage: admission -bind <binding.json> -record <overgodb> | admission -succeed <id> -prior <id> -approval <id> -record <overgodb>")
	}
	if *bind != "" {
		return runBind(*bind, *recordStore, output)
	}
	if *prior == "" || *approval == "" {
		return errors.New("succession requires -prior and -approval")
	}
	return runSucceed(*succeed, *prior, *approval, *recordStore, output)
}

func runBind(path, recordStore string, output io.Writer) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var specification runrecord.AdmissionBinding
	if err := strictjson.DecodeBytes(data, &specification); err != nil {
		return err
	}
	binding, err := runrecord.NewAdmissionBinding(specification)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	content, err := binding.Content()
	if err != nil {
		return err
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "admission/binding/" + binding.ID.String(), Contents: []artifact.Content{content},
	}); err != nil {
		return err
	}
	fmt.Fprintf(output, "admission binding committed: %s generation=%d\n", binding.ID, binding.Generation)
	return nil
}

func runSucceed(currentText, priorText, approvalText, recordStore string, output io.Writer) error {
	store, err := overgodb.OpenReadOnly(recordStore)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	read := func(text string) ([]byte, error) {
		id, err := artifact.ParseID(text)
		if err != nil {
			return nil, err
		}
		content, ok, err := artifact.ReadContent(ctx, store, id)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, fmt.Errorf("artifact %s content is absent", text)
		}
		return content.Data, nil
	}
	currentData, err := read(currentText)
	if err != nil {
		return err
	}
	current, err := runrecord.ParseAdmissionBinding(currentData)
	if err != nil {
		return err
	}
	priorData, err := read(priorText)
	if err != nil {
		return err
	}
	priorBinding, err := runrecord.ParseAdmissionBinding(priorData)
	if err != nil {
		return err
	}
	approvalData, err := read(approvalText)
	if err != nil {
		return err
	}
	decision, err := recipe.ParseDecision(approvalData)
	if err != nil {
		return err
	}
	if err := runrecord.ValidateAdmissionSuccession(current, priorBinding, decision); err != nil {
		return err
	}
	fmt.Fprintf(output, "succession VALID: generation %d -> %d under decider domain %q\n",
		priorBinding.Generation, current.Generation, priorBinding.Decider.Name)
	return nil
}
