// Command admission is the owner-authority tool for the independent-admission
// contract: it commits per-generation authority bindings and validates
// generation succession. Deliberately NOT a controller action -- rebinding
// proposer, evaluator or decider authority is never expressible in the
// controller's language and always flows through this explicit tool.
//
//	go run ./cmd/admission -bind <binding.json> -record <repodb>
//	go run ./cmd/admission -succeed <current-id> -prior <prior-id> -approval <decision-id> -record <repodb>
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
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
	promoteEvaluator := flags.String("promote-evaluator", "", "candidate evaluator artifact ID to judge against sealed oracle cases")
	oracleCases := flags.String("cases", "", "path to the sealed oracle cases JSON (models with known outcomes and evaluator scores)")
	priorAuthority := flags.String("prior-authority", "", "the prior generation's decider identity")
	recordStore := flags.String("record", "", "RepoDB root")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *promoteEvaluator != "" {
		if flags.NArg() != 0 || *oracleCases == "" || *approval == "" || *priorAuthority == "" || *recordStore == "" {
			return errors.New("usage: admission -promote-evaluator <id> -cases <cases.json> -approval <decision-id> -prior-authority <id> -record <repodb>")
		}
		return runPromoteEvaluator(*promoteEvaluator, *oracleCases, *approval, *priorAuthority, *recordStore, output)
	}
	if flags.NArg() != 0 || *recordStore == "" || (*bind == "") == (*succeed == "") {
		return errors.New("usage: admission -bind <binding.json> -record <repodb> | admission -succeed <id> -prior <id> -approval <id> -record <repodb>")
	}
	if *bind != "" {
		return runBind(*bind, *recordStore, output)
	}
	if *prior == "" || *approval == "" {
		return errors.New("succession requires -prior and -approval")
	}
	return runSucceed(*succeed, *prior, *approval, *recordStore, output)
}

// runPromoteEvaluator judges a candidate evaluator against sealed oracle
// cases under prior-generation approval, committing the promotion (or
// refusal) with full lineage.
func runPromoteEvaluator(evaluatorText, casesPath, approvalText, priorAuthorityText, recordStore string, output io.Writer) error {
	evaluator, err := artifact.ParseID(evaluatorText)
	if err != nil {
		return err
	}
	priorAuthority, err := artifact.ParseID(priorAuthorityText)
	if err != nil {
		return err
	}
	var cases []runrecord.EvaluatorCase
	if err := jsonfile.Decode(casesPath, &cases); err != nil {
		return err
	}
	store, err := repodb.Open(recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	ctx := context.Background()
	approvalID, err := artifact.ParseID(approvalText)
	if err != nil {
		return err
	}
	content, ok, err := store.Content(ctx, approvalID)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("approval decision %s is not committed", approvalText)
	}
	decision, err := recipe.ParseDecision(content.Data)
	if err != nil {
		return err
	}
	promotion, err := runrecord.PromoteEvaluator(evaluator, cases, decision, priorAuthority)
	if err != nil {
		return err
	}
	batch, err := promotion.Batch("admission/evaluator/" + promotion.ID.String())
	if err != nil {
		return err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return err
	}
	verdict := "REFUSED"
	if promotion.Promoted {
		verdict = "PROMOTED"
	}
	fmt.Fprintf(output, "evaluator promotion committed: %s (%s)\n", promotion.ID, verdict)
	fmt.Fprintf(output, "reason: %s\n", promotion.Reason)
	return nil
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
	store, err := repodb.Open(recordStore)
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
	store, err := repodb.OpenReadOnly(recordStore)
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
		content, ok, err := store.Content(ctx, id)
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
