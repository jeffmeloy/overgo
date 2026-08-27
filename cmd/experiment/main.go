// Command experiment runs competing worker strategies against one plan
// step from one baseline commit, each in an isolated git worktree, and
// records the controlled comparison durably: every trial is judged by
// the step's own machine verify, selection is verify outcome first and
// measured cost second, and failed trials persist as counterexamples
// instead of vanishing. The winning worktree holds the candidate diff;
// adoption stays with the operator through the normal gate.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
)

func main() {
	clioptions.MainNamed("experiment", run)
}

func run() error {
	repository := flag.String("repo", "overgodb-store", "OvergoDB store directory recording the experiment")
	flag.Parse()
	if flag.NArg() != 1 {
		return errors.New("usage: experiment [-repo <store>] <spec.json>")
	}
	var spec Spec
	if err := jsonfile.Decode(flag.Arg(0), &spec); err != nil {
		return err
	}
	report, err := Run(&execHarness{}, spec)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(*repository)
	if err != nil {
		return err
	}
	defer store.Close()
	id, err := publishReport(context.Background(), store, report)
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(encoded))
	fmt.Printf("experiment %s recorded\n", id)
	return nil
}

const (
	experimentMediaType = "application/vnd.overgo.strategy-experiment+json"
	experimentSchema    = "overgo/strategy-experiment/v1"
)

// publishReport commits the comparison as evidence: the durable record
// controlled strategy experiments demand, failures included.
func publishReport(ctx context.Context, store *overgodb.Store, report Report) (artifact.ID, error) {
	contract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: experimentMediaType, Schema: experimentSchema,
	}
	encoded, err := json.Marshal(report)
	if err != nil {
		return artifact.ID{}, err
	}
	content, err := contract.ContentBytes(encoded)
	if err != nil {
		return artifact.ID{}, err
	}
	_, err = store.Commit(ctx, artifact.Batch{
		Key:       "experiment/" + strings.ReplaceAll(report.Step, "/", ".") + "/" + content.Descriptor.ID.String(),
		Artifacts: []artifact.Descriptor{content.Descriptor},
		Contents:  []artifact.Content{content},
	})
	if err != nil {
		return artifact.ID{}, err
	}
	return content.Descriptor.ID, nil
}
