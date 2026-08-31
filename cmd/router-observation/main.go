// Command router-observation publishes one exact per-layer MoE router observation
// through the bounded raw-chunk and coverage authorities.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

func main() {
	clioptions.MainNamed("router-observation", func() error { return run(os.Args[1:], os.Stdout) })
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("router-observation", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	input := flags.String("input", "", "path to a router observation JSON document")
	storePath := flags.String("store", "", "OvergoDB root")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *input == "" || *storePath == "" {
		return errors.New("usage: router-observation -input <observation.json> -store <overgodb>")
	}
	data, err := os.ReadFile(*input)
	if err != nil {
		return err
	}
	var specification runrecord.MoERouterObservation
	if err := strictjson.DecodeBytes(data, &specification); err != nil {
		return err
	}
	observation, err := runrecord.NewMoERouterObservation(specification)
	if err != nil {
		return err
	}
	chunk, err := runrecord.NewMoERouterObservationChunk([]runrecord.MoERouterObservation{observation}, artifact.ID{})
	if err != nil {
		return err
	}
	steps := uint64(len(chunk.Observations))
	coverage, err := runrecord.NewMoERouterObservationCoverage(chunk, observation.Step, steps, []uint32{observation.Layer})
	if err != nil {
		return err
	}
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	batch, err := coverage.Batch(context.Background(), store, chunk)
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(context.Background(), store, batch); err != nil {
		return err
	}
	fmt.Fprintf(output, "router-observation: committed %s chunk=%s step=%d layer=%d rows=%d selections=%d\n",
		coverage.ID, chunk.ID, observation.Step, observation.Layer, observation.Rows, len(observation.Selections))
	return nil
}
