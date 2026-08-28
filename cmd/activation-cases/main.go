// Command activation-cases publishes and inspects typed behavioral case denominators.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"

	"overgo/internal/clioptions"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/strictjson"
)

func main() {
	clioptions.Main(run)
}

func run() error {
	repository := flag.String("repo", "overgodb-store", "OvergoDB root")
	flag.Parse()
	if flag.NArg() == 0 {
		store, err := overgodb.OpenReadOnly(*repository)
		if err != nil {
			return err
		}
		defer store.Close()
		registry, cases, found, err := evaluation.ResolveActivationCaseRegistry(context.Background(), store)
		if err != nil {
			return err
		}
		if !found {
			return errors.New("activation-cases: registry is absent")
		}
		fmt.Printf("registry=%s denominator=%d\n", registry.ID, registry.Denominator)
		for _, testCase := range cases {
			fmt.Printf("%s task=%s case=%s\n", testCase.Name, testCase.Task, testCase.ID)
		}
		return nil
	}
	if flag.NArg() != 1 {
		return errors.New("usage: activation-cases [-repo store] [cases.json]")
	}
	data, err := os.ReadFile(flag.Args()[0])
	if err != nil {
		return err
	}
	var cases []evaluation.ActivationCase
	if err := strictjson.DecodeBytes(data, &cases); err != nil {
		return fmt.Errorf("activation-cases: decode: %w", err)
	}
	store, err := overgodb.Open(*repository)
	if err != nil {
		return err
	}
	defer store.Close()
	registry, commit, err := evaluation.PublishActivationCaseRegistry(context.Background(), store, cases)
	if err != nil {
		return err
	}
	fmt.Printf("registry=%s denominator=%d commit=%s\n", registry.ID, registry.Denominator, commit)
	return nil
}
