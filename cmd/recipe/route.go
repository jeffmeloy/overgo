package main

import (
	"encoding/json"
	"errors"
	"flag"
	"os"
	"strings"

	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
)

// routeDecision derives or audits one routing decision. With -input, the
// signal and judged candidates decode from one strict document and the
// registered derivation rule selects; with -parse, a canonical decision
// document re-parses and proves its content identity. Both paths print the
// canonical record, and a signal no candidate satisfies exits nonzero with
// the exact refusal.
func routeDecision(arguments []string) error {
	flags := flag.NewFlagSet("recipe route", flag.ContinueOnError)
	input := flags.String("input", "", "routing input JSON path ({signal, candidates})")
	parse := flags.String("parse", "", "canonical routing decision JSON path to re-parse and audit")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	hasInput, hasParse := strings.TrimSpace(*input) != "", strings.TrimSpace(*parse) != ""
	if hasInput == hasParse || flags.NArg() != 0 {
		return errors.New("usage: recipe route -input <routing.json> | -parse <decision.json>")
	}
	var decision modelrecipe.RoutingDecision
	if hasParse {
		content, err := os.ReadFile(*parse)
		if err != nil {
			return err
		}
		decision, err = modelrecipe.ParseRoutingDecision(content)
		if err != nil {
			return err
		}
	} else {
		var routing struct {
			Signal     modelrecipe.RoutingSignal      `json:"signal"`
			Candidates []modelrecipe.RoutingCandidate `json:"candidates"`
		}
		if err := jsonfile.DecodeStrict(*input, &routing); err != nil {
			return err
		}
		derived, err := modelrecipe.DeriveRoutingDecision(routing.Signal, routing.Candidates)
		if err != nil {
			return err
		}
		decision = derived
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(decision); err != nil {
		return err
	}
	_, err := os.Stdout.WriteString(decision.ID.String() + "\n")
	return err
}
