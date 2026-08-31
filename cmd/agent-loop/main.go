// Command agent-loop proposes one agent tool step through the
// coordinator: it admits the step under the inspection-before-mutation
// and approval rules and prints the durably recorded result. The
// serving identity the step records against is supplied explicitly.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"overgo/internal/agentloop"
	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

func main() {
	clioptions.MainNamed("agent-loop", func() error { return run(os.Args[1:], os.Stdout) })
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("agent-loop", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	repository := flags.String("repo", "", "OvergoDB root")
	sessionID := flags.String("session", "", "agent session identifier")
	tool := flags.String("tool", "", "registered tool manual name to propose")
	arguments := flags.String("arguments", "{}", "strict JSON object of tool arguments")
	approve := flags.String("approve", "", "grant the mutation step by naming the operation identity -preview printed")
	preview := flags.Bool("preview", false, "print the decision preview for this step instead of proposing it")
	recipeText := flags.String("recipe", "", "serving recipe artifact id the step records against")
	modelText := flags.String("model", "", "serving model artifact id the step records against")
	node := flags.String("node", "respond", "interaction node the step records against")
	var maxSteps int
	flags.IntVar(&maxSteps, "max-steps", maxSteps, "maximum admitted steps in this session")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || strings.TrimSpace(*sessionID) == "" || strings.TrimSpace(*tool) == "" || maxSteps <= 0 {
		return errors.New("usage: agent-loop [-repo <path>] -session <id> -tool <name> -recipe <id> -model <id> -max-steps <n> [-node <id>] [-arguments <json>] [-preview] [-approve <operation-id>]")
	}
	recipeID, err := artifact.ParseID(strings.TrimSpace(*recipeText))
	if err != nil {
		return fmt.Errorf("agent-loop: recipe id: %w", err)
	}
	modelID, err := artifact.ParseID(strings.TrimSpace(*modelText))
	if err != nil {
		return fmt.Errorf("agent-loop: model id: %w", err)
	}
	root := strings.TrimSpace(*repository)
	if root == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return err
		}
		root = roots.Store
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	executor := agenttool.NewOperatorExecutor()
	if err := agenttool.RegisterStandardBuiltins(executor, store); err != nil {
		return err
	}
	coordinator, err := agentloop.New(store, executor, agentloop.Identity{
		Recipe: recipeID, Model: modelID, Node: recipe.NodeID(strings.TrimSpace(*node)),
	}, maxSteps)
	if err != nil {
		return err
	}
	session := &agentloop.Session{ID: strings.TrimSpace(*sessionID)}
	if *preview {
		// Approving is a separate invocation: this one only projects the
		// decision facts -- including the operation identity a later
		// -approve must name -- and records nothing.
		projected, err := coordinator.PreviewMutationDecision(ctx, session, strings.TrimSpace(*tool), json.RawMessage(*arguments))
		if err != nil {
			return err
		}
		encoded, err := json.Marshal(projected)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(output, "%s\n", encoded)
		return err
	}
	if strings.TrimSpace(*approve) != "" {
		approved, err := artifact.ParseID(strings.TrimSpace(*approve))
		if err != nil {
			return fmt.Errorf("agent-loop: approval operation id: %w", err)
		}
		if _, err := coordinator.ApproveMutation(ctx, session, strings.TrimSpace(*tool), json.RawMessage(*arguments), approved); err != nil {
			return err
		}
	}
	result, err := coordinator.Propose(ctx, session, strings.TrimSpace(*tool), json.RawMessage(*arguments))
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s\n", result)
	return err
}
