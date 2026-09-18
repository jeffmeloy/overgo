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
	"overgo/internal/representation"
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
	steered := flags.String("steered", "", "print the steered proposal phase for this residual direction instead of proposing")
	delegated := flags.String("delegated", "", "run the step under this delegated agent invocation identity")
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
	root, err := dataroot.StoreRoot(*repository)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	if strings.TrimSpace(*steered) != "" {
		return printSteeredPhase(ctx, store, strings.TrimSpace(*steered), output)
	}
	if strings.TrimSpace(*delegated) != "" {
		return runDelegatedStep(
			ctx, store, strings.TrimSpace(*delegated), strings.TrimSpace(*sessionID),
			strings.TrimSpace(*tool), json.RawMessage(*arguments), *preview, output,
		)
	}
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

// runDelegatedStep proposes one step under the exact delegated invocation:
// the joined delegated capability runtime compiles the grant, wires the
// coordinator, checkpoints, and grant-admitted proxy, and the step runs
// only over the invocation's own manual set. With -preview it prints the
// session's visible manuals and proposes nothing.
func runDelegatedStep(
	ctx context.Context,
	store *overgodb.Store,
	invocationText, sessionName, toolName string,
	arguments json.RawMessage,
	preview bool,
	output io.Writer,
) error {
	invocationID, err := artifact.ParseID(invocationText)
	if err != nil {
		return fmt.Errorf("agent-loop: delegated invocation id: %w", err)
	}
	invocation, err := recipe.RequireDelegatedAgentInvocation(ctx, store, invocationID)
	if err != nil {
		return err
	}
	worker, err := recipe.RequireAgentDefinition(ctx, store, invocation.Worker)
	if err != nil {
		return err
	}
	task, err := recipe.RequireAgentTaskContract(ctx, store, invocation.Task)
	if err != nil {
		return err
	}
	executor := agenttool.NewOperatorExecutor()
	if err := agenttool.RegisterStandardBuiltins(executor, store); err != nil {
		return err
	}
	runtime, err := agentloop.NewDelegatedCapabilityRuntime(
		ctx, store, executor, invocation, worker, task, "agent", ".",
	)
	if err != nil {
		return err
	}
	if preview {
		manuals, manualErr := runtime.SessionManuals()
		if manualErr != nil {
			return manualErr
		}
		names := make([]string, 0, len(manuals))
		for _, manual := range manuals {
			names = append(names, manual.Name)
		}
		encoded, encodeErr := json.Marshal(struct {
			Invocation  artifact.ID `json:"invocation"`
			Manuals     []string    `json:"manuals"`
			Obligations int         `json:"obligations"`
		}{invocation.ID, names, len(runtime.Obligations)})
		if encodeErr != nil {
			return encodeErr
		}
		_, err = fmt.Fprintf(output, "%s\n", encoded)
		return err
	}
	if sessionName == "" || toolName == "" {
		return errors.New("agent-loop: delegated step requires -session and -tool")
	}
	session := &agentloop.Session{ID: sessionName}
	result, err := runtime.Coordinator.Propose(ctx, session, toolName, arguments)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s\n", result)
	return err
}

// printSteeredPhase projects the steered proposal phase one residual
// direction would run under the ACTIVE tool catalog: the masked
// inspection-only action space and the gate-derived alpha. It records and
// activates nothing -- an operator reads exactly what a steered decode
// could see before any admission is proposed.
func printSteeredPhase(ctx context.Context, store *overgodb.Store, directionText string, output io.Writer) error {
	directionID, err := artifact.ParseID(directionText)
	if err != nil {
		return fmt.Errorf("agent-loop: steered direction id: %w", err)
	}
	content, found, err := artifact.ReadContent(ctx, store, directionID)
	if err != nil {
		return err
	}
	if !found {
		return errors.New("agent-loop: steered direction is not in the store")
	}
	direction, err := representation.ParseResidualDirection(content.Data)
	if err != nil {
		return err
	}
	active, activeFound, err := store.ResolveAlias(ctx, agenttool.ActiveCatalogAlias)
	if err != nil {
		return err
	}
	if !activeFound {
		return errors.New("agent-loop: no active tool catalog to steer over")
	}
	snapshot, err := agenttool.RequireCatalogSnapshot(ctx, store, active)
	if err != nil {
		return err
	}
	visible := make([]agenttool.Manual, 0, len(snapshot.Entries))
	for _, entry := range snapshot.Entries {
		manual, manualErr := agenttool.LoadManual(ctx, store, entry.Manual)
		if manualErr != nil {
			return manualErr
		}
		visible = append(visible, manual)
	}
	phase := agentloop.ProposalSteeringPhase(direction.ID, direction.Selector.Target, visible, visible)
	names := make([]string, 0, len(phase.Manuals))
	for _, manual := range phase.Manuals {
		names = append(names, manual.Name)
	}
	encoded, err := json.MarshalIndent(struct {
		Direction artifact.ID `json:"direction"`
		Alpha     float64     `json:"alpha"`
		Manuals   []string    `json:"manuals"`
	}{phase.Direction, phase.Alpha, names}, "", " ")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s\n", encoded)
	return err
}
