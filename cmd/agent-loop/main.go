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
	"overgo/internal/invocation"
	"overgo/internal/operatoraction"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/representation"
	"overgo/internal/runrecord"
)

// usage names the command's forms.
//   - a step under a serving identity, or under a delegated invocation
//   - the restore of a delegated step's checkpoint
const usage = "usage: agent-loop [-repo <path>] -session <id> -tool <name> (-recipe <id> -model <id> -max-steps <n> [-node <id>] | -delegated <invocation-id>) [-arguments <json>] [-preview] [-approve <operation-id>]\n" +
	"   or: agent-loop [-repo <path>] -delegated <invocation-id> -restore <checkpoint-id> [-preview] [-approve <operation-id>]"

// stepCoordinator admits one step of a session.
//   - the coordinator, or the delegated coordinator that binds an invocation's grant and ceiling
type stepCoordinator interface {
	PreviewMutationDecision(context.Context, *agentloop.Session, string, json.RawMessage) (agentloop.DecisionPreview, error)
	ApproveMutation(context.Context, *agentloop.Session, string, json.RawMessage, artifact.ID) (artifact.ID, error)
	Propose(context.Context, *agentloop.Session, string, json.RawMessage) (json.RawMessage, error)
}

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
	approve := flags.String("approve", "", "grant the mutation step or the restore by naming the operation identity -preview printed")
	preview := flags.Bool("preview", false, "print the decision preview for this step or restore instead of running it")
	steered := flags.String("steered", "", "print the steered proposal phase for this residual direction instead of proposing")
	delegated := flags.String("delegated", "", "run the step under this delegated agent invocation identity")
	restore := flags.String("restore", "", "restore this checkpoint of a delegated step instead of proposing")
	recipeText := flags.String("recipe", "", "serving recipe artifact id the step records against")
	modelText := flags.String("model", "", "serving model artifact id the step records against")
	node := flags.String("node", "respond", "interaction node the step records against")
	var maxSteps int
	flags.IntVar(&maxSteps, "max-steps", maxSteps, "maximum admitted steps in this session")
	if err := flags.Parse(args); err != nil {
		return err
	}
	name, toolName := strings.TrimSpace(*sessionID), strings.TrimSpace(*tool)
	invocationText, checkpointText := strings.TrimSpace(*delegated), strings.TrimSpace(*restore)
	if flags.NArg() != 0 || checkpointText == "" && (name == "" || toolName == "") ||
		checkpointText != "" && invocationText == "" || invocationText == "" && maxSteps <= 0 {
		return errors.New(usage)
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
	executor := agenttool.NewOperatorExecutor()
	if err := agenttool.RegisterStandardBuiltins(executor, store); err != nil {
		return err
	}
	var coordinator stepCoordinator
	var session *agentloop.Session
	project := func(decision agentloop.DecisionPreview) any { return decision }
	if invocationText != "" {
		runtime, err := delegatedRuntime(ctx, store, executor, invocationText)
		if err != nil {
			return err
		}
		if checkpointText != "" {
			return restoreCheckpoint(ctx, store, runtime, checkpointText, *preview, strings.TrimSpace(*approve), output)
		}
		if session, err = runtime.RestoreSession(ctx, name); err != nil {
			return err
		}
		manuals, err := runtime.SessionManuals()
		if err != nil {
			return err
		}
		names := make([]string, 0, len(manuals))
		for _, manual := range manuals {
			names = append(names, manual.Name)
		}
		coordinator, project = runtime.Coordinator, func(decision agentloop.DecisionPreview) any {
			return struct {
				Invocation  artifact.ID               `json:"invocation"`
				Manuals     []string                  `json:"manuals"`
				Obligations int                       `json:"obligations"`
				Decision    agentloop.DecisionPreview `json:"decision"`
			}{runtime.Compiled.Invocation.ID, names, len(runtime.Obligations), decision}
		}
	} else {
		recipeID, err := artifact.ParseID(strings.TrimSpace(*recipeText))
		if err != nil {
			return fmt.Errorf("agent-loop: recipe id: %w", err)
		}
		modelID, err := artifact.ParseID(strings.TrimSpace(*modelText))
		if err != nil {
			return fmt.Errorf("agent-loop: model id: %w", err)
		}
		served, err := agentloop.New(store, executor, agentloop.Identity{
			Recipe: recipeID, Model: modelID, Node: recipe.NodeID(strings.TrimSpace(*node)),
		}, maxSteps)
		if err != nil {
			return err
		}
		if session, err = served.RestoreSession(ctx, name); err != nil {
			return err
		}
		coordinator = served
	}
	if *preview {
		// A preview records nothing, and a later run's -approve names the operation identity it prints.
		projected, err := coordinator.PreviewMutationDecision(ctx, session, toolName, json.RawMessage(*arguments))
		if err != nil {
			return err
		}
		return printJSON(output, project(projected))
	}
	if strings.TrimSpace(*approve) != "" {
		approved, err := artifact.ParseID(strings.TrimSpace(*approve))
		if err != nil {
			return fmt.Errorf("agent-loop: approval operation id: %w", err)
		}
		if _, err := coordinator.ApproveMutation(ctx, session, toolName, json.RawMessage(*arguments), approved); err != nil {
			return err
		}
	}
	result, err := coordinator.Propose(ctx, session, toolName, json.RawMessage(*arguments))
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s\n", result)
	return err
}

// delegatedRuntime loads one delegated invocation, its worker and its task, and joins their runtime.
//   - the workspace = the working directory, under which every checkpoint target resolves
func delegatedRuntime(ctx context.Context, store *overgodb.Store, executor *agenttool.Executor, invocationText string) (*agentloop.DelegatedCapabilityRuntime, error) {
	invocationID, err := artifact.ParseID(invocationText)
	if err != nil {
		return nil, fmt.Errorf("agent-loop: delegated invocation id: %w", err)
	}
	delegation, err := recipe.RequireDelegatedAgentInvocation(ctx, store, invocationID)
	if err != nil {
		return nil, err
	}
	worker, err := recipe.RequireAgentDefinition(ctx, store, delegation.Worker)
	if err != nil {
		return nil, err
	}
	task, err := recipe.RequireAgentTaskContract(ctx, store, delegation.Task)
	if err != nil {
		return nil, err
	}
	return agentloop.NewDelegatedCapabilityRuntime(ctx, store, executor, delegation, worker, task, "agent", ".")
}

// restoreCheckpoint restores one stored checkpoint of a delegated step's mutation.
//   - preview prints the restore's approval request and changes nothing
//   - an approval that names the request's operation records the grant, and any other is refused
//   - the restore runs only behind a recorded grant, and it binds the effect the checkpointed step recorded
func restoreCheckpoint(
	ctx context.Context,
	store *overgodb.Store,
	runtime *agentloop.DelegatedCapabilityRuntime,
	checkpointText string,
	preview bool,
	approval string,
	output io.Writer,
) error {
	checkpointID, err := artifact.ParseID(checkpointText)
	if err != nil {
		return fmt.Errorf("agent-loop: checkpoint id: %w", err)
	}
	checkpoint, err := runrecord.RequireAgentMutationCheckpoint(ctx, store, checkpointID)
	if err != nil {
		return err
	}
	ceiling := runtime.Compiled.Invocation.ID
	request, err := agentloop.RestoreApprovalRequest(checkpoint, ceiling)
	if err != nil {
		return err
	}
	if preview {
		return printJSON(output, request)
	}
	if approval != "" {
		approved, err := artifact.ParseID(approval)
		if err != nil {
			return fmt.Errorf("agent-loop: approval operation id: %w", err)
		}
		if approved != request.Operation {
			return fmt.Errorf("agent-loop: approval names operation %s, and the restore of checkpoint %s is operation %s, which -preview prints", approved, checkpoint.ID, request.Operation)
		}
		// A decision already on record stands, so approving again publishes nothing.
		_, recorded, err := runrecord.ResolveHumanDecision(ctx, store, request.Operation)
		if err != nil {
			return err
		}
		if !recorded {
			decision, err := runrecord.NewHumanDecision(request, operatoraction.AnswerGrant)
			if err != nil {
				return err
			}
			if err := runrecord.PublishHumanDecision(ctx, store, request, decision); err != nil {
				return err
			}
		}
	}
	stimulus, found, err := runrecord.ResolveAttemptStimulus(ctx, store, checkpoint.Operation, uint32(artifact.InitialDocumentVersion))
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("agent-loop: checkpoint %s names operation %s, which has no recorded step", checkpoint.ID, checkpoint.Operation)
	}
	effect, err := invocation.RequireEffect(ctx, store, stimulus.Effect)
	if err != nil {
		return err
	}
	receipt, err := runtime.Checkpoints.RestoreCheckpoint(ctx, checkpoint, effect, ceiling)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s\n", receipt)
	return err
}

// printJSON writes one value as a line of JSON.
func printJSON(output io.Writer, value any) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s\n", encoded)
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
