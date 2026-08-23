package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/evaluation"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

func main() {
	clioptions.MainNamed("eval-lane", run)
}

func run() error {
	repository := flag.String("repo", "", "RepoDB root")
	requiredText := flag.String("require", "", "comma-separated exact evaluation plan IDs")
	manifest := flag.String("manifest", "", "pinned evaluation manifest to execute first")
	flag.Parse()
	if flag.NArg() != 0 {
		return errors.New("usage: eval-lane [-repo path -require plan,... [-manifest path]]")
	}
	required, err := parseRequiredPlans(*requiredText)
	if err != nil {
		return err
	}
	if len(required) == 0 {
		if *manifest != "" {
			return errors.New("eval-lane: manifest requires exact plan IDs")
		}
		fmt.Println("eval-lane: no required plans configured; no evaluation verdict")
		return nil
	}
	if strings.TrimSpace(*repository) == "" {
		return runrecord.LaneError(runrecord.LaneUnavailable, "RepoDB path is absent")
	}
	if *manifest != "" {
		command := exec.Command("go", "run", "./cmd/evaluate", "-manifest", *manifest)
		command.Stdout, command.Stderr = os.Stdout, os.Stderr
		if err := command.Run(); err != nil {
			return runrecord.LaneError(runrecord.LaneFailed, "pinned evaluation manifest failed")
		}
	}
	store, err := repodb.OpenReadOnly(*repository)
	if err != nil {
		return runrecord.LaneError(runrecord.LaneUnavailable, err.Error())
	}
	defer store.Close()
	evidence, err := evidenceByPlan(context.Background(), store)
	if err != nil {
		return err
	}
	for _, plan := range required {
		values := evidence[plan]
		if len(values) == 0 {
			return runrecord.LaneError(runrecord.LaneUnavailable, "required plan evidence absent: "+plan.String())
		}
		for _, value := range values {
			if err := evaluation.ValidateEvaluationEvidence(context.Background(), store, value); err != nil {
				return runrecord.LaneError(runrecord.LaneFailed, plan.String()+": "+err.Error())
			}
		}
		fmt.Printf("[evaluation] plan=%s evidence=%d PASS\n", plan, len(values))
	}
	fmt.Printf("=== EVALUATION LANE %d exact plans PASS ===\n", len(required))
	return nil
}

func parseRequiredPlans(value string) ([]artifact.ID, error) {
	var result []artifact.ID
	for _, field := range strings.Split(value, ",") {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		id, err := artifact.ParseID(field)
		if err != nil || id.Kind() != artifact.KindProfile {
			return nil, errors.New("eval-lane: invalid required plan ID")
		}
		result = append(result, id)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].String() < result[j].String() })
	return slices.Compact(result), nil
}

func evidenceByPlan(ctx context.Context, store *repodb.Store) (map[artifact.ID][]evaluation.EvaluationEvidence, error) {
	result, err := store.Query(ctx, repodb.Query{
		Kind: artifact.KindEvidence, MaxResults: store.QueryExtent(), Projection: repodb.ProjectArtifacts,
	})
	if err != nil {
		return nil, err
	}
	if result.Truncated {
		return nil, errors.New("eval-lane: evidence query truncated")
	}
	byPlan := make(map[artifact.ID][]evaluation.EvaluationEvidence)
	for _, descriptor := range result.Artifacts {
		value, found, err := evaluation.LoadEvaluationEvidence(ctx, store, descriptor.ID)
		if err != nil {
			return nil, err
		}
		if found {
			byPlan[value.Plan] = append(byPlan[value.Plan], value)
		}
	}
	return byPlan, nil
}
