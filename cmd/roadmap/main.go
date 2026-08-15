package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/plan"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

func main() { clioptions.MainNamed("roadmap", run) }

func run() error {
	root := flag.String("root", ".", "repository root")
	roadmapPath := flag.String("roadmap", "docs/rsi_plan.json", "roadmap design record")
	storePath := flag.String("store", "repodb-store", "RepoDB evidence store")
	recordProbe := flag.String("record-probe", "", "record owner-approved isolated probe JSON in RepoDB")
	flag.Parse()
	if *recordProbe != "" {
		return recordRoadmapProbe(*root, *storePath, *recordProbe)
	}
	data, err := os.ReadFile(filepath.Join(*root, filepath.FromSlash(*roadmapPath)))
	if err != nil {
		return err
	}
	evidence, err := collectEvidence(*root, filepath.Join(*root, filepath.FromSlash(*storePath)))
	if err != nil {
		return err
	}
	report, err := plan.EvaluateRoadmap(data, evidence)
	if err != nil {
		return err
	}
	return clioptions.WritePrettyJSON(os.Stdout, report)
}

func collectEvidence(root, storePath string) (plan.RoadmapEvidence, error) {
	document, err := plan.Load(filepath.Join(root, filepath.FromSlash(plan.Path)))
	if err != nil {
		return plan.RoadmapEvidence{}, err
	}
	evidence := plan.RoadmapEvidence{Live: map[string]bool{}, InFlight: map[string]bool{}, Landed: map[string]bool{}, ProbeBound: map[string]string{}}
	for _, item := range document.Items {
		evidence.Live[item.ID] = true
	}
	reachable, err := gitCommits(root)
	if err != nil {
		return plan.RoadmapEvidence{}, err
	}
	store, err := repodb.OpenReadOnly(storePath)
	if err != nil {
		return plan.RoadmapEvidence{}, err
	}
	defer store.Close()
	ctx := context.Background()
	result, err := store.Query(ctx, repodb.Query{Kind: artifact.KindEvidence, MaxResults: repodb.MaxQueryResults})
	if err != nil {
		return plan.RoadmapEvidence{}, err
	}
	if result.Truncated {
		return plan.RoadmapEvidence{}, fmt.Errorf("roadmap evidence query truncated")
	}
	now := time.Now()
	for _, descriptor := range result.Artifacts {
		if descriptor.MediaType == runrecord.GateMediaType && descriptor.Schema == runrecord.GateSchema {
			content, ok, err := store.Content(ctx, descriptor.ID)
			if err != nil || !ok {
				continue
			}
			gate, err := runrecord.ParseGateResult(content.Data)
			if err != nil || gate.Outcome != runrecord.OutcomeSucceeded || !reachable[gate.CodeCommit] {
				continue
			}
			for _, step := range gate.Steps {
				if step.Name == "acceptance" {
					id, _, _ := strings.Cut(step.Evidence, "/")
					evidence.Landed[id] = id != ""
				}
			}
		}
		probe, probeOK, err := plan.ReadRoadmapProbe(ctx, store, descriptor.ID)
		if err == nil && probeOK && reachable[probe.CodeCommit] {
			runContent, ok, err := store.Content(ctx, probe.Run)
			run, parseErr := runrecord.ParseRun(runContent.Data)
			if err == nil && ok && parseErr == nil && run.Outcome == runrecord.OutcomeFailed &&
				run.CodeCommit == probe.CodeCommit && run.Failure == probe.ExpectedFailure {
				evidence.ProbeBound[probe.Row] = probe.Verifier
			}
		}
		lease, leaseOK, err := plan.ReadWorkLease(ctx, store, descriptor.ID)
		if err == nil && leaseOK {
			expires, err := time.Parse(time.RFC3339Nano, lease.ExpiresAt)
			id, _, _ := strings.Cut(lease.Task, "/")
			if err == nil && expires.After(now) && id != "" {
				evidence.InFlight[id] = true
			}
		}
	}
	return evidence, nil
}

func recordRoadmapProbe(root, storePath, input string) error {
	data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(input)))
	if err != nil {
		return err
	}
	var probe plan.RoadmapProbe
	if err := strictjson.DecodeBytes(data, &probe); err != nil {
		return err
	}
	store, err := repodb.Open(filepath.Join(root, filepath.FromSlash(storePath)))
	if err != nil {
		return err
	}
	defer store.Close()
	probe, err = plan.RecordRoadmapProbe(context.Background(), store, probe)
	if err != nil {
		return err
	}
	fmt.Println(probe.ID)
	return nil
}

func gitCommits(root string) (map[string]bool, error) {
	command := exec.Command("git", "rev-list", "HEAD")
	command.Dir = root
	output, err := command.Output()
	if err != nil {
		return nil, err
	}
	commits := map[string]bool{}
	for _, commit := range strings.Fields(string(output)) {
		commits[commit] = true
	}
	return commits, nil
}
