package plan

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/textcheck"
)

const (
	roadmapProbeMediaType = "application/vnd.overgo.roadmap-probe+json"
	roadmapProbeSchema    = "overgo/roadmap-probe/v1"
)

// RoadmapProbe binds one isolated expected failure to a roadmap verifier.
type RoadmapProbe struct {
	Row             string      `json:"row"`
	Verifier        string      `json:"verifier"`
	CodeCommit      string      `json:"code_commit"`
	Run             artifact.ID `json:"run"`
	ExpectedFailure string      `json:"expected_failure"`
	ID              artifact.ID `json:"-"`
}

var roadmapProbeCodec = artifact.JSONDocumentCodec("roadmap probe", artifact.KindEvidence, roadmapProbeMediaType, roadmapProbeSchema,
	canonicalizeRoadmapProbe, func(value RoadmapProbe) artifact.ID { return value.ID },
	func(value *RoadmapProbe, id artifact.ID) { value.ID = id }, nil)

func identifyRoadmapProbe(value RoadmapProbe) (RoadmapProbe, artifact.Content, error) {
	identified, err := roadmapProbeCodec.New(value)
	if err != nil {
		return RoadmapProbe{}, artifact.Content{}, err
	}
	content, err := roadmapProbeCodec.Content(identified)
	if err != nil {
		return RoadmapProbe{}, artifact.Content{}, err
	}
	return identified, content, nil
}

// ReadRoadmapProbe returns false when id is not roadmap-probe evidence.
func ReadRoadmapProbe(ctx context.Context, reader artifact.Reader, id artifact.ID) (RoadmapProbe, bool, error) {
	return readTypedDocument(ctx, reader, id, roadmapProbeCodec.Contract, roadmapProbeCodec.Read)
}

// ExecuteRoadmapProbe accepts only a healthy verifier run whose named target
// is absent, then binds that observation to immutable run evidence and HEAD.
func ExecuteRoadmapProbe(ctx context.Context, root string, repository artifact.Repository, row, verifier string) (RoadmapProbe, error) {
	shell, err := clioptions.POSIXShell()
	if err != nil {
		return RoadmapProbe{}, err
	}
	started := time.Now()
	output, commandErr := clioptions.CombinedOutputIn(root, os.Environ(), shell, "-c", testevidence.JSONCommand(verifier))
	measured := uint64(max(time.Since(started).Nanoseconds(), 1))
	if commandErr != nil {
		return RoadmapProbe{}, fmt.Errorf("roadmap probe command failed: %w: %s", commandErr, clioptions.Tail(output, 2000))
	}
	if err := testevidence.VerifyGoTestTargetAbsent(verifier, output); err != nil {
		return RoadmapProbe{}, fmt.Errorf("roadmap probe is not an isolated absent target: %w", err)
	}
	command := exec.Command("git", "rev-parse", "HEAD")
	command.Dir = root
	head, err := command.Output()
	if err != nil {
		return RoadmapProbe{}, err
	}
	host, err := os.Hostname()
	if err != nil {
		host = "unknown"
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: host, OS: runtime.GOOS, Arch: runtime.GOARCH, Device: "host", Backend: "go", Driver: runtime.Version(),
	})
	if err != nil {
		return RoadmapProbe{}, err
	}
	recipe, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("overgo-roadmap-probe/v1"))
	if err != nil {
		return RoadmapProbe{}, err
	}
	commit := strings.TrimSpace(string(head))
	run, err := runrecord.NewBoundRun(recipe, runrecord.OutcomeFailed, nil, nil, "target-absent", commit, environment.ID,
		measured, []runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: measured}})
	if err != nil {
		return RoadmapProbe{}, err
	}
	batch, err := run.Batch("automation/roadmap-probe-run/" + run.ID.String())
	if err != nil {
		return RoadmapProbe{}, err
	}
	environmentContent, err := environment.Content()
	if err != nil {
		return RoadmapProbe{}, err
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: recipe})
	probe, probeContent, err := identifyRoadmapProbe(RoadmapProbe{
		Row: row, Verifier: verifier, CodeCommit: commit, Run: run.ID, ExpectedFailure: "target-absent",
	})
	if err != nil {
		return RoadmapProbe{}, err
	}
	batch.Contents = append(batch.Contents, environmentContent, probeContent)
	batch.Lineage = append(batch.Lineage, artifact.Lineage{
		Child: probe.ID, Parent: run.ID, Relation: artifact.RelationDependsOn,
	})
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return RoadmapProbe{}, err
	}
	return probe, nil
}

func canonicalizeRoadmapProbe(value *RoadmapProbe) error {
	if value == nil || !textcheck.LowerIdentifier(value.Row, 2048) ||
		!validCommit(value.CodeCommit) || value.Run.Kind() != artifact.KindRun ||
		!textcheck.LowerIdentifier(value.ExpectedFailure, 2048) {
		return errors.New("plan: invalid roadmap probe")
	}
	if err := testevidence.ValidateGoTestCommand(value.Verifier); err != nil {
		return fmt.Errorf("plan: invalid roadmap probe verifier: %w", err)
	}
	return nil
}
