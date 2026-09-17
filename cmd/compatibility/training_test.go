package main

import (
	"encoding/json"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestTrainingVerificationSpecs(t *testing.T) {
	root := trainingFixture(t)
	if _, err := loadTrainingVerifications(root); err != nil {
		t.Fatal(err)
	}
	testutil.WriteTextFile(t, root, "docs/verification/base-training-run.txt", "changed evidence\n")
	if _, err := loadTrainingVerifications(root); err == nil || !strings.Contains(err.Error(), "unclaimed") {
		t.Fatalf("changed evidence error = %v", err)
	}
}

func TestTrainingVerificationMatrix(t *testing.T) {
	root := trainingFixture(t)
	data, err := generateTraining(root)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, required := range []string{
		"1 model artifacts, 2 strongest training claims.",
		"| `fixture-model` | `training` | `real-artifact-smoke` | 3 steps / 12 tokens |",
		"| `fixture-model` | `full-pipeline-training` | `real-artifact-smoke` | 2 steps / 8 tokens |",
		"does not imply convergence",
	} {
		if !strings.Contains(text, required) {
			t.Fatalf("training matrix missing %q:\n%s", required, text)
		}
	}
}

func trainingFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		command := exec.Command("git", append([]string{"-C", root}, args...)...)
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, output)
		}
		return strings.TrimSpace(string(output))
	}
	git("init")
	git("config", "user.name", "Overgo Test")
	git("config", "user.email", "overgo@example.invalid")
	testutil.WriteTextFile(t, root, "marker", "fixture\n")
	git("add", "marker")
	git("commit", "-m", "fixture")
	commit := git("rev-parse", "HEAD")

	model := testutil.ArtifactID(t, artifact.KindModel, "model")
	datasetData := []byte("dataset\n")
	dataset := testutil.ArtifactID(t, artifact.KindDatasetShard, string(datasetData))
	testutil.WriteTextFile(t, root, "docs/verification/dataset.txt", string(datasetData))
	baseEvidenceData := []byte("base training evidence\n")
	fullEvidenceData := []byte("full training evidence\n")
	baseEvidence := testutil.ArtifactID(t, artifact.KindEvidence, string(baseEvidenceData))
	fullEvidence := testutil.ArtifactID(t, artifact.KindEvidence, string(fullEvidenceData))
	testutil.WriteTextFile(t, root, "docs/verification/base-training-run.txt", string(baseEvidenceData))
	testutil.WriteTextFile(t, root, "docs/verification/full-training-run.txt", string(fullEvidenceData))
	specification := verificationSpecification{
		Model: model, Name: "fixture-model",
		EvidenceFiles: []string{
			"docs/verification/base-training-run.txt",
			"docs/verification/full-training-run.txt",
		},
		DatasetFiles: []string{"docs/verification/dataset.txt"},
		Claims: []runrecord.CapabilityClaim{
			{
				Capability: "training", Tier: runrecord.TierRealArtifactSmoke, Commit: commit,
				Dataset: dataset, SpanSteps: 3, SpanTokens: 12, WallNS: 2_000_000,
				Evidence: []artifact.ID{baseEvidence},
			},
			{
				Capability: "full-pipeline-training", Tier: runrecord.TierRealArtifactSmoke, Commit: commit,
				Dataset: dataset, SpanSteps: 2, SpanTokens: 8, WallNS: 3_000_000,
				Evidence: []artifact.ID{fullEvidence},
			},
		},
	}
	data, err := json.MarshalIndent(specification, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	testutil.WriteTextFile(t, root, filepath.FromSlash("docs/verification/fixture-training.json"), string(data)+"\n")
	return root
}
