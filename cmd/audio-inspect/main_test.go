package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestAudioDecodeCommand(t *testing.T) {
	for _, test := range []struct {
		name     string
		samples  []int16
		accepted bool
	}{
		{"accepted", []int16{16384, -16384}, true},
		{"silent", []int16{0, 0}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			input := filepath.Join(root, "input.wav")
			encoded := testutil.MonoPCM16WAV(16000, test.samples)
			if err := os.WriteFile(input, encoded, 0o600); err != nil {
				t.Fatal(err)
			}
			policy := dataset.AudioInspectionPolicy{MaximumEncodedBytes: 1024, MaximumSamples: 2, ClipThreshold: 1,
				Admission: recipecontract.AudioAdmissionPolicy{MinimumChannels: 1, MaximumChannels: 1, MaximumAbsoluteDCOffset: 1}}
			data, err := json.Marshal(policy)
			if err != nil {
				t.Fatal(err)
			}
			policyPath := filepath.Join(root, "policy.json")
			if err := os.WriteFile(policyPath, data, 0o600); err != nil {
				t.Fatal(err)
			}
			args := []string{"-repo", filepath.Join(root, "store"), "-input", input, "-policy", policyPath}
			var output bytes.Buffer
			err = run(t.Context(), args, &output)
			if (err == nil) != test.accepted {
				t.Fatalf("error=%v output=%s", err, &output)
			}
			var result dataset.AudioInspection
			if err := json.Unmarshal(bytes.SplitN(output.Bytes(), []byte{'\n'}, 2)[0], &result); err != nil {
				t.Fatal(err)
			}
			if result.SignalID.Kind() != artifact.KindEvidence || result.Signal.SampleCount != 2 || !strings.Contains(output.String(), "inspected=1") {
				t.Fatalf("output=%s", &output)
			}
			output.Reset()
			if err := run(t.Context(), args, &output); (err == nil) != test.accepted {
				t.Fatalf("repeated inspection: %v", err)
			}
			// A wrong pinned source must fail before inspecting any samples.
			expect := filepath.Join(root, "expect.json")
			if err := os.WriteFile(expect, []byte(`{"container_sha256":"wrong","observations":[{}]}`), 0o600); err != nil {
				t.Fatal(err)
			}
			output.Reset()
			if err := run(t.Context(), append(args, "-expect", expect), &output); err == nil || output.Len() != 0 {
				t.Fatal("wrong source expectation accepted")
			}
		})
	}
}

func TestAudioDecodeCommandRequiresExplicitPolicy(t *testing.T) {
	for _, args := range [][]string{nil, {"-repo", "store"}, {"-repo", "store", "-input", "file", "-policy", "policy", "-column", "bytes"}} {
		if err := run(t.Context(), args, &bytes.Buffer{}); err == nil {
			t.Fatalf("accepted flags %v", args)
		}
	}
}
