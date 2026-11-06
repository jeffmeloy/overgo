package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

var proofChildOutput = flag.String("out", "", "controlled proof child checkpoint")

// TestTrainingProofChild is the real child process for acquisition boundaries;
// ordinary test execution does nothing. It uses the same typed verification
// publisher as lifecycle fixtures, rather than printing a claimed success.
func TestTrainingProofChild(t *testing.T) {
	t.Parallel()
	mode := os.Getenv("OVERGO_TEST_PROOF_CHILD")
	if mode == "" {
		return
	}
	out := *proofChildOutput
	testcheck.NoError(t, os.MkdirAll(out, 0o700))
	testcheck.NoError(t, os.WriteFile(filepath.Join(out, "owned"), []byte("child still owns this"), 0o600))
	if address := os.Getenv("OVERGO_TEST_PROOF_CONTROL"); address != "" {
		control := testcheck.Must(net.Dial("tcp", address))(t)
		defer control.Close()
		if _, err := io.WriteString(control, out+"\n"); err != nil {
			t.Fatal(err)
		}
		if _, err := io.ReadFull(control, make([]byte, 1)); err != nil {
			t.Fatal(err)
		}
	}
	if mode == "missing" {
		return
	}
	var result modelrecipe.Verification
	testcheck.NoError(t, json.Unmarshal([]byte(os.Getenv("OVERGO_TEST_PROOF_PAIR")), &result))
	if mode == "fresh" || mode == "missing-with-concurrent-proof" {
		store := overgodbtest.Create(t, os.Getenv("OVERGO_TEST_PROOF_STORE"))
		id := testcheck.Must(artifact.ParseID(os.Getenv("OVERGO_TEST_PROOF_RECIPE")))(t)
		result = testcheck.Must(modelrecipetest.PublishVerification(t.Context(), store, "fixture/child/fresh", id))(t)
		if mode == "missing-with-concurrent-proof" {
			return
		}
	}
	if path := os.Getenv("OVERGO_TRAIN_VERIFICATION"); path != "" {
		if mode == "malformed" {
			testcheck.NoError(t, os.WriteFile(path, []byte("{} {}"), 0o600))
		} else if err := jsonfile.Write(path, result, 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

// TestTrainingAcquisitionRequiresItsFreshProof checks exact selected identities,
// a zero exit without evidence, and a reader opened before the child publishes.
func TestTrainingAcquisitionRequiresItsFreshProof(t *testing.T) {
	t.Parallel()
	executable := testcheck.Must(os.Executable())(t)
	for _, mode := range []string{"missing", "stale", "wrong", "malformed", "missing-with-concurrent-proof", "fresh"} {
		t.Run(mode, func(t *testing.T) {
			root := t.TempDir()
			writer := overgodbtest.Create(t, filepath.Join(root, "store"))
			selected := testutil.ArtifactID(t, artifact.KindRecipe, "selected")
			other := testutil.ArtifactID(t, artifact.KindRecipe, "other")
			if _, err := writer.Commit(t.Context(), artifact.Batch{Key: "recipes", Artifacts: []artifact.Descriptor{{ID: selected}, {ID: other}}}); err != nil {
				t.Fatal(err)
			}
			id := selected
			if mode == "wrong" {
				id = other
			}
			pair, err := modelrecipetest.PublishVerification(t.Context(), writer, "fixture/old", id)
			testcheck.NoError(t, err)
			encoded := testcheck.Must(json.Marshal(pair))(t)
			reader := overgodbtest.OpenReadOnly(t, filepath.Join(root, "store"))
			_, before := reader.Head()
			err = acquireTraining(t.Context(), io.Discard, root, reader, ModelValidation{Recipe: selected}, executable,
				[]string{"-test.run=^TestTrainingProofChild$", "-out", filepath.Join(root, "shared", "checkpoint")},
				"OVERGO_TEST_PROOF_CHILD="+mode, "OVERGO_TEST_PROOF_PAIR="+string(encoded), "OVERGO_TEST_PROOF_STORE="+filepath.Join(root, "store"), "OVERGO_TEST_PROOF_RECIPE="+selected.String())
			if mode == "fresh" {
				_, after := reader.Head()
				if err != nil || after <= before {
					t.Errorf("fresh child proof credit=%v head=%d/%d, want refreshed exact proof", err, before, after)
				}
			} else {
				if err == nil {
					t.Errorf("%s zero-exit producer credited without its fresh selected proof", mode)
				} else if reason := map[string]string{"stale": "predates", "wrong": "identity mismatch", "malformed": "trailing JSON value"}[mode]; reason != "" && !strings.Contains(err.Error(), reason) {
					t.Errorf("%s proof refused for %v, want %q", mode, err, reason)
				}
				if mode == "missing-with-concurrent-proof" {
					_, after := reader.Head()
					if after <= before {
						t.Error("fixture did not publish the competing fresh proof")
					}
				}
			}
		})
	}
}

// TestTrainingAcquisitionRetainsPrivateScratchUntilCompletion holds one child
// open while another same-recipe attempt finishes. Its files must still exist;
// each directory disappears only after its own process has actually completed.
func TestTrainingAcquisitionRetainsPrivateScratchUntilCompletion(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store := overgodbtest.Create(t, filepath.Join(root, "store"))
	executable := testcheck.Must(os.Executable())(t)
	control := testcheck.Must(net.Listen("tcp", "127.0.0.1:0"))(t)
	defer control.Close()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	stop := context.AfterFunc(ctx, func() { control.Close() })
	defer stop()
	args := []string{"-test.run=^TestTrainingProofChild$", "-out", filepath.Join(root, "shared", "checkpoint")}
	cell := ModelValidation{Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "shared recipe")}
	done := make(chan error, 1)
	go func() {
		done <- acquireTraining(ctx, io.Discard, root, store, cell, executable, args,
			"OVERGO_TEST_PROOF_CHILD=missing", "OVERGO_TEST_PROOF_CONTROL="+control.Addr().String())
		control.Close()
	}()
	connection := testcheck.Must(control.Accept())(t)
	defer connection.Close()
	path := testcheck.Must(bufio.NewReader(connection).ReadString('\n'))(t)
	path = strings.TrimSpace(path)
	if err := acquireTraining(ctx, io.Discard, root, store, cell, executable, args, "OVERGO_TEST_PROOF_CHILD=missing"); err == nil {
		t.Error("second missing-proof attempt credited")
	}
	if _, err := os.Stat(filepath.Join(path, "owned")); err != nil {
		t.Errorf("running child's scratch removed by same-recipe attempt: %v", err)
	}
	if _, err := connection.Write([]byte{1}); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err == nil {
		t.Error("first missing-proof attempt credited")
	}
	if _, err := os.Stat(filepath.Dir(path)); !os.IsNotExist(err) {
		t.Errorf("completed child's private scratch remains: %v", err)
	}
}
