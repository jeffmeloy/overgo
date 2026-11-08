package main

import (
	"context"
	"errors"
	"strings"
	"testing"

	"overgo/internal/modelrecipe"
)

// TestAJudgeWithStaleEvidenceIsVerifiedBeforeItJudges holds a judged run to a judge with current evidence: the Wan
// student's 0.898 on 2026-11-08 was judged by a judge whose verification predated the architecture profile its build
// compiled. A judge whose evidence is out of date is re-verified once, however many models it judges, and the run goes
// on once it reads verified; a re-verification that fails, and a judge never verified, refuse the run, naming the
// command that re-verifies it; a verified judge, and a speech model's recognizer, are not re-verified.
func TestAJudgeWithStaleEvidenceIsVerifiedBeforeItJudges(t *testing.T) {
	t.Parallel()
	const judge = "C:/models/judge.gguf"
	judged := []modelRequest{
		{Path: "C:/models/wan", Judge: judge, Projector: "C:/models/mmproj.gguf"},
		{Path: "C:/models/student", Judge: judge, Projector: "C:/models/mmproj.gguf"},
		{Path: "C:/models/speech", Judge: "C:/models/recognizer"},
	}
	// run asks currentJudges of the judge starting in state, and says how many re-verifications it ran.
	run := func(state modelrecipe.VerificationState, failing bool) (int, error) {
		reverified := 0
		read := func(_ context.Context, path string) (modelrecipe.ModelState, error) {
			if path != judge {
				t.Fatalf("the state of %s was read; only the vision-language judge is", path)
			}
			return modelrecipe.ModelState{Verification: state, Reason: "verified against an architecture profile this build no longer compiles"}, nil
		}
		reverify := func(context.Context, string) error {
			reverified++
			if failing {
				return errors.New("case first answered no")
			}
			state = modelrecipe.StateVerified
			return nil
		}
		return reverified, currentJudges(t.Context(), judged, read, reverify)
	}
	if reverified, err := run(modelrecipe.StateOutOfDate, false); err != nil || reverified != 1 {
		t.Fatalf("a stale judge was re-verified %d times, ending %v; want once and the run going on", reverified, err)
	}
	if reverified, err := run(modelrecipe.StateOutOfDate, true); err == nil || reverified != 1 || !strings.Contains(err.Error(), "case first answered no") {
		t.Fatalf("a failed re-verification ended %v after %d, want the run refused with its failure", err, reverified)
	}
	if reverified, err := run(modelrecipe.StateNotVerified, false); err == nil || reverified != 0 || !strings.Contains(err.Error(), "-recorded "+judge) {
		t.Fatalf("a judge never verified ended %v after %d re-verifications, want the run refused naming the command", err, reverified)
	}
	if reverified, err := run(modelrecipe.StateVerified, false); err != nil || reverified != 0 {
		t.Fatalf("a verified judge ended %v after %d re-verifications, want none", err, reverified)
	}
}
