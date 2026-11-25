package runrecord

import "testing"

// fullMeasurementCheck is full-measurement/run's check as it was filed at 1bae67d52, before 5147c611f rewrote it as
// one pipeline: it keeps the scorecard's output in a shell variable and reads each model's row from it.
const fullMeasurementCheck = `s=$(go run ./cmd/validate -scorecard) && for m in Krea-2-Turbo SenseNova-U1-8B-MoT-Infographic-V3 SimpleDiffusion-TensorProductAttentionRope Un-0 MOSS-Image-SFT-797fa8e678b4f; do printf '%s\n' "$s" | grep -qE "^\| $m \| image-gen \| [0-9]" || exit 1; done && go run ./cmd/validate -readme && grep -qE '^\| Un-0 \| [0-9]' README.md`

// TestAnAssignmentOfACommandsOutputRunsIt holds check-form-reads/do (SQA-483): a check that assigns a command's
// output, as full-measurement/run's did, is read as the composite command the shell runs, with no error, the command
// it captures among its segments; a plain && chain whose first segment only assigns is still refused, as it exports
// nothing to the commands after it. ParseVerify had split s=$(...) at its parenthesis and refused s=$ as a segment
// that runs nothing, so the check was refused at its commit after nine hours of measurement.
func TestAnAssignmentOfACommandsOutputRunsIt(t *testing.T) {
	t.Parallel()
	parsed, err := ParseVerify(fullMeasurementCheck)
	if err != nil || !parsed.Composite {
		t.Fatalf("the assigning check parsed as %+v, %v; want a composite command with no error", parsed, err)
	}
	captured := false
	for _, segment := range parsed.Segments {
		if len(segment.Fields) >= 3 && segment.Fields[0] == "go" && segment.Fields[1] == "run" && segment.Fields[2] == "./cmd/validate" {
			captured = true
		}
	}
	if !captured {
		t.Errorf("the assigning check's segments %+v hold no go run ./cmd/validate; want the command it captures", parsed.Segments)
	}
	if _, err := ParseVerify("A=1 && go test ./internal/plan -run '^TestX$'"); err == nil {
		t.Error("a plain chain whose first segment only assigns was read as a check")
	}
}
