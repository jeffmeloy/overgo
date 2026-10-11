package evaluation

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
	"overgo/internal/tensorstats"
	"overgo/internal/testutil"
)

// TestInstructionPilotRecordIsTheAcceptanceItNames reads the record of the
// instruction pilot's acceptance, fixed over Qwen2.5-0.5B from the store's
// instruction cases. The acceptance document the record carries is the
// acceptance it names, and the training specification the specification it
// names: each canonicalizes to the identity beside it, so no figure in either
// was typed, and the acceptance names that specification. What it fixes is
// the pilot's: a gain of 0.20 in the share of groups passed, the two adoption
// comparisons, answers drawn from the model's unmodified distribution within
// 2,048 tokens and 1,280 of answer, scored by the strict instruction scorer.
// And it follows from the reading the record states: its basis is the
// development groups' draws, one a group; arms differ on at most the upper
// bound of the share of those groups the base differs on read twice, and the
// gain more; and its power is the exact chance of the gain's being shown over
// its promotion groups at that share, which is not less than was wanted.
func TestInstructionPilotRecordIsTheAcceptanceItNames(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "docs", "verification", "instruction-pilot-qwen25-05b.txt"))
	if err != nil {
		t.Fatal(err)
	}
	record := map[string]string{}
	for line := range strings.Lines(string(raw)) {
		if key, value, found := strings.Cut(strings.TrimSpace(line), ": "); found && !strings.Contains(key, " ") {
			record[key] = value
		}
	}
	named, err := artifact.ParseID(record["acceptance"])
	if err != nil {
		t.Fatal(err)
	}
	var document InstructionAcceptance
	if err := strictjson.DecodeBytes([]byte(record["document"]), &document); err != nil {
		t.Fatal(err)
	}
	acceptance, err := instructionAcceptanceCodec.New(document)
	if err != nil || acceptance.ID != named {
		t.Fatalf("the record's document is acceptance %s (%v), and the record names %s", acceptance.ID, err, named)
	}
	specification, err := artifact.IdentifyBytes(artifact.KindProfile, []byte(record["specification"]))
	if err != nil || specification.String() != record["training"] || acceptance.Training != specification {
		t.Fatalf("the record's specification is %s (%v), the record names %s and the acceptance %s", specification, err, record["training"], acceptance.Training)
	}
	pilot := InstructionProtocol{Scorer: InstructionScorer, Template: "chat/user-turn", Thinking: "off", Sampling: "unmodified", Stop: "end of generation", Window: 2048, MaxTokens: 1280}
	if acceptance.Protocol != pilot || acceptance.Gain != 0.2 || len(acceptance.Comparisons) != 2 || acceptance.Comparisons[0].Name != "reward over base" || acceptance.Comparisons[1].Name != "reward over control" {
		t.Fatalf("the acceptance fixes %+v, a gain of %v and %+v, which are not the pilot's", acceptance.Protocol, acceptance.Gain, acceptance.Comparisons)
	}
	var groups, differing uint64
	if _, err := fmt.Sscanf(record["development"], "%d groups read twice, %d differing", &groups, &differing); err != nil {
		t.Fatalf("the record's development reading %q: %v", record["development"], err)
	}
	_, repeated := tensorstats.ExactShareBounds(differing, groups, falseGain)
	power := instructionPower(acceptance.Groups, acceptance.Discordance, acceptance.Gain, falseGain/float64(len(acceptance.Comparisons)))
	if uint64(len(acceptance.Basis)) != groups || acceptance.Discordance != repeated+acceptance.Gain || acceptance.Power != power || power < 1-missedGain {
		t.Fatalf("the acceptance has a basis of %d draws, arms differing on %v and power %v; the record's %d development groups, %d of them differing, give %v and %v", len(acceptance.Basis), acceptance.Discordance, acceptance.Power, groups, differing, repeated+acceptance.Gain, power)
	}
	t.Logf("acceptance %s: %d promotion groups show a gain of %.2f with chance %.3f, arms differing on at most %.3f", named, acceptance.Groups, acceptance.Gain, acceptance.Power, acceptance.Discordance)
}
