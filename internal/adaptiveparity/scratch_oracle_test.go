package adaptiveparity

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

func TestScratchOracleBundle(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "fixtures", "adaptive_scratch_oracle.json"))
	if err != nil {
		t.Fatal(err)
	}
	oracle, canonical, err := NormalizeScratchOracle(raw)
	if err != nil {
		t.Fatal(err)
	}
	if oracle.ID.Kind() != artifact.KindEvidence || oracle.SourceCommit != "214950b3b0316bcdcab38a3b95127a927c0ab5be" {
		t.Fatalf("oracle identity/source = %s / %s", oracle.ID, oracle.SourceCommit)
	}
	corpusID, err := oracle.CorpusID()
	if err != nil {
		t.Fatal(err)
	}
	splitID, err := oracle.SplitID()
	if err != nil {
		t.Fatal(err)
	}
	if corpusID.Kind() != artifact.KindDataset || splitID.Kind() != artifact.KindDatasetShard || corpusID.String() == splitID.String() {
		t.Fatalf("oracle corpus/split identities = %s / %s", corpusID, splitID)
	}
	if len(oracle.Groups) != 21 || oracle.Config.EstimatedParams != 2704 || oracle.ParameterCount != 2720 || len(oracle.LossHistory) != 3 {
		t.Fatalf("oracle group/estimated/actual/trajectory coverage = %d / %d / %d / %d", len(oracle.Groups), oracle.Config.EstimatedParams, oracle.ParameterCount, len(oracle.LossHistory))
	}
	if oracle.LossHistory[0] != 2.3507537193457484 || oracle.LossHistory[2] != 1.660797282109651 ||
		oracle.FinalValLoss != 2.17824882011284 {
		t.Fatalf("oracle trajectory changed: %v final-val=%g", oracle.LossHistory, oracle.FinalValLoss)
	}
	content, err := oracle.Content()
	if err != nil {
		t.Fatal(err)
	}
	if content.Descriptor.ID != oracle.ID || !bytes.Equal(content.Data, canonical) {
		t.Fatal("oracle content identity or bytes changed")
	}

	t.Run("unknown-field-refused", func(t *testing.T) {
		mutated := bytes.Replace(raw, []byte(`"seed":7`), []byte(`"seed":7,"unowned":true`), 1)
		if _, err := ParseScratchOracle(mutated); err == nil {
			t.Fatal("unknown oracle field accepted")
		}
	})
	t.Run("digest-refused", func(t *testing.T) {
		mutated := bytes.Replace(raw, []byte(oracle.Groups[0].WeightSHA256), []byte("not-a-sha"), 1)
		if _, err := ParseScratchOracle(mutated); err == nil {
			t.Fatal("invalid group digest accepted")
		}
	})
}
