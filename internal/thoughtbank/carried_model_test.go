//go:build modeltest

package thoughtbank

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testutil"
)

// carriedGainScript writes the fixture this test reads, from the model's own
// code. It is kept where Python is kept, on the machine that ran it and not
// in git: the fixture is the data.
const carriedGainScript = "scripts/python_golden_references/fractale_carried_gain.py"

const (
	// carriedSuite is the held-out suite of documents, beneath the datasets
	// root, that the fixture's losses are of.
	carriedSuite = "wikitext2_test/carried-documents-suite.json"
	// carriedEvery is how many documents apart the documents whose deferred
	// turn is scored are: every document's token ids are held to the model's
	// own tokenizer, and one in twelve is read through the model, at most of a
	// minute a document on the host.
	carriedEvery = 12
	// carriedTolerance is how far a deferred turn's loss a token may be from
	// the model's own code's, in nats: the turn is read through four chunks of
	// float32 arithmetic and a bank they wrote.
	carriedTolerance = 2e-2
)

// carriedGain mirrors fixtures/fractale_carried_gain.json: for each document
// of the held-out suite, what Fractale's own code reads on its deferred turn.
type carriedGain struct {
	WindowTokens int    `json:"window_tokens"`
	Chunks       int    `json:"chunks"`
	Opening      int    `json:"opening"`
	SeedSlots    int    `json:"seed_slots"`
	SeedInit     string `json:"seed_init"`
	Cases        []struct {
		Name       string      `json:"name"`
		TextSHA256 string      `json:"text_sha256"`
		IDsSHA256  string      `json:"ids_sha256"`
		Tokens     uint64      `json:"tokens"`
		Carried    float64     `json:"carried"`
		Reset      float64     `json:"reset"`
		Seeded     [][]float64 `json:"seeded"`
	} `json:"cases"`
}

// TestCarriedNotesGainOnTheNextChunk holds the measure of what the model is
// for, the gain its carried notes give on the opening of a chunk it has not
// seen, to the model's own code on the documents the model is scored on. Every
// document's token ids are the ids that code's tokenizer gives; and of one
// document in twelve, the loss of the deferred turn after four chunks, with
// the bank they wrote and with a fresh one, is that code's to within a
// tolerance a token. A fresh bank is seed slots of zeros on both sides.
//
// That code seeds a fresh bank with random numbers. The fixture holds, for its
// first documents, the two losses it reads from four seedings, and the test
// says what the mean gain is each way.
func TestCarriedNotesGainOnTheNextChunk(t *testing.T) {
	raw, err := os.ReadFile(testutil.FixturePath(t, "fractale_carried_gain.json"))
	if err != nil {
		t.Fatalf("the upstream deferred turns are absent (%v); write them with %s", err, carriedGainScript)
	}
	var fixture carriedGain
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	suitePath := filepath.Join(roots.Datasets, filepath.FromSlash(carriedSuite))
	data, err := os.ReadFile(suitePath)
	if err != nil {
		t.Skipf("UNAVAILABLE: held-out documents absent at %s; parity NOT verified", suitePath)
	}
	var suite struct {
		Chunks  int `json:"chunks"`
		Opening int `json:"opening"`
		Cases   []struct {
			Name string `json:"name"`
			Text string `json:"text"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &suite); err != nil {
		t.Fatal(err)
	}
	texts := map[string]string{}
	for _, document := range suite.Cases {
		texts[document.Name] = document.Text
	}
	generator, err := LoadGenerator(fractaleDirectory(t))
	if err != nil {
		t.Skipf("UNAVAILABLE: Fractale artifact unavailable: %v; parity NOT verified", err)
	}
	if len(fixture.Cases) == 0 || fixture.WindowTokens != scoringWindowTokens || fixture.SeedSlots != generator.config.MemSeedSlots ||
		fixture.Chunks != suite.Chunks || fixture.Opening != suite.Opening {
		t.Fatalf("the fixture holds %d documents read as %d chunks of %d tokens and an opening of %d from %d slots, and the suite and the served path read %d chunks of %d and an opening of %d from %d",
			len(fixture.Cases), fixture.Chunks, fixture.WindowTokens, fixture.Opening, fixture.SeedSlots, suite.Chunks, scoringWindowTokens, suite.Opening, generator.config.MemSeedSlots)
	}
	digest := func(text string) string {
		sum := sha256.Sum256([]byte(text))
		return hex.EncodeToString(sum[:])
	}
	var scored, seeded int
	var worst, zeroGain, seededGain, allGain float64
	for index, reference := range fixture.Cases {
		text, held := texts[reference.Name]
		if !held || digest(text) != reference.TextSHA256 {
			t.Fatalf("%s: the suite's document is not the one the fixture's losses are of; write the fixture again with %s", reference.Name, carriedGainScript)
		}
		encoded, err := generator.tokenizer.Encode(text)
		if err != nil {
			t.Fatal(err)
		}
		ids := make([]string, len(encoded))
		for position, token := range encoded {
			ids[position] = strconv.Itoa(token)
		}
		if digest(strings.Join(ids, ",")) != reference.IDsSHA256 {
			t.Fatalf("%s: the %d token ids the served path reads are not the ids the model's own tokenizer gives", reference.Name, len(encoded))
		}
		tokens := float64(reference.Tokens)
		allGain += (reference.Reset - reference.Carried) / tokens / float64(len(fixture.Cases))
		if len(reference.Seeded) != 0 {
			seeded++
			zeroGain += (reference.Reset - reference.Carried) / tokens
			for _, losses := range reference.Seeded {
				seededGain += (losses[1] - losses[0]) / tokens / float64(len(reference.Seeded))
			}
		}
		if index%carriedEvery != 0 {
			continue
		}
		carried, reset, err := generator.ScoreCarried(t.Context(), text, fixture.Chunks, fixture.Opening)
		if err != nil {
			t.Fatal(err)
		}
		apart := max(math.Abs(-carried.LogProbability-reference.Carried), math.Abs(-reset.LogProbability-reference.Reset)) / tokens
		if carried.Tokens != reference.Tokens || reset.Tokens != reference.Tokens || apart > carriedTolerance {
			t.Fatalf("%s: the deferred turn loses %.4f nats a token carried and %.4f reset, and the model's own code reads %.4f and %.4f, %.4f apart where %.4f is allowed",
				reference.Name, -carried.LogProbability/tokens, -reset.LogProbability/tokens, reference.Carried/tokens, reference.Reset/tokens, apart, carriedTolerance)
		}
		scored, worst = scored+1, max(worst, apart)
	}
	t.Logf("%d documents: the token ids are the model's own tokenizer's in every one; of %d scored, both losses are within %.2g nats a token of the model's own code's; that code's mean gain over all of them, from zeros, is %.3f nats a token",
		len(fixture.Cases), scored, worst, allGain)
	if seeded != 0 {
		t.Logf("seeding: that code seeds a fresh bank %s; over %d documents its mean gain from four seedings is %.3f nats a token, and from zeros %.3f",
			fixture.SeedInit, seeded, seededGain/float64(seeded), zeroGain/float64(seeded))
	}
}
