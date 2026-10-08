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

// passageLossScript writes the fixture this test reads, from the model's own
// code. It is kept where Python is kept, on the machine that ran it and not
// in git: the fixture is the data.
const passageLossScript = "scripts/python_golden_references/fractale_passage_loss.py"

const (
	// passageSuite is the held-out suite Fractale is scored on, beneath the
	// datasets root: the passages the fixture's losses are of.
	passageSuite = "wikitext2_test/passages-suite.json"
	// scoredEvery is how many passages apart the passages whose loss is
	// scored are: every passage's token ids are held to the model's own
	// tokenizer, which costs nothing, and one passage in eight is read through
	// the model, at ten seconds a passage on the host.
	scoredEvery = 8
	// lossTolerance is how far overgo's loss a token may be from the model's
	// own code's, in nats: both compute in float32, in different orders.
	lossTolerance = 2e-3
)

// passageLoss mirrors fixtures/fractale_passage_loss.json: for each passage of
// the held-out suite, what Fractale's own code reads.
type passageLoss struct {
	WindowTokens int    `json:"window_tokens"`
	SeedSlots    int    `json:"seed_slots"`
	SeedInit     string `json:"seed_init"`
	Cases        []struct {
		Name           string    `json:"name"`
		TextSHA256     string    `json:"text_sha256"`
		IDsSHA256      string    `json:"ids_sha256"`
		Tokens         uint64    `json:"tokens"`
		LogProbability float64   `json:"log_probability"`
		Seeded         []float64 `json:"seeded_log_probability"`
	} `json:"cases"`
}

// TestPassageLossMatchesUpstream holds the path a held-out passage takes, from
// its text to the loss the scorecard's perplexity is made of, to the model's
// own code on the passages the model is scored on. Every passage's token ids
// are the ids that code's tokenizer gives, by their digest; and of one passage
// in eight, the summed log-probability of its tokens, read in the windows the
// served path reads it in, each from a bank of zeros, is that code's to within
// a tolerance a token.
//
// That code starts a fresh bank from random numbers, where the served path
// starts it from zeros. The fixture holds the losses it reads from four
// randomly seeded banks beside the loss from zeros, and the test says how far
// apart they are: what the difference in seeding is worth to the reading.
func TestPassageLossMatchesUpstream(t *testing.T) {
	raw, err := os.ReadFile(testutil.FixturePath(t, "fractale_passage_loss.json"))
	if err != nil {
		t.Fatalf("the upstream losses are absent (%v); write them with %s", err, passageLossScript)
	}
	var fixture passageLoss
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	suitePath := filepath.Join(roots.Datasets, filepath.FromSlash(passageSuite))
	data, err := os.ReadFile(suitePath)
	if err != nil {
		t.Skipf("UNAVAILABLE: held-out passages absent at %s; parity NOT verified", suitePath)
	}
	var suite struct {
		Cases []struct {
			Name string `json:"name"`
			Text string `json:"text"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &suite); err != nil {
		t.Fatal(err)
	}
	texts := map[string]string{}
	for _, passage := range suite.Cases {
		texts[passage.Name] = passage.Text
	}
	generator, err := LoadGenerator(fractaleDirectory(t))
	if err != nil {
		t.Skipf("UNAVAILABLE: Fractale artifact unavailable: %v; parity NOT verified", err)
	}
	if len(fixture.Cases) == 0 || fixture.WindowTokens != scoringWindowTokens || fixture.SeedSlots != generator.config.MemSeedSlots {
		t.Fatalf("the fixture holds %d passages read in windows of %d tokens from %d slots, and the served path reads windows of %d from %d",
			len(fixture.Cases), fixture.WindowTokens, fixture.SeedSlots, scoringWindowTokens, generator.config.MemSeedSlots)
	}
	digest := func(text string) string {
		sum := sha256.Sum256([]byte(text))
		return hex.EncodeToString(sum[:])
	}
	var scored, seededPassages int
	var worst, seededGap, seededSpread float64
	for index, reference := range fixture.Cases {
		text, held := texts[reference.Name]
		if !held || digest(text) != reference.TextSHA256 {
			t.Fatalf("%s: the suite's passage is not the one the fixture's loss is of; write the fixture again with %s", reference.Name, passageLossScript)
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
		if seeds := len(reference.Seeded); seeds != 0 {
			var mean, low, high float64
			low, high = reference.Seeded[0], reference.Seeded[0]
			for _, seeded := range reference.Seeded {
				mean += seeded / float64(seeds)
				low, high = min(low, seeded), max(high, seeded)
			}
			seededPassages++
			seededGap += (mean - reference.LogProbability) / float64(reference.Tokens)
			seededSpread += (high - low) / float64(reference.Tokens)
		}
		if index%scoredEvery != 0 {
			continue
		}
		score, err := generator.scoreSequence(t.Context(), text)
		if err != nil {
			t.Fatal(err)
		}
		if score.Tokens != reference.Tokens {
			t.Fatalf("%s: %d tokens were scored, and the model's own code scores %d", reference.Name, score.Tokens, reference.Tokens)
		}
		apart := math.Abs(score.LogProbability-reference.LogProbability) / float64(reference.Tokens)
		if apart > lossTolerance {
			t.Fatalf("%s: the loss is %.5f nats a token and the model's own code reads %.5f, %.5f apart where %.5f is allowed",
				reference.Name, -score.LogProbability/float64(score.Tokens), -reference.LogProbability/float64(reference.Tokens), apart, lossTolerance)
		}
		scored, worst = scored+1, max(worst, apart)
	}
	t.Logf("%d passages: the token ids are the model's own tokenizer's in every one; of %d scored, the loss is within %.2g nats a token of the model's own code's",
		len(fixture.Cases), scored, worst)
	if seededPassages != 0 {
		t.Logf("seeding: that code seeds a fresh bank %s; over %d passages its loss from four seeded banks is %+.4f nats a token from its loss from zeros on average, and the four differ among themselves by %.4f",
			fixture.SeedInit, seededPassages, -seededGap/float64(seededPassages), seededSpread/float64(seededPassages))
	}
}
