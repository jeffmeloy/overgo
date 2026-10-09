//go:build modeltest

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/modelcli"
	"overgo/internal/sampling"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// answersScript writes the fixtures this test reads, from transformers
// running each model's own weights over the tokens its prompt fixture holds.
// It is kept where Python is kept, on the machine that ran it and not in
// git: the fixtures are the data.
const answersScript = "scripts/python_golden_references/answer_log_probabilities.py"

const (
	// answerTolerance is how far the log-probability overgo gives a token
	// may stand from the reference's, as a mean over the tokens scored. The
	// reference computes in float32 on the host over the weights as
	// published; overgo computes on the device over the scored file, which
	// holds them in half precision. Measured over eight-token answers, the
	// widest mean is 0.021, on Qwen2.5-0.5B.
	answerTolerance = 0.05
	// decisiveMargin is how far the reference's likeliest token must stand
	// above its next for overgo's greedy choice to be held to it. One token's
	// log-probability was measured as far as 0.11 from the reference's, on
	// Qwen2.5-0.5B, and a choice between two can move by both. Under this
	// margin the two are a tie that half precision may break either way, and
	// the answers past it are no longer answers to the same words.
	decisiveMargin = 0.25
	// tokenTolerance is how far one token's log-probability may stand from
	// the reference's: half the decisive margin, the 0.11 measured rounded
	// up. A continuation of a single token, as a candidate letter is, has no
	// mean to average its noise away, so it is held to this, and one of a few
	// tokens to this over the square root of their count, down to the mean's
	// own tolerance.
	tokenTolerance = decisiveMargin / 2
)

// knownAnswerDifferences are the scored files whose answers stand further
// from the reference than half precision explains, each for a way of
// computing that is the file's own and that the reference does not share.
// Such a file is held, by the name of its fixture, to a wider band, set a
// little past what was measured on it, and must still stand outside the
// common band on some case: an entry that outlives its difference fails this
// test until it is removed.
var knownAnswerDifferences = map[string]struct {
	tolerance, margin float64
	reason            string
}{
	"gemma-4-12B-it-fp8-native": {0.3, 2.5, "over a prompt of 32 tokens or more this build rounds what goes into each feed-forward product to fp8, " +
		"a scale to a token, as the checkpoint's quantization declares; the reference and the bf16 build round the weights alone. Measured: " +
		"0.223 a token on one answer, and one of 24 second tokens other than the reference's, which held it 1.86 above its next (finding 170f2bfa)"},
}

// answerReference is one model's fixture: for each case of its prompt
// fixture, the tokens transformers generates greedily from the case's tokens
// with the log-probability of each and its margin over the next likeliest,
// or the log-probability of each continuation the case holds.
type answerReference struct {
	Schema        string `json:"schema"`
	Model         string `json:"model"`
	File          string `json:"file"`
	Protocol      string `json:"protocol"`
	NewTokens     int    `json:"new_tokens"`
	Source        string `json:"source"`
	Prompts       string `json:"prompts"`
	PromptsSHA256 string `json:"prompts_sha256"`
	Cases         []struct {
		Name           string              `json:"name"`
		Generated      []tokenizer.TokenID `json:"generated"`
		Margins        []float64           `json:"margins"`
		LogProbability float64             `json:"log_probability"`
		Continuations  []struct {
			Tokens         int     `json:"tokens"`
			LogProbability float64 `json:"log_probability"`
		} `json:"continuations"`
	} `json:"cases"`
}

// TestModelsAnswerAsTransformersDoes holds the second stretch of the
// path a text reading takes, from the tokens a model is given to what it
// answers, to transformers running the model's own weights. The first
// stretch, from an evaluation's question to those tokens, is held by
// TestPromptsReachEachModelAsItsOwnTokenizerGivesThem; this one starts from
// the very tokens that test's fixtures hold.
//
// A question asked through the chat template is answered by greedy
// generation. The scored file is given the tokens transformers generated and
// must find them as likely, within a tolerance for half precision; and its
// own greedy tokens must be the reference's wherever the reference's choice
// was not a near tie. A sequence is scored whole, and its log-probability
// must be the reference's within the same tolerance a token.
//
// A fixture named for a model's conversations starts instead from the tokens
// TestChatModesReachAModelAsItsOwnTemplateGivesThem holds: a request with the
// thinking switch on and off, with tools offered, and with a tool called and
// answered. Its answers are longer, long enough to show whether the model
// opens a tool call and with which arguments.
//
// The models are those whose own weights are on this machine and small enough
// for transformers to run on the host.
func TestModelsAnswerAsTransformersDoes(t *testing.T) {
	cudatest.Require(t)
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	fixtures, err := filepath.Glob(filepath.Join(root, "fixtures", "answer_log_probabilities", "*.json"))
	if err != nil || len(fixtures) == 0 {
		t.Fatalf("no model's reference answers are on file (%v); write them with %s", err, answersScript)
	}
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	for _, fixture := range fixtures {
		raw, err := os.ReadFile(fixture)
		if err != nil {
			t.Fatal(err)
		}
		var reference answerReference
		if err := json.Unmarshal(raw, &reference); err != nil || len(reference.Cases) == 0 {
			t.Fatalf("%s holds no case: %v", fixture, err)
		}
		name := strings.TrimSuffix(filepath.Base(fixture), filepath.Ext(fixture))
		t.Run(name, func(t *testing.T) {
			// The band the file is held to, and whether any case stands outside
			// the common one.
			tolerance, margin, outside := answerTolerance, decisiveMargin, false
			known, differs := knownAnswerDifferences[name]
			if differs {
				tolerance, margin = known.tolerance, known.margin
			}
			asked, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(reference.Prompts)))
			if err != nil {
				t.Fatal(err)
			}
			if digest := sha256.Sum256(asked); hex.EncodeToString(digest[:]) != reference.PromptsSHA256 {
				t.Fatalf("%s is not the file the reference answered: write the answers again with %s", reference.Prompts, answersScript)
			}
			var prompts promptReference
			if err := json.Unmarshal(asked, &prompts); err != nil || len(prompts.Cases) != len(reference.Cases) || prompts.Protocol != reference.Protocol {
				t.Fatalf("%s holds %d cases asked as %q, the reference %d as %q: %v",
					reference.Prompts, len(prompts.Cases), prompts.Protocol, len(reference.Cases), reference.Protocol, err)
			}
			model := roots.ResolveModelPath(reference.File)
			if _, err := os.Stat(model); err != nil {
				t.Skipf("the scored file %s is not on this machine", reference.File)
			}
			runner, err := modelcli.OpenRunner(t.Context(), roots.Store, model, inference.OpenOptions{})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = runner.Close() })
			widest, same, tokens := 0.0, 0, 0
			for index, own := range reference.Cases {
				given := prompts.Cases[index]
				if given.Name != own.Name {
					t.Fatalf("case %d is %s in the prompts and %s in the answers", index, given.Name, own.Name)
				}
				if reference.Protocol != promptChat {
					scores, err := runner.ScoreContinuationTokens(t.Context(), given.IDs, given.Continuations)
					if err != nil || len(scores) != len(own.Continuations) {
						t.Fatalf("%s: %d of %d continuations were scored: %v", own.Name, len(scores), len(own.Continuations), err)
					}
					// The candidate each finds likeliest, and how far the reference's
					// stood above its next: what a choice among candidates is read from.
					chosen, likeliest, next := 0, 0, math.Inf(-1)
					for candidate, score := range scores {
						held := own.Continuations[candidate]
						apart := math.Abs(score.LogProbability-held.LogProbability) / float64(held.Tokens)
						// A mean over few tokens averages less of each token's own noise away.
						few := max(1, tokenTolerance/math.Sqrt(float64(held.Tokens))/answerTolerance)
						widest = max(widest, apart)
						tokens += held.Tokens
						outside = outside || apart > answerTolerance*few
						if int(score.Tokens) != held.Tokens || apart > tolerance*few {
							t.Errorf("%s continuation %d: %d tokens at %.4f, where the reference gives %d at %.4f: %.4f a token apart, over %.3f",
								own.Name, candidate, score.Tokens, score.LogProbability, held.Tokens, held.LogProbability, apart, tolerance*few)
						}
						if score.LogProbability > scores[chosen].LogProbability {
							chosen = candidate
						}
						switch best := own.Continuations[likeliest].LogProbability; {
						case candidate == 0:
						case held.LogProbability > best:
							likeliest, next = candidate, best
						default:
							next = max(next, held.LogProbability)
						}
					}
					if stood := own.Continuations[likeliest].LogProbability - next; len(scores) > 1 && chosen != likeliest {
						outside = outside || stood > decisiveMargin
						if stood > margin {
							t.Errorf("%s: candidate %d is found likeliest where the reference finds candidate %d, which it held %.3f above its next",
								own.Name, chosen, likeliest, stood)
						} else {
							t.Logf("%s: the likeliest candidate parts from the reference's where its first two stood %.4f apart", own.Name, stood)
						}
					} else if len(scores) > 1 {
						same++
					}
					continue
				}
				scores, err := runner.ScoreContinuationTokens(t.Context(), given.IDs, [][]tokenizer.TokenID{own.Generated})
				if err != nil {
					t.Fatalf("%s: the reference's answer was not scored: %v", own.Name, err)
				}
				apart := math.Abs(scores[0].LogProbability-own.LogProbability) / float64(len(own.Generated))
				widest = max(widest, apart)
				tokens += len(own.Generated)
				outside = outside || apart > answerTolerance
				if apart > tolerance {
					t.Errorf("%s: the reference's answer is given %.4f where the reference gives it %.4f: %.4f a token apart, over %.2f",
						own.Name, scores[0].LogProbability, own.LogProbability, apart, tolerance)
				}
				// The evaluation's own generation, but for the stop at an end of turn:
				// the reference wrote its whole length, so both are read to it.
				ids, _, err := runner.Generate(t.Context(), "", inference.GenerateOptions{
					PromptTokenIDs: given.IDs, MaxNewTokens: reference.NewTokens, Sampler: greedy, DeviceGreedy: true, ContinueAfterEOG: true,
				})
				if err != nil || len(ids) != len(given.IDs)+reference.NewTokens {
					t.Fatalf("%s: %d tokens came back for %d given and %d asked: %v", own.Name, len(ids), len(given.IDs), reference.NewTokens, err)
				}
				generated := ids[len(given.IDs):]
				if slices.Equal(generated, own.Generated) {
					same++
					continue
				}
				for position, token := range own.Generated {
					if generated[position] == token {
						continue
					}
					outside = outside || own.Margins[position] > decisiveMargin
					if own.Margins[position] > margin {
						t.Errorf("%s: token %d of the answer is %d where the reference's is %d, which it held %.3f above its next",
							own.Name, position, generated[position], token, own.Margins[position])
					} else {
						t.Logf("%s: token %d of the answer parts from the reference's where its first two stood %.4f apart",
							own.Name, position, own.Margins[position])
					}
					break
				}
			}
			switch {
			case differs && !outside:
				t.Errorf("%s stands within the common band on every case: its difference is gone, so remove its entry (%s)", name, known.reason)
			case differs:
				t.Logf("%s is held to %.2f a token and a margin of %.2f, not the common %.2f and %.2f: %s", name, tolerance, margin, answerTolerance, decisiveMargin, known.reason)
			}
			if reference.Protocol == promptChat {
				t.Logf("%s: %d of %d answers are the reference's token for token, any other parting at a near tie; the reference's answers are scored within %.4f a token over %d tokens",
					reference.Model, same, len(reference.Cases), widest, tokens)
				return
			}
			if same != 0 {
				t.Logf("%s: on %d of %d questions the candidate found likeliest is the reference's, any other parting at a near tie; the candidates are scored within %.4f a token over %d tokens",
					reference.Model, same, len(reference.Cases), widest, tokens)
				return
			}
			t.Logf("%s: %d sequences are scored within %.6f a token of the reference over %d tokens", reference.Model, len(reference.Cases), widest, tokens)
		})
	}
}
