//go:build windows && modeltest

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"image"
	_ "image/png"
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
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/sampling"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// pictureReference is one model's picture fixture: the scored file and its projector, the frames asked about under the
// references root, the question, and for each frame the digest of the prompt the projector built for it with the
// answer the reference gave that prompt, each token's log-probability and its margin over the next likeliest.
type pictureReference struct {
	Schema    string `json:"schema"`
	Model     string `json:"model"`
	File      string `json:"file"`
	Projector string `json:"projector"`
	Frames    string `json:"frames"`
	Question  string `json:"question"`
	NewTokens int    `json:"new_tokens"`
	Source    string `json:"source"`
	Cases     []struct {
		Name             string              `json:"name"`
		IDsSHA256        string              `json:"ids_sha256"`
		Generated        []tokenizer.TokenID `json:"generated"`
		LogProbabilities []float64           `json:"log_probabilities"`
		Margins          []float64           `json:"margins"`
	} `json:"cases"`
}

// TestPicturesAnswerAsTheirReferenceDoes holds the path a picture question takes, from the picture file through the
// projector's resize and encoding and the prompt it builds to the answer the model generates, to a reference that read
// the same pixels. The prompt must be the one the reference answered, by the digest of its tokens; then, token by
// token while the two answers agree, the scored file's log-probability of the reference's token must stand within the
// band its text answers are held to (knownAnswerDifferences, by the fixture's name, else the common one), and where
// its greedy choice parts from the reference's the reference must have held its own within the band's margin.
func TestPicturesAnswerAsTheirReferenceDoes(t *testing.T) {
	cudatest.Require(t)
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	fixtures, err := filepath.Glob(filepath.Join(root, "fixtures/picture_answers/*.json"))
	if err != nil || len(fixtures) == 0 {
		t.Fatalf("no model's reference picture answers are on file: %v", err)
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
		var reference pictureReference
		if err := json.Unmarshal(raw, &reference); err != nil || len(reference.Cases) == 0 {
			t.Fatalf("%s holds no case: %v", fixture, err)
		}
		name := strings.TrimSuffix(filepath.Base(fixture), filepath.Ext(fixture))
		t.Run(name, func(t *testing.T) {
			tolerance, margin := answerTolerance, decisiveMargin
			if known, differs := knownAnswerDifferences[name]; differs {
				tolerance, margin = known.tolerance, known.margin
			}
			model, projectorPath := roots.ResolveModelPath(reference.File), roots.ResolveModelPath(reference.Projector)
			frames := filepath.Join(roots.References(), filepath.FromSlash(reference.Frames))
			for _, path := range []string{model, projectorPath, frames} {
				if _, err := os.Stat(path); err != nil {
					t.Skipf("UNAVAILABLE: %s cannot be read: %v", path, err)
				}
			}
			store, err := overgodb.OpenReadOnly(roots.Store)
			if err != nil {
				t.Fatal(err)
			}
			modelID, _, err := modelcli.ResolveModel(t.Context(), store, model)
			if err != nil {
				t.Fatal(err)
			}
			reserve, bind := modelcli.ModelReserve(t.Context())
			vision, err := projector.OpenActiveSession(t.Context(), store, modelID, projectorPath, projector.OpenOptions{CUDA: true, MemoryReserve: reserve})
			if closeErr := store.Close(); err != nil || closeErr != nil {
				t.Fatal(err, closeErr)
			}
			t.Cleanup(func() { _ = vision.Close() })
			runner, err := modelcli.OpenRunner(t.Context(), roots.Store, model, inference.OpenOptions{})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = runner.Close() })
			bind(runner)
			widest, same, tokens := 0.0, 0, 0
			for _, own := range reference.Cases {
				file, err := os.Open(filepath.Join(frames, own.Name))
				if err != nil {
					t.Fatal(err)
				}
				picture, _, err := image.Decode(file)
				if closeErr := file.Close(); err != nil || closeErr != nil {
					t.Fatal(err, closeErr)
				}
				built, err := vision.BuildImagePrompt(t.Context(), runner, picture, "", reference.Question, false)
				if err != nil {
					t.Fatal(err)
				}
				ids, projected, err := inference.CompileProjectedInputs(built, runner.Spec().EmbeddingLength)
				if err != nil {
					t.Fatal(err)
				}
				if digest := promptDigest(ids); digest != own.IDsSHA256 {
					t.Fatalf("%s: the prompt the projector builds is not the one the reference answered (%s, %d tokens)", own.Name, digest, len(ids))
				}
				// Each step's logits, read where the greedy answer has followed the reference's so far.
				var steps [][]float32
				if _, _, err := runner.Generate(t.Context(), reference.Question, inference.GenerateOptions{
					PromptTokenIDs: ids, ProjectedInputs: &projected, MaxNewTokens: len(own.Generated), Sampler: greedy, ContinueAfterEOG: true,
					OnToken: func(event inference.TokenEvent) error {
						steps = append(steps, slices.Clone(event.Logits))
						return nil
					},
				}); err != nil || len(steps) != len(own.Generated) {
					t.Fatalf("%s: %d of %d steps were read: %v", own.Name, len(steps), len(own.Generated), err)
				}
				apart, followed := 0.0, true
				for position, token := range own.Generated {
					logits := steps[position]
					chosen := tokenizer.TokenID(slices.Index(logits, slices.Max(logits)))
					apart += math.Abs(logProbability(logits, token) - own.LogProbabilities[position])
					tokens++
					if chosen == token {
						continue
					}
					followed = false
					if own.Margins[position] > margin {
						t.Errorf("%s: token %d of the answer is %d where the reference's is %d, which it held %.3f above its next",
							own.Name, position, chosen, token, own.Margins[position])
					} else {
						t.Logf("%s: token %d of the answer parts from the reference's where its first two stood %.4f apart",
							own.Name, position, own.Margins[position])
					}
					apart /= float64(position + 1)
					break
				}
				if followed {
					same++
					apart /= float64(len(own.Generated))
				}
				widest = max(widest, apart)
				if apart > tolerance {
					t.Errorf("%s: the reference's tokens are given %.4f a token apart from its own log-probabilities, over %.2f", own.Name, apart, tolerance)
				}
			}
			t.Logf("%s: %d of %d answers are the reference's token for token, any other parting at a near tie; its tokens are scored within %.4f a token over %d tokens",
				reference.Model, same, len(reference.Cases), widest, tokens)
		})
	}
}

// promptDigest is the SHA-256 of a prompt's token ids, each as four little-endian bytes.
func promptDigest(ids []tokenizer.TokenID) string {
	bytes := make([]byte, 0, binaryTokenBytes*len(ids))
	for _, id := range ids {
		bytes = binary.LittleEndian.AppendUint32(bytes, uint32(id))
	}
	digest := sha256.Sum256(bytes)
	return hex.EncodeToString(digest[:])
}

// binaryTokenBytes is a token id's bytes in a prompt's digest.
const binaryTokenBytes = 4

// logProbability is a token's log-probability under logits.
func logProbability(logits []float32, token tokenizer.TokenID) float64 {
	peak := float64(slices.Max(logits))
	sum := 0.0
	for _, value := range logits {
		sum += math.Exp(float64(value) - peak)
	}
	return float64(logits[token]) - peak - math.Log(sum)
}
