package evaluation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/media"
	"overgo/internal/overgodb"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

// levelShift is an autoencoder that gives back every value moved by a fixed number of
// eight-bit levels, through a latent a twelfth the size of what it took.
type levelShift struct {
	levels map[string]uint8
	asked  []ClipCut
}

func (r *levelShift) Reconstruct(_ context.Context, clip ReconstructionCase, cut ClipCut) (RoundTrip, error) {
	r.asked = append(r.asked, cut)
	values := 3 * cut.Frames * cut.Height * cut.Width
	trip := RoundTrip{Source: make([]float32, values), Output: make([]float32, values), LatentValues: values / 12}
	for index := range trip.Source {
		trip.Source[index] = media.U8ToSignedUnitF32(100)
		trip.Output[index] = media.U8ToSignedUnitF32(100 + r.levels[clip.Clip])
	}
	return trip, nil
}

func (*levelShift) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("an autoencoder generates no text")
}

func (*levelShift) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("an autoencoder scores no continuation")
}

func reconstructionFixture() ReconstructionSuite {
	digest := strings.Repeat("ab", sha256.Size)
	return ReconstructionSuite{
		Kind: reconstructionKind, Schema: "fixture/v1", Source: "fixture",
		Cut: ClipCut{Frames: 5, FPS: 16, Width: 8, Height: 8},
		Cases: []ReconstructionCase{
			{Name: "one level", Clip: "one.mp4", SHA256: digest},
			{Name: "ten levels", Clip: "ten.mp4", SHA256: digest},
		},
	}
}

// TestReconstructionScoresTheMeanSignalToNoiseOfEveryClip holds the report to the peak
// signal-to-noise ratio of each clip as its eight-bit pixels came back, their mean over
// the clips, and the latent's share of the values that went in; to giving every clip at
// the suite's cut; and to refusing a runtime that cannot reconstruct.
func TestReconstructionScoresTheMeanSignalToNoiseOfEveryClip(t *testing.T) {
	t.Parallel()
	suite := reconstructionFixture()
	compiled, err := compileReconstruction(suite)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := bindReconstruction(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)
	autoencoder := &levelShift{levels: map[string]uint8{"one.mp4": 1, "ten.mp4": 10}}
	report, err := evaluateReconstruction(t.Context(), store, autoencoder, compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	// Every value off by one level is a mean squared error of 1, and by ten of 100: the
	// peak of 255 squared over each, in decibels.
	one, ten := 10*math.Log10(255*255), 10*math.Log10(255*255/100.0)
	if len(report.Observations) != 2 || math.Abs(report.Observations[0].PSNR-one) > 1e-9 || math.Abs(report.Observations[1].PSNR-ten) > 1e-9 ||
		math.Abs(report.PSNR-(one+ten)/2) > 1e-9 || math.Abs(report.LatentShare-1.0/12) > 1e-12 {
		t.Fatalf("report = %+v, want %.4f and %.4f dB and a latent share of a twelfth", report, one, ten)
	}
	if len(autoencoder.asked) != 2 || autoencoder.asked[0] != suite.Cut || autoencoder.asked[1] != suite.Cut {
		t.Fatalf("the clips were given at %+v, want the suite's cut %+v twice", autoencoder.asked, suite.Cut)
	}
	metrics := reconstructionMetrics(report.PSNR)
	values := map[string]float64{}
	for _, metric := range metrics {
		values[metric.Name] = metric.Value
	}
	if name, value, ok := HeadlineMetric(values); !ok || name != "psnr_db" || value != report.PSNR {
		t.Fatalf("the headline of %v is %s = %v, want the signal-to-noise ratio", values, name, value)
	}
	if _, err := evaluateReconstruction(t.Context(), store, textOnlyRuntime{}, compiled, plan); err == nil {
		t.Fatal("a runtime that cannot reconstruct was evaluated")
	}
	// The suite file compiles through the same door every suite does.
	data, err := json.Marshal(suite)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := compileSuite(data, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	}); err != nil {
		t.Fatalf("the suite file does not compile: %v", err)
	}
}

// TestSignalToNoiseRefusesWhatIsNoRoundTrip holds the ratio to refusing a round trip that
// lost nothing, one that came back another length, and one that is no number, and to
// reading both sides as the eight-bit pixels shown: a value past the range is clipped to it.
func TestSignalToNoiseRefusesWhatIsNoRoundTrip(t *testing.T) {
	t.Parallel()
	source := []float32{-1, 0, 1, 0.5}
	for name, output := range map[string][]float32{
		"what went in":   {-1, 0, 1, 0.5},
		"another length": {-1, 0, 1},
		"no number":      {-1, 0, float32(math.NaN()), 0.5},
		"nothing at all": nil,
	} {
		if ratio, err := PeakSignalToNoise(source, output); err == nil {
			t.Errorf("a ratio of %v was given for %s", ratio, name)
		}
	}
	// 1.5 is shown as the brightest pixel, as 1 is; only the last value differs, by the 64
	// levels between 0.5 and 0: a mean squared error of 64 squared over four values.
	ratio, err := PeakSignalToNoise(source, []float32{-1, 0, 1.5, 0})
	levels := float64(media.SignedUnitF32ToU8(0.5)) - float64(media.SignedUnitF32ToU8(0))
	if want := 10 * math.Log10(255*255*4/(levels*levels)); err != nil || math.Abs(ratio-want) > 1e-9 {
		t.Fatalf("ratio = %v, %v, want %v", ratio, err, want)
	}
}

// TestReconstructionRefusesWhatItCannotCompile holds the compiler to a cut that is a
// size and a length, and to cases that name a file inside the media root by a digest.
func TestReconstructionRefusesWhatItCannotCompile(t *testing.T) {
	t.Parallel()
	if _, err := compileReconstruction(reconstructionFixture()); err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*ReconstructionSuite){
		"no frames":                  func(s *ReconstructionSuite) { s.Cut.Frames = 0 },
		"no size":                    func(s *ReconstructionSuite) { s.Cut.Width = 0 },
		"no rate":                    func(s *ReconstructionSuite) { s.Cut.FPS = 0 },
		"a clip outside the root":    func(s *ReconstructionSuite) { s.Cases[0].Clip = "../one.mp4" },
		"a digest that is not one":   func(s *ReconstructionSuite) { s.Cases[0].SHA256 = "abcd" },
		"a repeated case":            func(s *ReconstructionSuite) { s.Cases[1].Name = s.Cases[0].Name },
		"no cases":                   func(s *ReconstructionSuite) { s.Cases = nil },
		"another kind":               func(s *ReconstructionSuite) { s.Kind = imageRetrievalKind },
		"a case with no name at all": func(s *ReconstructionSuite) { s.Cases[0].Name = " " },
	} {
		suite := reconstructionFixture()
		change(&suite)
		if _, err := compileReconstruction(suite); err == nil {
			t.Errorf("a suite with %s compiled", name)
		}
	}
	// The same clips at another cut are another set to measure on.
	wider := reconstructionFixture()
	wider.Cut.Width = 16
	left, err := compileReconstruction(reconstructionFixture())
	if err != nil {
		t.Fatal(err)
	}
	right, err := compileReconstruction(wider)
	if err != nil {
		t.Fatal(err)
	}
	if left.dataset == right.dataset {
		t.Fatal("two cuts of the same clips are one dataset")
	}
}

// TestClipReconstructionSuiteBindsEachClipToItsBytes holds the builder to drawing the
// number of clips asked, the same ones for the same seed, each bound to the digest of its
// file, under a source that names the cut; and to refusing more clips than there are.
func TestClipReconstructionSuiteBindsEachClipToItsBytes(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	contents := map[string]string{"a-Scene-0001.mp4": "first", "b-Scene-0001.mp4": "second", "c-Scene-0002.mp4": "third", "notes.txt": "no clip"}
	for name, content := range contents {
		if err := os.WriteFile(filepath.Join(directory, name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	cut := ClipCut{Frames: 21, FPS: 16, Width: 832, Height: 480}
	suite, err := BuildClipReconstruction(directory, 2, 17, cut)
	if err != nil {
		t.Fatal(err)
	}
	if suite.Source != "vidgen-1m/reconstruction/21x480x832" || suite.Cut != cut || len(suite.Cases) != 2 {
		t.Fatalf("suite = %+v, want two clips under a source that names the cut", suite)
	}
	for _, testCase := range suite.Cases {
		sum := sha256.Sum256([]byte(contents[testCase.Clip]))
		if contents[testCase.Clip] == "" || testCase.SHA256 != hex.EncodeToString(sum[:]) ||
			testCase.Name != suite.Source+"/"+strings.TrimSuffix(testCase.Clip, ".mp4") {
			t.Fatalf("case %+v is not a clip of the directory bound to its bytes", testCase)
		}
	}
	again, err := BuildClipReconstruction(directory, 2, 17, cut)
	if err != nil || again.Cases[0] != suite.Cases[0] || again.Cases[1] != suite.Cases[1] {
		t.Fatalf("the same seed drew %+v after %+v, %v", again.Cases, suite.Cases, err)
	}
	if _, err := compileReconstruction(suite); err != nil {
		t.Fatalf("the suite does not compile: %v", err)
	}
	if _, err := BuildClipReconstruction(directory, 4, 17, cut); err == nil {
		t.Fatal("four clips were drawn from three")
	}
	// A cut that crops is another set under a source that says so; a suite
	// built before cuts could crop keeps the identity it had.
	cropped, err := BuildClipReconstruction(directory, 2, 17, ClipCut{Frames: 1, FPS: 16, Width: 256, Height: 256, Crop: true})
	if err != nil || cropped.Source != "vidgen-1m/reconstruction/1x256x256-cropped" {
		t.Fatalf("a cropped cut is named %q, %v", cropped.Source, err)
	}
	stretched := cropped
	stretched.Cut.Crop = false
	left, err := compileReconstruction(cropped)
	if err != nil {
		t.Fatal(err)
	}
	right, err := compileReconstruction(stretched)
	if err != nil {
		t.Fatal(err)
	}
	if left.dataset == right.dataset {
		t.Fatal("a cropped and a stretched cut of the same clips are one dataset")
	}
	encoded, err := json.Marshal(suite.Cut)
	if err != nil || strings.Contains(string(encoded), "crop") {
		t.Fatalf("a cut that does not crop is written as %s, %v", encoded, err)
	}
}
