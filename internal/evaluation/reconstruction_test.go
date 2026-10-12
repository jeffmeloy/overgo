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
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

// levelShift is an autoencoder that gives back every value moved by a fixed number of
// eight-bit levels, through a latent a twelfth the size of what it took. Its nudge is
// added to every value that comes back: one too small to cross a level leaves the floats
// unequal and the pixels shown the same.
type levelShift struct {
	levels map[string]uint8
	nudge  float32
	asked  []ClipCut
}

func (r *levelShift) Reconstruct(_ context.Context, clip ReconstructionCase, cut ClipCut) (RoundTrip, error) {
	r.asked = append(r.asked, cut)
	values := 3 * cut.Frames * cut.Height * cut.Width
	trip := RoundTrip{Source: make([]float32, values), Output: make([]float32, values), LatentValues: values / 12}
	for index := range trip.Source {
		trip.Source[index] = media.U8ToSignedUnitF32(100)
		trip.Output[index] = media.U8ToSignedUnitF32(100+r.levels[clip.Clip]) + r.nudge
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

// TestReconstructionScoresTheMeanLevelErrorOfEveryClip holds the report to the mean
// squared level error of each clip as its eight-bit pixels came back, their mean over
// the clips, and the latent's share of the values that went in; to giving every clip at
// the suite's cut; and to refusing a runtime that cannot reconstruct.
func TestReconstructionScoresTheMeanLevelErrorOfEveryClip(t *testing.T) {
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
	// Every value off by one level is a mean squared error of 1, and by ten of 100. Sums of
	// whole squares are exact, so the errors are held to their values and not to a tolerance.
	if len(report.Observations) != 2 || report.Observations[0].LevelError != 1 || report.Observations[1].LevelError != 100 ||
		report.LevelError != 50.5 || math.Abs(report.LatentShare-1.0/12) > 1e-12 {
		t.Fatalf("report = %+v, want level errors of 1 and 100, their mean 50.5, and a latent share of a twelfth", report)
	}
	if len(autoencoder.asked) != 2 || autoencoder.asked[0] != suite.Cut || autoencoder.asked[1] != suite.Cut {
		t.Fatalf("the clips were given at %+v, want the suite's cut %+v twice", autoencoder.asked, suite.Cut)
	}
	metrics := reconstructionMetrics(report.LevelError)
	values := map[string]float64{}
	for _, metric := range metrics {
		values[metric.Name] = metric.Value
	}
	if name, value, ok := HeadlineMetric(values); !ok || name != "mean_squared_level_error" || value != report.LevelError {
		t.Fatalf("the headline of %v is %s = %v, want the mean squared level error", values, name, value)
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

// TestReconstructionScoresAClipThatCameBackAsItWentIn holds a campaign to publishing a
// reconstruction suite when a clip's pixels came back as they went in: that clip is
// scored a level error of zero beside what the others lost, the zero is written in the
// report the store holds, and the suite's one metric is their mean, where lower is
// better. The suite once failed whole at such a clip, for want of a ratio to give it.
func TestReconstructionScoresAClipThatCameBackAsItWentIn(t *testing.T) {
	t.Parallel()
	source, err := json.Marshal(reconstructionFixture())
	if err != nil {
		t.Fatal(err)
	}
	identity := modelrecipe.ProgramIdentity{
		Model:      planID(t, artifact.KindModel, "autoencoder"),
		Definition: planID(t, artifact.KindModelDefinition, "autoencoder definition"),
		Recipe:     planID(t, artifact.KindRecipe, "autoencoder recipe"),
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "fixture", Driver: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	for name, test := range map[string]struct {
		autoencoder *levelShift
		clips       [2]float64
	}{
		"one clip exact beside one a level off": {&levelShift{levels: map[string]uint8{"ten.mp4": 1}}, [2]float64{0, 1}},
		"every clip exact":                      {&levelShift{}, [2]float64{}},
		// A ten-thousandth is far less than the half level that would show another pixel.
		"floats shown as the same bytes": {&levelShift{nudge: 1e-4}, [2]float64{}},
	} {
		t.Run(name, func(t *testing.T) {
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			if _, err := store.Commit(t.Context(), artifact.Batch{Key: "fixture/autoencoder", Artifacts: []artifact.Descriptor{{ID: identity.Model}, {ID: identity.Definition}, {ID: identity.Recipe}}}); err != nil {
				t.Fatal(err)
			}
			campaign, err := NewCampaign(store, test.autoencoder, identity, environment, planTestCommit)
			if err != nil {
				t.Fatal(err)
			}
			suite, err := CompileSuite(source, campaign.Authorities())
			if err != nil {
				t.Fatal(err)
			}
			result, err := campaign.Evaluate(t.Context(), suite)
			if err != nil {
				t.Fatalf("the suite was not published: %v", err)
			}
			want := runrecord.Metric{Name: "mean_squared_level_error", Value: (test.clips[0] + test.clips[1]) / 2, Direction: runrecord.DirectionMinimize}
			if !result.Evaluation.Valid() || len(result.Metrics) != 1 || result.Metrics[0] != want {
				t.Fatalf("the campaign published evaluation %s with metrics %+v, want one evaluation with %+v", result.Evaluation, result.Metrics, want)
			}
			content, found, err := artifact.ReadContent(t.Context(), store, result.Report)
			if err != nil || !found {
				t.Fatalf("the report is not in the store: found %t, %v", found, err)
			}
			// An error is read through a pointer, so one left out of the report for being
			// zero reads apart from one written as zero.
			var stored struct {
				Version      uint16 `json:"version"`
				Observations []struct {
					LevelError *float64 `json:"mean_squared_level_error"`
				} `json:"observations"`
				LevelError  *float64 `json:"mean_squared_level_error"`
				LatentShare float64  `json:"latent_share"`
			}
			if err := json.Unmarshal(content.Data, &stored); err != nil {
				t.Fatal(err)
			}
			if content.Descriptor.Schema != "overgo/reconstruction-report/v2" || stored.Version != artifact.SecondDocumentVersion ||
				len(stored.Observations) != len(test.clips) || stored.LevelError == nil || *stored.LevelError != want.Value ||
				math.Abs(stored.LatentShare-1.0/12) > 1e-12 {
				t.Fatalf("the stored report is %s under schema %q, want version two of the report with the suite's error written", content.Data, content.Descriptor.Schema)
			}
			for clip, observation := range stored.Observations {
				if observation.LevelError == nil || *observation.LevelError != test.clips[clip] {
					t.Errorf("clip %d is stored as %s, want a level error of %v written", clip, content.Data, test.clips[clip])
				}
			}
		})
	}
}

// oneTrip is an autoencoder that gives every clip the one round trip it holds, or fails.
type oneTrip struct {
	levelShift
	trip RoundTrip
	err  error
}

func (r *oneTrip) Reconstruct(context.Context, ReconstructionCase, ClipCut) (RoundTrip, error) {
	return r.trip, r.err
}

// TestReconstructionRefusesAClipItCouldNotMeasure holds the suite to failing, as it did
// before a clip that came back as it went in was scored, at a clip the autoencoder could
// not take through, at a latent that holds no values, and at a clip of no values or one
// that came back at another length or as no number, each of those three for its own
// reason: a clip of no values has no mean, and is not left to fail as a number that
// cannot be written.
func TestReconstructionRefusesAClipItCouldNotMeasure(t *testing.T) {
	t.Parallel()
	compiled, err := compileReconstruction(reconstructionFixture())
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
	failed := errors.New("the autoencoder failed")
	went := []float32{0, 0.5}
	for name, test := range map[string]struct {
		autoencoder oneTrip
		says        string
	}{
		"a clip the autoencoder failed at":           {autoencoder: oneTrip{err: failed}},
		"a latent that holds no values":              {autoencoder: oneTrip{trip: RoundTrip{Source: went, Output: []float32{0, 0}}}},
		"an exact clip whose latent holds no values": {autoencoder: oneTrip{trip: RoundTrip{Source: went, Output: went}}},
		"what came back at another length":           {oneTrip{trip: RoundTrip{Source: went, Output: went[:1], LatentValues: 1}}, "2 values went in and 1 came back"},
		"what came back as no number":                {oneTrip{trip: RoundTrip{Source: went, Output: []float32{0, float32(math.NaN())}, LatentValues: 1}}, "not finite"},
		"a clip of no values":                        {oneTrip{trip: RoundTrip{LatentValues: 1}}, "0 values went in and 0 came back"},
	} {
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { store.Close() })
		publishPlanFixtureAuthorities(t, store, plan)
		report, err := evaluateReconstruction(t.Context(), store, &test.autoencoder, compiled, plan)
		if err == nil || !strings.Contains(err.Error(), test.says) || test.autoencoder.err != nil && !errors.Is(err, failed) {
			t.Errorf("%s: report = %+v, %v, want the clip refused for its own reason %q", name, report, err, test.says)
		}
	}
}

// TestLevelErrorScoresARoundTripByThePixelsShown holds the mean squared level error to
// scoring a round trip that lost nothing as zero, where the ratio it replaces refused it;
// to refusing one that came back another length, as no number or as nothing; and to
// reading both sides as the eight-bit pixels shown: a value past the range is clipped to it.
func TestLevelErrorScoresARoundTripByThePixelsShown(t *testing.T) {
	t.Parallel()
	source := []float32{-1, 0, 1, 0.5}
	for name, output := range map[string][]float32{
		"another length": {-1, 0, 1},
		"no number":      {-1, 0, float32(math.NaN()), 0.5},
		"nothing at all": nil,
	} {
		if levelError, err := MeanSquaredLevelError(source, output); err == nil {
			t.Errorf("a level error of %v was given for %s", levelError, name)
		}
	}
	if levelError, err := MeanSquaredLevelError(source, []float32{-1, 0, 1, 0.5}); err != nil || levelError != 0 {
		t.Fatalf("what came back as it went in is scored %v, %v, want no error at all", levelError, err)
	}
	// 1.5 is shown as the brightest pixel, as 1 is; only the last value differs, by the 63
	// levels between 0.5, shown as 191, and 0, shown as 128: a mean squared error of 63
	// squared over four values.
	levelError, err := MeanSquaredLevelError(source, []float32{-1, 0, 1.5, 0})
	went, back := media.SignedUnitF32ToU8(0.5), media.SignedUnitF32ToU8(0)
	if want := 63 * 63 / 4.0; err != nil || went != 191 || back != 128 || levelError != want {
		t.Fatalf("level error = %v, %v, want %v from one value shown as %d that came back as %d", levelError, err, want, went, back)
	}
}

// TestLevelErrorSaysWhyItRefusedARoundTrip holds the level error, which the bridge judge
// reads outside a suite, to naming its own reason for each refusal: what came back at
// another length, a clip of no values, and what came back as no number.
func TestLevelErrorSaysWhyItRefusedARoundTrip(t *testing.T) {
	t.Parallel()
	if _, err := MeanSquaredLevelError([]float32{0, 0.5}, []float32{0}); err == nil || !strings.Contains(err.Error(), "2 values went in and 1 came back") {
		t.Fatalf("a round trip of another length was refused with %v, want the two lengths named", err)
	}
	if _, err := MeanSquaredLevelError(nil, nil); err == nil || !strings.Contains(err.Error(), "0 values went in and 0 came back") {
		t.Fatalf("a round trip of no values was refused with %v, want it told it holds none", err)
	}
	if _, err := MeanSquaredLevelError([]float32{0}, []float32{float32(math.Inf(1))}); err == nil || !strings.Contains(err.Error(), "not finite") {
		t.Fatalf("what came back past every number was refused with %v, want it told it is not finite", err)
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
