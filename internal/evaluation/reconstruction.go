package evaluation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/runrecord"
)

const (
	// reconstructionKind names the suite whose cases are clips an autoencoder takes into its
	// latent and gives back: it is as good as what comes back is close to what went in.
	reconstructionKind         = "reconstruction"
	reconstructionMediaType    = "application/vnd.overgo.reconstruction-dataset+json"
	reconstructionSchema       = "overgo/reconstruction-dataset/v1"
	reconstructionSplitMedia   = "application/vnd.overgo.reconstruction-split+json"
	reconstructionSplitSchema  = "overgo/reconstruction-split/v1"
	reconstructionReportMedia  = "application/vnd.overgo.reconstruction-report+json"
	reconstructionReportSchema = "overgo/reconstruction-report/v1"
	// clipReconstructionSource and clipReconstructionSchema name the suites of held-out VidGen
	// clips, each cut to the same number of frames at the same size; a suite's source ends in
	// its cut, frames by height by width.
	clipReconstructionSource = "vidgen-1m/reconstruction"
	clipReconstructionSchema = "overgo/vidgen-reconstruction/v1"
	// decibelsPerBel states a ratio of powers in decibels: ten times its common logarithm.
	decibelsPerBel = 10
)

var (
	reconstructionDatasetContract = artifact.DocumentContract{
		Kind: artifact.KindDataset, MediaType: reconstructionMediaType, Schema: reconstructionSchema,
	}
	reconstructionSplitContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: reconstructionSplitMedia, Schema: reconstructionSplitSchema,
	}
	reconstructionReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: reconstructionReportMedia, Schema: reconstructionReportSchema,
	}
)

// ClipCut is how a reconstruction suite gives every clip to the autoencoder: its first
// Frames frames at FPS a second, each scaled to Width by Height. One frame is a picture.
// Crop keeps a frame's proportions, scaling it until it covers the size and taking its
// middle, where a cut without it scales the frame to the size as it is.
type ClipCut struct {
	Frames int  `json:"frames"`
	FPS    int  `json:"fps"`
	Width  int  `json:"width"`
	Height int  `json:"height"`
	Crop   bool `json:"crop,omitzero"`
}

// RoundTrip is a clip as it went into an autoencoder and as it came back: the same count
// of values in the same order, each in the signed unit range, and how many values the
// latent between them held.
type RoundTrip struct {
	Source       []float32
	Output       []float32
	LatentValues int
}

// ClipReconstructor is what an autoencoder session offers a suite: the round trip of a clip
// of the media root, cut as the suite says, once the clip's bytes are shown to be the
// digest the case carries.
type ClipReconstructor interface {
	Reconstruct(ctx context.Context, clip ReconstructionCase, cut ClipCut) (RoundTrip, error)
}

// ReconstructionCase is one clip: its file in the media root and the digest of that file.
type ReconstructionCase struct {
	Name   string `json:"name"`
	Clip   string `json:"clip"`
	SHA256 string `json:"sha256"`
}

// ReconstructionSuite is a set of clips and the cut each is given at.
type ReconstructionSuite struct {
	Kind   string               `json:"kind"`
	Schema string               `json:"schema"`
	Source string               `json:"source"`
	Cut    ClipCut              `json:"cut"`
	Cases  []ReconstructionCase `json:"cases"`
}

// reconstructionDataset is what a reconstruction suite measures on: the clips and the cut
// they are given at, since the same clips at another size are another set.
type reconstructionDataset struct {
	Cut   ClipCut              `json:"cut"`
	Cases []ReconstructionCase `json:"cases"`
}

func (suite ReconstructionSuite) dataset() reconstructionDataset {
	return reconstructionDataset{Cut: suite.Cut, Cases: suite.Cases}
}

type reconstructionPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    ReconstructionSuite
}

type reconstructionObservation struct {
	Name string  `json:"name"`
	PSNR float64 `json:"psnr_db"`
}

type reconstructionReport struct {
	ID           artifact.ID                 `json:"-"`
	Version      uint16                      `json:"version"`
	Plan         artifact.ID                 `json:"plan"`
	Dataset      artifact.ID                 `json:"dataset"`
	Observations []reconstructionObservation `json:"observations"`
	PSNR         float64                     `json:"psnr_db"`
	LatentShare  float64                     `json:"latent_share"`
}

// compileReconstruction validates and identifies the suite.
func compileReconstruction(suite ReconstructionSuite) (reconstructionPlan, error) {
	cut := suite.Cut
	if suite.Kind != reconstructionKind || strings.TrimSpace(suite.Schema) == "" || strings.TrimSpace(suite.Source) == "" ||
		len(suite.Cases) == 0 || cut.Frames <= 0 || cut.FPS <= 0 || cut.Width <= 0 || cut.Height <= 0 {
		return reconstructionPlan{}, errors.New("evaluation: invalid reconstruction suite")
	}
	names := make(map[string]struct{}, len(suite.Cases))
	for _, testCase := range suite.Cases {
		digest, err := hex.DecodeString(testCase.SHA256)
		if strings.TrimSpace(testCase.Name) == "" || !filepath.IsLocal(testCase.Clip) || err != nil || len(digest) != sha256.Size {
			return reconstructionPlan{}, errors.New("evaluation: invalid reconstruction case")
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return reconstructionPlan{}, errors.New("evaluation: duplicate reconstruction case")
		}
		names[testCase.Name] = struct{}{}
	}
	identity, err := artifact.JSONID(artifact.KindProfile, suite)
	if err != nil {
		return reconstructionPlan{}, err
	}
	dataset, err := artifact.JSONID(artifact.KindDataset, suite.dataset())
	if err != nil {
		return reconstructionPlan{}, err
	}
	split, err := artifact.JSONID(artifact.KindDatasetShard, struct {
		Dataset artifact.ID `json:"dataset"`
	}{Dataset: dataset})
	if err != nil {
		return reconstructionPlan{}, err
	}
	return reconstructionPlan{identity: identity, dataset: dataset, split: split, suite: suite}, nil
}

func bindReconstruction(compiled reconstructionPlan, authorities ExactAuthorities) (Plan, error) {
	return bindKindPlan(compiled.dataset, compiled.split, compiled.identity, compiled.suite, reconstructionKind, authorities)
}

// PeakSignalToNoise is the peak signal-to-noise ratio, in decibels, of what came back
// against what went in, both read as the eight-bit pixels a viewer is shown. A round trip
// that changes nothing has no ratio and is refused: nothing was lost for it to measure.
func PeakSignalToNoise(source, output []float32) (float64, error) {
	if len(source) == 0 || len(source) != len(output) {
		return 0, fmt.Errorf("evaluation: %d values went in and %d came back", len(source), len(output))
	}
	var squared float64
	for index, value := range source {
		back := output[index]
		if math.IsNaN(float64(back)) || math.IsInf(float64(back), 0) {
			return 0, errors.New("evaluation: what came back is not finite")
		}
		difference := float64(media.SignedUnitF32ToU8(value)) - float64(media.SignedUnitF32ToU8(back))
		squared += difference * difference
	}
	if squared == 0 {
		return 0, errors.New("evaluation: what came back is what went in, so the round trip lost nothing to measure")
	}
	return decibelsPerBel * math.Log10(math.MaxUint8*math.MaxUint8*float64(len(source))/squared), nil
}

// evaluateReconstruction passes every clip through the autoencoder and publishes, through
// the campaign ledger, the mean over the clips of the peak signal-to-noise ratio of what
// came back, and the latents' values as a share of the clips' values.
func evaluateReconstruction(
	ctx context.Context,
	repository artifact.Repository,
	runtime Runtime,
	compiled reconstructionPlan,
	plan Plan,
) (reconstructionReport, error) {
	reconstructor, reconstructs := runtime.(ClipReconstructor)
	if ctx == nil || repository == nil || !reconstructs || compiled.identity != plan.body.CaseProfile {
		return reconstructionReport{}, errors.New("evaluation: reconstruction authority differs")
	}
	contents, err := datasetContents(
		compiled.dataset, compiled.split, compiled.suite.dataset(), reconstructionDatasetContract, reconstructionSplitContract,
	)
	if err != nil {
		return reconstructionReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return reconstructionReport{}, err
	}
	report := reconstructionReport{Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset}
	var total, sourceValues, latentValues float64
	progress := trackProgress(ctx, compiled.suite.Source, len(compiled.suite.Cases))
	for _, testCase := range compiled.suite.Cases {
		trip, err := reconstructor.Reconstruct(ctx, testCase, compiled.suite.Cut)
		if err != nil {
			return reconstructionReport{}, fmt.Errorf("evaluation: reconstruction case %q: %w", testCase.Name, err)
		}
		ratio, err := PeakSignalToNoise(trip.Source, trip.Output)
		if err != nil || trip.LatentValues <= 0 {
			return reconstructionReport{}, fmt.Errorf("evaluation: reconstruction case %q: a latent of %d values: %w", testCase.Name, trip.LatentValues, err)
		}
		report.Observations = append(report.Observations, reconstructionObservation{Name: testCase.Name, PSNR: ratio})
		total += ratio
		sourceValues += float64(len(trip.Source))
		latentValues += float64(trip.LatentValues)
		progress.advance()
	}
	report.PSNR = total / float64(len(report.Observations))
	report.LatentShare = latentValues / sourceValues
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return reconstructionReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(
		ctx, repository, report.Plan, report.Dataset, report.ID, reconstructionReportContract, report,
	); err != nil {
		return reconstructionReport{}, err
	}
	return report, nil
}

// reconstructionMetrics reports the mean peak signal-to-noise ratio: higher is better. The
// latent's share of the input's values is the price the ratio was bought at and has no
// better direction of its own, so it stays in the report beside the ratio and is no metric.
func reconstructionMetrics(ratio float64) []runrecord.Metric {
	return []runrecord.Metric{{Name: "psnr_db", Value: ratio, Direction: runrecord.DirectionMaximize}}
}

// BuildClipReconstruction builds the suite of count clips of a directory, picked by seed
// and each bound to the digest of its file, to be given to an autoencoder at one cut.
func BuildClipReconstruction(directory string, count int, seed uint64, cut ClipCut) (ReconstructionSuite, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return ReconstructionSuite{}, err
	}
	var clips []string
	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), clipExtension) {
			clips = append(clips, entry.Name())
		}
	}
	slices.Sort(clips)
	if count <= 0 || len(clips) < count {
		return ReconstructionSuite{}, fmt.Errorf("evaluation: %s holds %d clips, too few to draw %d", directory, len(clips), count)
	}
	rng := rand.New(rand.NewPCG(seed, seed))
	rng.Shuffle(len(clips), func(left, right int) { clips[left], clips[right] = clips[right], clips[left] })
	// The suite is named by its cut, so a set of pictures and a set of clips read apart.
	source := fmt.Sprintf("%s/%dx%dx%d", clipReconstructionSource, cut.Frames, cut.Height, cut.Width)
	if cut.Crop {
		source += "-cropped"
	}
	suite := ReconstructionSuite{Kind: reconstructionKind, Schema: clipReconstructionSchema, Source: source, Cut: cut}
	for _, clip := range clips[:count] {
		data, err := os.ReadFile(filepath.Join(directory, clip))
		if err != nil {
			return ReconstructionSuite{}, err
		}
		sum := sha256.Sum256(data)
		suite.Cases = append(suite.Cases, ReconstructionCase{
			Name: source + "/" + strings.TrimSuffix(clip, clipExtension), Clip: clip, SHA256: hex.EncodeToString(sum[:]),
		})
	}
	return suite, nil
}
