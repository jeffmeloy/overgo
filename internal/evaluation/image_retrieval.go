package evaluation

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	// imageRetrievalKind names the suite whose cases are pictures in groups: a model that
	// embeds pictures is as good as the share of the pictures whose nearest neighbour,
	// by cosine similarity among all the others, is in their own group.
	imageRetrievalKind         = "image-retrieval"
	imageRetrievalMediaType    = "application/vnd.overgo.image-retrieval-dataset+json"
	imageRetrievalSchema       = "overgo/image-retrieval-dataset/v1"
	imageRetrievalSplitMedia   = "application/vnd.overgo.image-retrieval-split+json"
	imageRetrievalSplitSchema  = "overgo/image-retrieval-split/v1"
	imageRetrievalReportMedia  = "application/vnd.overgo.image-retrieval-report+json"
	imageRetrievalReportSchema = "overgo/image-retrieval-report/v1"
	// sceneRetrievalSource and sceneRetrievalSchema name the suite of frames of VidGen
	// clips, grouped by the source video the scenes were cut from.
	sceneRetrievalSource = "vidgen-1m/scene-retrieval"
	sceneRetrievalSchema = "overgo/vidgen-scene-retrieval/v1"
	// minimumRetrievalGroup is the fewest pictures a group holds to be asked about: a picture
	// with no sibling has no neighbour to find.
	minimumRetrievalGroup = 2
)

var (
	imageRetrievalDatasetContract = artifact.DocumentContract{
		Kind: artifact.KindDataset, MediaType: imageRetrievalMediaType, Schema: imageRetrievalSchema,
	}
	imageRetrievalSplitContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: imageRetrievalSplitMedia, Schema: imageRetrievalSplitSchema,
	}
	imageRetrievalReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: imageRetrievalReportMedia, Schema: imageRetrievalReportSchema,
	}
)

// ImageEmbedder is what an embedding session offers a suite: the vector of a picture the
// session holds, named by its file.
type ImageEmbedder interface {
	Embed(ctx context.Context, name string) ([]float32, error)
}

// ImageRetrievalCase is one picture and the group it belongs to.
type ImageRetrievalCase struct {
	Name  string `json:"name"`
	Group string `json:"group"`
	Image string `json:"image"`
}

// ImageRetrievalSuite is a set of pictures in groups.
type ImageRetrievalSuite struct {
	Kind   string               `json:"kind"`
	Schema string               `json:"schema"`
	Source string               `json:"source"`
	Cases  []ImageRetrievalCase `json:"cases"`
}

type imageRetrievalPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    ImageRetrievalSuite
}

type imageRetrievalReport struct {
	ID       artifact.ID `json:"-"`
	Version  uint16      `json:"version"`
	Plan     artifact.ID `json:"plan"`
	Dataset  artifact.ID `json:"dataset"`
	Queries  uint64      `json:"queries"`
	Correct  uint64      `json:"correct"`
	Accuracy float64     `json:"accuracy"`
}

// compileImageRetrieval validates and identifies the suite: at least one group of two
// pictures, so there is something to retrieve.
func compileImageRetrieval(suite ImageRetrievalSuite) (imageRetrievalPlan, error) {
	if suite.Kind != imageRetrievalKind || strings.TrimSpace(suite.Schema) == "" || strings.TrimSpace(suite.Source) == "" || len(suite.Cases) == 0 {
		return imageRetrievalPlan{}, errors.New("evaluation: invalid image-retrieval suite")
	}
	names := make(map[string]struct{}, len(suite.Cases))
	sizes := map[string]int{}
	for _, testCase := range suite.Cases {
		if strings.TrimSpace(testCase.Name) == "" || strings.TrimSpace(testCase.Group) == "" || strings.TrimSpace(testCase.Image) == "" {
			return imageRetrievalPlan{}, errors.New("evaluation: invalid image-retrieval case")
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return imageRetrievalPlan{}, errors.New("evaluation: duplicate image-retrieval case")
		}
		names[testCase.Name] = struct{}{}
		sizes[testCase.Group]++
	}
	if !slices.ContainsFunc(suite.Cases, func(testCase ImageRetrievalCase) bool { return sizes[testCase.Group] >= minimumRetrievalGroup }) {
		return imageRetrievalPlan{}, errors.New("evaluation: an image-retrieval suite holds no group of two pictures to retrieve")
	}
	identity, dataset, split, err := suiteIdentities(suite, suite.Cases)
	if err != nil {
		return imageRetrievalPlan{}, err
	}
	return imageRetrievalPlan{identity: identity, dataset: dataset, split: split, suite: suite}, nil
}

func bindImageRetrieval(compiled imageRetrievalPlan, authorities ExactAuthorities) (Plan, error) {
	return bindKindPlan(compiled.dataset, compiled.split, compiled.identity, compiled.suite, imageRetrievalKind, authorities)
}

// nearestSameGroup is the share of the pictures with a sibling whose nearest neighbour by
// cosine similarity, among every other picture, is a sibling, and how many pictures were asked.
func nearestSameGroup(groups []string, vectors [][]float32) (queries, correct uint64, err error) {
	normalized := make([][]float64, len(vectors))
	for index, vector := range vectors {
		var norm float64
		for _, value := range vector {
			norm += float64(value) * float64(value)
		}
		if norm == 0 || math.IsNaN(norm) || math.IsInf(norm, 0) || len(vector) != len(vectors[0]) {
			return 0, 0, fmt.Errorf("evaluation: the embedding of picture %d is empty, not finite or another width", index)
		}
		normalized[index] = make([]float64, len(vector))
		for position, value := range vector {
			normalized[index][position] = float64(value) / math.Sqrt(norm)
		}
	}
	sizes := map[string]int{}
	for _, group := range groups {
		sizes[group]++
	}
	for query := range vectors {
		if sizes[groups[query]] < minimumRetrievalGroup {
			continue
		}
		best, bestScore := -1, math.Inf(-1)
		for other := range vectors {
			if other == query {
				continue
			}
			var score float64
			for position, value := range normalized[query] {
				score += value * normalized[other][position]
			}
			if score > bestScore {
				best, bestScore = other, score
			}
		}
		queries++
		if groups[best] == groups[query] {
			correct++
		}
	}
	return queries, correct, nil
}

// evaluateImageRetrieval embeds every picture, finds each picture with a sibling's nearest
// neighbour among all the others, and publishes the share that are siblings through the
// campaign ledger.
func evaluateImageRetrieval(
	ctx context.Context,
	repository artifact.Repository,
	runtime Runtime,
	compiled imageRetrievalPlan,
	plan Plan,
) (imageRetrievalReport, error) {
	embedder, embeds := runtime.(ImageEmbedder)
	if ctx == nil || repository == nil || !embeds || compiled.identity != plan.body.CaseProfile {
		return imageRetrievalReport{}, errors.New("evaluation: image-retrieval authority differs")
	}
	contents, err := datasetContents(
		compiled.dataset, compiled.split, compiled.suite.Cases, imageRetrievalDatasetContract, imageRetrievalSplitContract,
	)
	if err != nil {
		return imageRetrievalReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return imageRetrievalReport{}, err
	}
	groups := make([]string, len(compiled.suite.Cases))
	vectors := make([][]float32, len(compiled.suite.Cases))
	progress := trackProgress(ctx, compiled.suite.Source, len(compiled.suite.Cases))
	for index, testCase := range compiled.suite.Cases {
		vector, err := embedder.Embed(ctx, testCase.Image)
		if err != nil {
			return imageRetrievalReport{}, fmt.Errorf("evaluation: image-retrieval case %q: %w", testCase.Name, err)
		}
		groups[index], vectors[index] = testCase.Group, vector
		progress.advance()
	}
	report := imageRetrievalReport{Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset}
	if report.Queries, report.Correct, err = nearestSameGroup(groups, vectors); err != nil {
		return imageRetrievalReport{}, err
	}
	report.Accuracy = float64(report.Correct) / float64(report.Queries)
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return imageRetrievalReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(
		ctx, repository, report.Plan, report.Dataset, report.ID, imageRetrievalReportContract, report,
	); err != nil {
		return imageRetrievalReport{}, err
	}
	return report, nil
}

// RetrievalPick is a clip a scene-retrieval suite asks about, and the source video it was cut from.
type RetrievalPick struct {
	Clip  string
	Group string
}

// PickRetrievalClips picks, from a directory of clips, every clip whose source video has
// another scene there, and distractors clips, each of a source video with no other, by
// seed: the scenes of one source video resemble each other, and the distractors are what
// a model must tell them from.
func PickRetrievalClips(directory string, distractors int, seed uint64) ([]RetrievalPick, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, err
	}
	bySource := map[string][]string{}
	for _, entry := range entries {
		if clip, isClip := strings.CutSuffix(entry.Name(), clipExtension); isClip {
			source, _, _ := strings.Cut(clip, sceneMarker)
			bySource[source] = append(bySource[source], clip)
		}
	}
	var picks, singles []RetrievalPick
	for _, source := range slices.Sorted(maps.Keys(bySource)) {
		clips := bySource[source]
		slices.Sort(clips)
		for _, clip := range clips {
			if len(clips) >= minimumRetrievalGroup {
				picks = append(picks, RetrievalPick{Clip: clip, Group: source})
			} else {
				singles = append(singles, RetrievalPick{Clip: clip, Group: source})
			}
		}
	}
	if len(picks) == 0 || distractors < 0 || len(singles) < distractors {
		return nil, fmt.Errorf("evaluation: %s holds %d clips with a sibling and %d without, for %d distractors", directory, len(picks), len(singles), distractors)
	}
	rng := rand.New(rand.NewPCG(seed, seed))
	rng.Shuffle(len(singles), func(left, right int) { singles[left], singles[right] = singles[right], singles[left] })
	return append(picks, singles[:distractors]...), nil
}

// BuildSceneRetrieval builds the suite of the frames cut from the picked clips, each named by
// the file the caller cut it to, which images maps the clip to.
func BuildSceneRetrieval(picks []RetrievalPick, images map[string]string) (ImageRetrievalSuite, error) {
	suite := ImageRetrievalSuite{Kind: imageRetrievalKind, Schema: sceneRetrievalSchema, Source: sceneRetrievalSource}
	for _, pick := range picks {
		image, cut := images[pick.Clip]
		if !cut {
			return ImageRetrievalSuite{}, fmt.Errorf("evaluation: clip %s has no frame", pick.Clip)
		}
		suite.Cases = append(suite.Cases, ImageRetrievalCase{Name: sceneRetrievalSource + "/" + filepath.ToSlash(pick.Clip), Group: pick.Group, Image: image})
	}
	return suite, nil
}
