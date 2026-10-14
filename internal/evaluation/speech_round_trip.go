package evaluation

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"unicode"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/runrecord"
)

const (
	// speechRoundTripKind names the suite whose cases have a speech model read a
	// text aloud and a speech recognizer, the judge, write down what it heard:
	// the model is as good as the share of its words that survive the trip.
	speechRoundTripKind         = "speech-round-trip"
	speechRoundTripMediaType    = "application/vnd.overgo.speech-round-trip-dataset+json"
	speechRoundTripSchema       = "overgo/speech-round-trip-dataset/v1"
	speechRoundTripSplitMedia   = "application/vnd.overgo.speech-round-trip-split+json"
	speechRoundTripSplitSchema  = "overgo/speech-round-trip-split/v1"
	speechRoundTripReportMedia  = "application/vnd.overgo.speech-round-trip-report+json"
	speechRoundTripReportSchema = "overgo/speech-round-trip-report/v1"
	// libriSpeechSource and libriSpeechSchema name the suite of LibriSpeech
	// test-clean's texts.
	libriSpeechSource = "librispeech-test-clean/round-trip"
	libriSpeechSchema = "overgo/librispeech-test-clean-round-trip/v1"
	// libriSpeechTextColumn is the column of the LibriSpeech parquet files that holds a
	// recording's transcript.
	libriSpeechTextColumn = "text"
	// libriSpeechRows bounds the rows of one test-clean file the builder reads.
	libriSpeechRows = 1 << 20
	// roundTripErrorMetric names a round trip's word error rate among its metrics.
	roundTripErrorMetric = "word_error_rate"
	// roundTripMinimumWords and roundTripMaximumWords bound a text the suite reads aloud:
	// a phrase too short to say anything, and a sentence long enough to span segments.
	roundTripMinimumWords = 6
	roundTripMaximumWords = 25
)

var (
	speechRoundTripDatasetContract = artifact.DocumentContract{
		Kind: artifact.KindDataset, MediaType: speechRoundTripMediaType, Schema: speechRoundTripSchema,
	}
	speechRoundTripSplitContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: speechRoundTripSplitMedia, Schema: speechRoundTripSplitSchema,
	}
	speechRoundTripReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: speechRoundTripReportMedia, Schema: speechRoundTripReportSchema,
	}
	// roundTripNormalization is how the text and the transcript are made
	// comparable in a suite that does not say: the three plain steps every
	// suite was counted with before one could.
	roundTripNormalization = []TranscriptionNormalization{TranscriptionLowercase, TranscriptionStripPunctuation, TranscriptionCollapseWhitespace}
	// writtenOneWay is how a suite built now is counted: the text and the
	// transcript both written by the recognizers' own normalizer, so that a
	// judge who writes "10" for "ten" is not heard as a speaker who said it
	// wrong.
	writtenOneWay = []TranscriptionNormalization{transcriptionWhisperEnglish}
)

// SpeechRoundTripper is what a speech session offers a suite: the model reads
// the text aloud in the voice and the judge writes down what it heard.
type SpeechRoundTripper interface {
	RoundTrip(ctx context.Context, text, voice string, seed int64) (string, error)
	// Judge names the recognizer that listens, which the score depends on.
	Judge() string
}

// SpeechRoundTripCase is one text the model reads aloud.
type SpeechRoundTripCase struct {
	Name string `json:"name"`
	Text string `json:"text"`
}

// SpeechRoundTripSuite is a set of texts, the voice that reads them, the seed
// that fixes its randomness, the judge that listens, and how the text and
// what the judge wrote are written before their words are compared. A suite
// that names no normalization is counted with the three plain steps, as every
// suite was before one could name any.
type SpeechRoundTripSuite struct {
	Kind          string                       `json:"kind"`
	Schema        string                       `json:"schema"`
	Source        string                       `json:"source"`
	Voice         string                       `json:"voice"`
	Seed          int64                        `json:"seed"`
	Judge         string                       `json:"judge"`
	Normalization []TranscriptionNormalization `json:"normalization,omitempty"`
	Cases         []SpeechRoundTripCase        `json:"cases"`
}

type speechRoundTripPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    SpeechRoundTripSuite
	// normalization is the suite's, or the plain steps of a suite that names none.
	normalization []TranscriptionNormalization
	// set is what the dataset identifies and holds: the texts, with how they
	// are counted when that is not the plain way.
	set any
}

type speechRoundTripObservation struct {
	Name       string `json:"name"`
	Transcript string `json:"transcript"`
	Errors     uint64 `json:"errors"`
	Words      uint64 `json:"words"`
}

type speechRoundTripReport struct {
	ID            artifact.ID                  `json:"-"`
	Version       uint16                       `json:"version"`
	Plan          artifact.ID                  `json:"plan"`
	Dataset       artifact.ID                  `json:"dataset"`
	Observations  []speechRoundTripObservation `json:"observations"`
	Errors        uint64                       `json:"errors"`
	Words         uint64                       `json:"words"`
	WordErrorRate float64                      `json:"word_error_rate"`
}

// compileSpeechRoundTrip validates and identifies the suite.
func compileSpeechRoundTrip(suite SpeechRoundTripSuite) (speechRoundTripPlan, error) {
	if suite.Kind != speechRoundTripKind || strings.TrimSpace(suite.Schema) == "" || strings.TrimSpace(suite.Source) == "" ||
		strings.TrimSpace(suite.Voice) == "" || strings.TrimSpace(suite.Judge) == "" || len(suite.Cases) == 0 {
		return speechRoundTripPlan{}, errors.New("evaluation: invalid speech-round-trip suite")
	}
	normalization := roundTripNormalization
	if len(suite.Normalization) != 0 {
		normalization = slices.Clone(suite.Normalization)
	}
	if err := checkTranscriptionNormalization(normalization); err != nil {
		return speechRoundTripPlan{}, err
	}
	names := make(map[string]struct{}, len(suite.Cases))
	for _, testCase := range suite.Cases {
		if strings.TrimSpace(testCase.Name) == "" || len(strings.Fields(normalizeTranscription(testCase.Text, normalization))) == 0 {
			return speechRoundTripPlan{}, errors.New("evaluation: invalid speech-round-trip case")
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return speechRoundTripPlan{}, errors.New("evaluation: duplicate speech-round-trip case")
		}
		names[testCase.Name] = struct{}{}
	}
	// What a judge is charged for a text depends on the text and on how both
	// sides are written, so a suite counted another way than the plain steps
	// is another set than the same texts counted plainly. Two contracts over
	// one set under one protocol have no order a reader could tell; two sets
	// of one measure do, the one evaluated last. The texts counted the plain
	// way are the texts alone, as they always were.
	var set any = suite.Cases
	if !slices.Equal(normalization, roundTripNormalization) {
		set = countedTexts{Normalization: normalization, Cases: suite.Cases}
	}
	identity, dataset, split, err := suiteIdentities(suite, set)
	if err != nil {
		return speechRoundTripPlan{}, err
	}
	return speechRoundTripPlan{identity: identity, dataset: dataset, split: split, suite: suite, normalization: normalization, set: set}, nil
}

// countedTexts is the set of a suite counted another way than the plain
// steps: its texts and how they and their transcripts are written.
type countedTexts struct {
	Normalization []TranscriptionNormalization `json:"normalization"`
	Cases         []SpeechRoundTripCase        `json:"cases"`
}

func bindSpeechRoundTrip(compiled speechRoundTripPlan, authorities ExactAuthorities) (Plan, error) {
	return bindKindPlan(compiled.dataset, compiled.split, compiled.identity, compiled.suite, speechRoundTripKind, authorities)
}

// evaluateSpeechRoundTrip has the model read every text aloud, has the judge
// write down what it heard, and publishes the word error rate of all the texts
// together through the campaign ledger: the edit distance of the transcripts to
// the texts, over the texts' words.
func evaluateSpeechRoundTrip(
	ctx context.Context,
	repository artifact.Repository,
	runtime Runtime,
	compiled speechRoundTripPlan,
	plan Plan,
) (speechRoundTripReport, error) {
	speaker, speaks := runtime.(SpeechRoundTripper)
	if ctx == nil || repository == nil || !speaks || compiled.identity != plan.body.CaseProfile {
		return speechRoundTripReport{}, errors.New("evaluation: speech-round-trip authority differs")
	}
	if speaker.Judge() != compiled.suite.Judge {
		return speechRoundTripReport{}, fmt.Errorf("evaluation: the suite is judged by %q, this session by %q", compiled.suite.Judge, speaker.Judge())
	}
	contents, err := datasetContents(
		compiled.dataset, compiled.split, compiled.set, speechRoundTripDatasetContract, speechRoundTripSplitContract,
	)
	if err != nil {
		return speechRoundTripReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return speechRoundTripReport{}, err
	}
	report := speechRoundTripReport{Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset}
	progress := trackProgress(ctx, compiled.suite.Source, len(compiled.suite.Cases))
	for _, testCase := range compiled.suite.Cases {
		transcript, err := speaker.RoundTrip(ctx, testCase.Text, compiled.suite.Voice, compiled.suite.Seed)
		if err != nil {
			return speechRoundTripReport{}, fmt.Errorf("evaluation: speech-round-trip case %q: %w", testCase.Name, err)
		}
		reference := strings.Fields(normalizeTranscription(testCase.Text, compiled.normalization))
		heard := strings.Fields(normalizeTranscription(transcript, compiled.normalization))
		observation := speechRoundTripObservation{
			Name: testCase.Name, Transcript: strings.ToValidUTF8(transcript, string(unicode.ReplacementChar)),
			Errors: uint64(sequenceEditDistance(reference, heard)), Words: uint64(len(reference)),
		}
		report.Observations = append(report.Observations, observation)
		report.Errors += observation.Errors
		report.Words += observation.Words
		progress.advance()
	}
	report.WordErrorRate = float64(report.Errors) / float64(report.Words)
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return speechRoundTripReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(
		ctx, repository, report.Plan, report.Dataset, report.ID, speechRoundTripReportContract, report,
	); err != nil {
		return speechRoundTripReport{}, err
	}
	return report, nil
}

// speechRoundTripMetrics reports the word error rate: lower is better.
func speechRoundTripMetrics(wordErrorRate float64) []runrecord.Metric {
	return []runrecord.Metric{{Name: roundTripErrorMetric, Value: wordErrorRate, Direction: runrecord.DirectionMinimize}}
}

// SpeechRoundTripSpec names what a LibriSpeech round-trip suite is built from:
// a parquet file of the corpus's transcripts, how many the suite reads aloud,
// the seed that picks them, and the voice and judge the suite is run under.
type SpeechRoundTripSpec struct {
	Parquet string
	Cases   int
	Seed    uint64
	Voice   string
	// SynthesisSeed fixes the speech model's randomness.
	SynthesisSeed int64
	Judge         string
}

// BuildLibriSpeechRoundTrip builds the round-trip suite of LibriSpeech
// test-clean's transcripts: Cases of those between six and twenty-five words,
// picked by seed, each said as a sentence, the way the corpus's capitals and
// missing punctuation are not. The suite is counted with the text and the
// transcript written one way, by the recognizers' own normalizer.
func BuildLibriSpeechRoundTrip(ctx context.Context, spec SpeechRoundTripSpec) (SpeechRoundTripSuite, error) {
	if spec.Cases <= 0 {
		return SpeechRoundTripSuite{}, errors.New("evaluation: a round-trip suite reads at least one text aloud")
	}
	var texts []string
	err := dataset.ReadParquetTextRows(ctx, spec.Parquet, libriSpeechTextColumn, libriSpeechRows, func(_ uint64, text string) error {
		if words := len(strings.Fields(text)); words >= roundTripMinimumWords && words <= roundTripMaximumWords {
			texts = append(texts, sentence(text))
		}
		return nil
	})
	if err != nil {
		return SpeechRoundTripSuite{}, err
	}
	return speechRoundTripSuiteOf(texts, spec)
}

// speechRoundTripSuiteOf picks the spec's cases from the usable texts by its seed.
func speechRoundTripSuiteOf(texts []string, spec SpeechRoundTripSpec) (SpeechRoundTripSuite, error) {
	if len(texts) < spec.Cases {
		return SpeechRoundTripSuite{}, fmt.Errorf("evaluation: %d usable texts in %s, fewer than the %d asked", len(texts), spec.Parquet, spec.Cases)
	}
	texts = slices.Clone(texts)
	rng := rand.New(rand.NewPCG(spec.Seed, spec.Seed))
	rng.Shuffle(len(texts), func(left, right int) { texts[left], texts[right] = texts[right], texts[left] })
	suite := SpeechRoundTripSuite{
		Kind: speechRoundTripKind, Schema: libriSpeechSchema, Source: libriSpeechSource,
		Voice: spec.Voice, Seed: spec.SynthesisSeed, Judge: spec.Judge, Normalization: slices.Clone(writtenOneWay),
	}
	for index, text := range texts[:spec.Cases] {
		suite.Cases = append(suite.Cases, SpeechRoundTripCase{Name: fmt.Sprintf("%s/%03d", libriSpeechSource, index), Text: text})
	}
	return suite, nil
}

// sentence writes a lower-case transcript as a sentence: it opens with a
// capital and closes with a full stop.
func sentence(text string) string {
	runes := []rune(strings.ToLower(strings.TrimSpace(text)))
	if len(runes) == 0 {
		return ""
	}
	runes[0] = unicode.ToUpper(runes[0])
	return string(runes) + "."
}
