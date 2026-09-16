package audioparity

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"errors"
	"io"
	"math/big"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingdata"
)

type alignmentCase struct {
	Row                               uint64 `json:"row"`
	Speaker, Agent, Channel, ID, Text string
	Samples                           uint64 `json:"samples"`
	EncodedBytes                      uint64 `json:"encoded_bytes"`
	AudioSHA256                       string `json:"audio_sha256"`
}

type alignmentFixture struct {
	Dataset           artifact.ID       `json:"dataset"`
	Shard             string            `json:"shard"`
	ShardSHA256       string            `json:"shard_sha256"`
	ShardBytes        uint64            `json:"shard_bytes"`
	Rows              uint64            `json:"rows"`
	AnnotationArchive string            `json:"annotation_archive"`
	AnnotationSHA256  string            `json:"annotation_sha256"`
	AnnotationBytes   uint64            `json:"annotation_bytes"`
	MeetingsSHA256    string            `json:"meetings_sha256"`
	WordXMLSHA256     map[string]string `json:"word_xml_sha256"`
	Selection         struct {
		Meeting         string `json:"meeting"`
		ExcludedMeeting int    `json:"excluded_meeting"`
		SingleWord      int    `json:"single_word"`
		AlreadySelected int    `json:"already_selected"`
		Selected        int    `json:"selected"`
	} `json:"selection"`
	Cases   []alignmentCase `json:"cases"`
	path    string
	archive *zip.ReadCloser
}

func loadAlignmentFixture(t *testing.T, storeRoot string) alignmentFixture {
	t.Helper()
	f := loadAlignmentAnnotations(t)
	reference, err := overgodb.OpenReadOnly(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer reference.Close()
	directory, err := artifact.AvailablePath(t.Context(), reference, f.Dataset, artifact.LocationDirectory)
	if err != nil {
		t.Fatal(err)
	}
	f.path = filepath.Join(directory, filepath.FromSlash(f.Shard))
	verifyASRFile(t, f.path, alignmentFileID(t, f.ShardSHA256), f.ShardBytes)
	// Metadata and audio use the existing bounded physical-row reader.
	metadata, err := dataset.OpenParquetRows(t.Context(), f.path, []string{"meeting_id", "audio_id", "text", "speaker_id"}, adapterAcceptanceMemory)
	if err != nil {
		t.Fatal(err)
	}
	defer metadata.Close()
	if metadata.Rows() != f.Rows || f.Rows == 0 {
		t.Fatal("annotation source row denominator differs")
	}
	seen := make(map[string]bool)
	excluded, single, prior, selected := 0, 0, 0, 0
	for row := range metadata.Rows() {
		values, err := metadata.Read(t.Context(), row)
		if err != nil {
			t.Fatal(err)
		}
		for _, column := range []string{"meeting_id", "audio_id", "text", "speaker_id"} {
			if values[column] == nil {
				t.Fatalf("null %s at row %d", column, row)
			}
		}
		if *values["meeting_id"] != f.Selection.Meeting {
			excluded++
			continue
		}
		if len(strings.Fields(*values["text"])) < 2 {
			single++
			continue
		}
		if seen[*values["speaker_id"]] {
			prior++
			continue
		}
		if selected >= len(f.Cases) {
			t.Fatal("unrecorded selected speaker")
		}
		want := f.Cases[selected]
		if row != want.Row || *values["audio_id"] != want.ID || *values["text"] != want.Text || *values["speaker_id"] != want.Speaker {
			t.Fatalf("selection differs at row %d", row)
		}
		seen[want.Speaker] = true
		selected++
	}
	if excluded != f.Selection.ExcludedMeeting || single != f.Selection.SingleWord || prior != f.Selection.AlreadySelected || selected != f.Selection.Selected || selected != len(f.Cases) || selected == 0 {
		t.Fatal("complete selection denominator differs")
	}
	t.Logf("annotation selection: rows=%d excluded-known-timing-meetings=%d single-word=%d speaker-already-selected=%d selected=%d; selection precedes model outputs", f.Rows, excluded, single, prior, selected)
	return f
}

func alignmentFileID(t *testing.T, digest string) artifact.ID {
	t.Helper()
	id, err := artifact.ParseID("file:sha256:" + digest)
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func (f alignmentFixture) decodeXML(t *testing.T, name, digest string, target any) {
	t.Helper()
	var data []byte
	for _, entry := range f.archive.File {
		if entry.Name != name {
			continue
		}
		if data != nil || entry.UncompressedSize64 > f.AnnotationBytes {
			t.Fatal("duplicate or oversized annotation entry")
		}
		reader, err := entry.Open()
		if err != nil {
			t.Fatal(err)
		}
		data, err = io.ReadAll(io.LimitReader(reader, int64(entry.UncompressedSize64)+1))
		closeErr := reader.Close()
		if err != nil || closeErr != nil || uint64(len(data)) != entry.UncompressedSize64 {
			t.Fatalf("annotation read: %v %v", err, closeErr)
		}
	}
	id, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil || len(data) == 0 || id.DigestHex() != digest {
		t.Fatal("annotation entry identity differs")
	}
	// These exact reference files declare Latin-1 but contain only ASCII.
	// Refuse other bytes instead of silently interpreting a different charset.
	for _, value := range data {
		if value > 127 {
			t.Fatal("reference XML exceeds the pinned ASCII subset")
		}
	}
	decoder := xml.NewDecoder(bytes.NewReader(data))
	decoder.CharsetReader = func(label string, input io.Reader) (io.Reader, error) {
		if !strings.EqualFold(label, "ISO-8859-1") {
			return nil, errors.New("unadmitted annotation charset")
		}
		return input, nil
	}
	if err := decoder.Decode(target); err != nil {
		t.Fatal(err)
	}
}

func annotationSample(t *testing.T, value string, rate uint64) uint64 {
	t.Helper()
	number, ok := new(big.Rat).SetString(value)
	if !ok || number.Sign() < 0 {
		t.Fatal("invalid annotation timestamp")
	}
	number.Mul(number, new(big.Rat).SetInt(new(big.Int).SetUint64(rate)))
	if !number.IsInt() || !number.Num().IsUint64() {
		t.Fatal("annotation timestamp is not an exact source-sample coordinate")
	}
	return number.Num().Uint64()
}

func (f alignmentFixture) referenceWords(t *testing.T, test alignmentCase, rate uint64, transform trainingdata.TextTransform) []recipecontract.AlignedText {
	t.Helper()
	var meetings struct {
		Meetings []struct {
			Observation string `xml:"observation,attr"`
			Speakers    []struct {
				Agent   string `xml:"nxt_agent,attr"`
				Name    string `xml:"global_name,attr"`
				Channel string `xml:"channel,attr"`
			} `xml:"speaker"`
		} `xml:"meeting"`
	}
	f.decodeXML(t, "corpusResources/meetings.xml", f.MeetingsSHA256, &meetings)
	matched := 0
	for _, meeting := range meetings.Meetings {
		if meeting.Observation == f.Selection.Meeting {
			for _, speaker := range meeting.Speakers {
				if speaker.Name == test.Speaker && speaker.Agent == test.Agent && speaker.Channel == test.Channel {
					matched++
				}
			}
		}
	}
	if matched != 1 {
		t.Fatal("speaker/channel annotation binding differs")
	}
	parts := strings.Split(test.ID, "_")
	if len(parts) != 6 || parts[1] != f.Selection.Meeting || parts[2] != "H0"+test.Channel || parts[3] != test.Speaker {
		t.Fatal("source clip identity differs")
	}
	startTicks, err := strconv.ParseUint(parts[4], 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	endTicks, err := strconv.ParseUint(parts[5], 10, 64)
	if err != nil || endTicks <= startTicks {
		t.Fatal("invalid clip interval")
	}
	// The source corpus IDs encode hundredths of seconds. Exact decimal
	// conversion avoids the Parquet float32 timestamp's representational error.
	start := annotationSample(t, new(big.Rat).SetFrac(new(big.Int).SetUint64(startTicks), big.NewInt(100)).RatString(), rate)
	end := annotationSample(t, new(big.Rat).SetFrac(new(big.Int).SetUint64(endTicks), big.NewInt(100)).RatString(), rate)
	if end-start != test.Samples {
		t.Fatal("clip timestamp duration differs from decoded samples")
	}
	var document struct {
		Words []struct {
			Text        string `xml:",chardata"`
			Start       string `xml:"starttime,attr"`
			End         string `xml:"endtime,attr"`
			Punctuation string `xml:"punc,attr"`
		} `xml:"w"`
	}
	f.decodeXML(t, "words/"+f.Selection.Meeting+"."+test.Agent+".words.xml", f.WordXMLSHA256[test.Agent], &document)
	var words []recipecontract.AlignedText
	for _, word := range document.Words {
		if word.Punctuation == "true" {
			continue
		}
		if word.Start == "" || word.End == "" {
			t.Fatal("reference word has no timestamp")
		}
		a, b := annotationSample(t, word.Start, rate), annotationSample(t, word.End, rate)
		if b <= start || a >= end {
			continue
		}
		if a < start || b > end || b <= a {
			t.Fatal("reference word crosses or invalidates the selected interval")
		}
		text, err := transform.Apply(word.Text)
		if err != nil {
			t.Fatal(err)
		}
		words = append(words, recipecontract.AlignedText{Text: text, Span: recipecontract.SampleSpan{Start: a - start, End: b - start}})
	}
	var text []string
	for _, word := range words {
		text = append(text, word.Text)
	}
	want, err := transform.Apply(test.Text)
	if err != nil || strings.Join(text, " ") != want || len(words) == 0 {
		t.Fatalf("annotation text differs: %q != %q", strings.Join(text, " "), want)
	}
	return words
}

func loadAlignmentAnnotations(t *testing.T) alignmentFixture {
	t.Helper()
	var fixture alignmentFixture
	readAudioFixtureJSON(t, "testdata/ami_alignment.json", &fixture)
	path := retainedAudioFile(t, fixture.AnnotationSHA256, fixture.AnnotationBytes)
	archive, err := zip.OpenReader(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { archive.Close() })
	fixture.archive = archive
	return fixture
}
