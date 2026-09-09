package audioparity

import (
	"archive/zip"
	"cmp"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

type speakerSegment struct {
	Speaker    string
	Start, End uint64
}
type speakerClip struct {
	Start, End   uint64
	File, SHA256 string
	Segments     []speakerSegment
}
type speakerCorpus struct {
	SourceSamples     uint64
	Scanned, Selected int
	Clips             []speakerClip
}

func verifySpeakerCorpus(t *testing.T, root string, publication *audioPublication) (speakerCorpus, alignmentFixture) {
	t.Helper()
	var corpus speakerCorpus
	readAudioFixtureJSON(t, "testdata/speaker_corpus.json", &corpus)
	waveID := alignmentFileID(t, "50e2c8e4737850c4bf3cc24e7087c755e1c733cd91822106afd0df485ba64893")
	path := filepath.Join(root, "EN2002b.Mix-Headset.wav")
	verifyASRFile(t, path, waveID, 57179180)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	audio, _, err := media.DecodeAudio(t.Context(), data, uint64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if audio.Format.SampleRate != 16000 || audio.Format.Channels != 1 || uint64(len(audio.Samples)) != corpus.SourceSamples {
		t.Fatal("continuous source geometry differs")
	}
	archivePath := cmp.Or(os.Getenv("OVERGO_AUDIO_ALIGNMENT_ANNOTATIONS"), filepath.Join(testutil.RepoRoot(t), "tmp", "ami_public_manual_1.6.2.zip"))
	archiveID := alignmentFileID(t, "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d")
	verifyASRFile(t, archivePath, archiveID, 22887865)
	archive, err := zip.OpenReader(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { archive.Close() })
	annotations := alignmentFixture{archive: archive, AnnotationBytes: 22887865}
	publication.commit(t, artifact.Batch{Key: "speaker/corpus-authorities", Artifacts: []artifact.Descriptor{{ID: waveID, Size: 57179180}, {ID: archiveID, Size: annotations.AnnotationBytes}}}, nil)
	var segments []speakerSegment
	for i, hash := range []string{
		"a7a3a786c0fa1f3d0f68eddfb84d3b1d9058b0a4d22d181085f62ef0ac094902",
		"ff67e672097453fb9307a6b616e5ed527c78d7b243b4beefdc24cbe09afffeb4",
		"c0bb221705ce2bea0ec07b24631f53cf3970ed0b4345dca531ff69ee0aee05eb",
		"5a6ceed9d6be4a03b642cf61c9e005be31a5c66a74d1092e2f6d77552c6db4bc",
	} {
		speaker := string(rune('A' + i))
		var document struct {
			Segments []struct {
				Start string `xml:"transcriber_start,attr"`
				End   string `xml:"transcriber_end,attr"`
			} `xml:"segment"`
		}
		annotations.decodeXML(t, "segments/EN2002b."+speaker+".segments.xml", hash, &document)
		for _, segment := range document.Segments {
			start, end := annotationSample(t, segment.Start, 16000), annotationSample(t, segment.End, 16000)
			if end <= start || end > corpus.SourceSamples {
				t.Fatal("annotation extent differs")
			}
			segments = append(segments, speakerSegment{speaker, start, end})
		}
	}
	slices.SortFunc(segments, func(a, b speakerSegment) int {
		if c := cmp.Compare(a.Start, b.Start); c != 0 {
			return c
		}
		return strings.Compare(a.Speaker, b.Speaker)
	})
	selected, scanned := 0, 0
	const window = 20 * 16000 // Predeclared disjoint 20-second conformance windows.
	for start := uint64(0); start+window <= corpus.SourceSamples && selected < corpus.Selected; start += window {
		scanned++
		var clipped []speakerSegment
		names := map[string]bool{}
		for _, segment := range segments {
			if segment.End <= start || segment.Start >= start+window {
				continue
			}
			names[segment.Speaker] = true
			clipped = append(clipped, speakerSegment{segment.Speaker, max(segment.Start, start) - start, min(segment.End, start+window) - start})
		}
		if len(names) < 2 {
			continue
		}
		if selected >= len(corpus.Clips) {
			t.Fatal("selection exceeds fixture")
		}
		clip := corpus.Clips[selected]
		if clip.Start != start || clip.End != start+window || !reflect.DeepEqual(clip.Segments, clipped) {
			t.Fatal("pre-inference annotated election differs")
		}
		encoded, err := media.EncodeWAVPCM16(audio.Samples[clip.Start:clip.End], 16000)
		if err != nil {
			t.Fatal(err)
		}
		id, err := artifact.IdentifyBytes(artifact.KindFile, encoded)
		if err != nil || id.DigestHex() != clip.SHA256 {
			t.Fatal("crop differs from continuous source")
		}
		verifyASRFile(t, filepath.Join(root, filepath.Base(clip.File)), id, uint64(len(encoded)))
		publication.commit(t, artifact.Batch{Key: "speaker/crop/" + id.String(), Artifacts: []artifact.Descriptor{{ID: id, Size: uint64(len(encoded))}}, Lineage: artifact.DependencyLineage(id, waveID, archiveID)}, nil)
		selected++
	}
	if selected != corpus.Selected || scanned != corpus.Scanned || selected != len(corpus.Clips) || selected != 3 {
		t.Fatal("selection denominator differs")
	}
	return corpus, annotations
}

func speakerConditioningWords(t *testing.T, annotations alignmentFixture, clip speakerClip) []recipecontract.AlignedText {
	t.Helper()
	// Select the alphabetically first annotated speaker before reading model
	// output. These are annotation-conditioned words, not an ASR/alignment claim.
	speaker := clip.Segments[0].Speaker
	for _, segment := range clip.Segments {
		speaker = min(speaker, segment.Speaker)
	}
	hashes := map[string]string{"A": "9690a5ad6eefeee14c5be6f9d019a1daa9fbba24467a03faf4d960c7abd16984", "B": "b9265cb5ac3391b6aefefc4d19f12a9b525650cbdda951f5252cd2725a0ca819"}
	var document struct {
		Words []struct {
			Text        string `xml:",chardata"`
			Start       string `xml:"starttime,attr"`
			End         string `xml:"endtime,attr"`
			Punctuation string `xml:"punc,attr"`
		} `xml:"w"`
	}
	annotations.decodeXML(t, "words/EN2002b."+speaker+".words.xml", hashes[speaker], &document)
	var words []recipecontract.AlignedText
	for _, word := range document.Words {
		if word.Punctuation != "" || word.Start == "" || word.End == "" {
			continue
		}
		start, end := annotationSample(t, word.Start, 16000), annotationSample(t, word.End, 16000)
		if end <= start || start < clip.Start || end > clip.End {
			continue
		}
		words = append(words, recipecontract.AlignedText{Span: recipecontract.SampleSpan{Start: start - clip.Start, End: end - clip.Start}, Text: strings.TrimSpace(word.Text)})
	}
	if len(words) == 0 {
		t.Fatal("conditioning word denominator absent")
	}
	return words
}
