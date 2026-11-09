//go:build modeltest

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/dataset"
	"overgo/internal/media"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// The cases gemma-4-12B-it's projection suite is drawn from (docs/verification/fixtures/gemma-4-12b).
const (
	// gemmaFixtures is where the suite's own fixtures and its case list are kept.
	gemmaFixtures = "docs/verification/fixtures/gemma-4-12b"
	// gemmaCaseList is the case list: each case's prompt and its ground-truth answer, from which the publisher's
	// processor and model write the suite (scripts/python_golden_references/gemma4_projection_suite.py).
	gemmaCaseList = "cases.json"
	// gemmaWriteFixtures, set to 1, writes the fixtures and the case list in place of holding the committed ones to them.
	gemmaWriteFixtures = "OVERGO_WRITE_GEMMA_FIXTURES"
	// gemmaSpeechCases is how many LibriSpeech utterances the sound cases ask of.
	gemmaSpeechCases = 4
	// gemmaSampleRate is the rate LibriSpeech is recorded at and Gemma 4's audio encoder reads.
	gemmaSampleRate = 16000
	// gemmaShortestWord is the fewest letters an utterance's first word has to be a choice: a function word is no
	// fair question of one heard word.
	gemmaShortestWord = 4
	// gemmaLongestUtterance is the most seconds an utterance runs, to keep its audio tokens few.
	gemmaLongestUtterance = 6
	// gemmaRowsRead bounds the LibriSpeech rows read for the utterances.
	gemmaRowsRead = 200
)

// gemmaChoices are the letters a case's answer is one of.
var gemmaChoices = []string{"A", "B", "C", "D"}

// gemmaCase is one case of the list: the prompt recipe verify reads, and the answer the ground truth gives.
type gemmaCase struct {
	Name   string `json:"name"`
	Prompt string `json:"prompt"`
	Answer string `json:"answer"`
}

// TestGemmaProjectionFixturesAreDrawnFromLibriSpeech draws the case list of gemma-4-12B-it's projection suite: the
// Qwen judge's picture, picture-set and clip cases, whose answers are the colours drawn; four sound cases, each a
// LibriSpeech test-clean utterance asked which word it speaks first among the first words of the four; and two cases of
// a picture and a sound together. The sounds are kept as 16 kHz mono 16-bit WAV, LibriSpeech's own depth. With
// OVERGO_WRITE_GEMMA_FIXTURES=1 it writes them; otherwise the committed fixtures must be what it draws.
func TestGemmaProjectionFixturesAreDrawnFromLibriSpeech(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots := testcheck.Must(dataroot.Resolve(root))(t)
	parquet := filepath.Join(roots.Datasets, "librispeech_asr-clean-xet", "clean", "test", "0000.parquet")
	if _, err := os.Stat(parquet); err != nil {
		testskip.NotApplicable(t, "LibriSpeech test-clean is not on this machine: "+err.Error())
	}
	files := map[string][]byte{}
	var cases []gemmaCase
	var qwen struct {
		Cases []struct{ Name, Prompt, Text string } `json:"cases"`
	}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(root, "docs/verification/fixtures/qwen27/suite.json")))(t), &qwen))
	asked := map[string]string{}
	for _, held := range qwen.Cases {
		cases = append(cases, gemmaCase{Name: held.Name, Prompt: held.Prompt, Answer: held.Text})
		asked[held.Name] = held.Text
	}
	speech := librispeechUtterances(t, parquet)
	firstWords := make([]string, len(speech))
	for index, utterance := range speech {
		firstWords[index] = utterance.first
	}
	turn, answerTurn := "<|turn>user\n", "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
	for index, utterance := range speech {
		path := fmt.Sprintf("%s/speech-%d.wav", gemmaFixtures, index)
		encoded := testcheck.Must(media.EncodeWAVPCM16(utterance.samples, gemmaSampleRate))(t)
		files[path] = encoded
		sound := map[string]any{"kind": "audio-wav", "path": path, "artifact": testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, encoded))(t).String()}
		// The four first words are the choices of every sound case, each case's own word at its own letter.
		question := "Choices: " + lettered(firstWords) + ". Which word is spoken first? Reply with exactly one letter: A, B, C, or D."
		cases = append(cases, gemmaCase{Name: fmt.Sprintf("audio-%d-first-word", index), Answer: gemmaChoices[index],
			Prompt: promptJSON(t, map[string]any{"kind": "audio", "files": []any{sound}, "text": []string{"", question}})})
		if index >= len(qwenPictures) {
			continue
		}
		picture := qwenPictures[index]
		answer, held := asked[fmt.Sprintf("image-%d-%s", index, picture.half)]
		if !held {
			t.Fatalf("the Qwen suite asks no colour of image-%d's %s half", index, picture.half)
		}
		colour := "Choices: A = red; B = blue; C = green; D = yellow. What color is on the " + picture.half +
			" half of the image? Reply with exactly one letter: A, B, C, or D."
		cases = append(cases,
			gemmaCase{Name: fmt.Sprintf("mixed-%d-picture", index), Answer: answer,
				Prompt: promptJSON(t, map[string]any{"kind": "mixed", "files": []any{picture.file, sound}, "text": []string{turn, "", colour + answerTurn}})},
			gemmaCase{Name: fmt.Sprintf("mixed-%d-sound", index), Answer: gemmaChoices[index],
				Prompt: promptJSON(t, map[string]any{"kind": "mixed", "files": []any{picture.file, sound}, "text": []string{turn, "", question + answerTurn}})})
	}
	list := testcheck.Must(json.MarshalIndent(map[string]any{"source": "overgo/gemma-4-12b-projection-cases/v1: qwen27 pictures; librispeech test-clean 0000.parquet", "cases": cases}, "", "  "))(t)
	files[gemmaFixtures+"/"+gemmaCaseList] = append(list, '\n')
	write := os.Getenv(gemmaWriteFixtures) == "1"
	for path, data := range files {
		target := filepath.Join(root, filepath.FromSlash(path))
		if write {
			testcheck.NoError(t, os.MkdirAll(filepath.Dir(target), 0o755))
			testcheck.NoError(t, os.WriteFile(target, data, 0o644))
			continue
		}
		if held, err := os.ReadFile(target); err != nil || !bytes.Equal(held, data) {
			t.Errorf("%s is not what the cases draw (%v); write them with %s=1", path, err, gemmaWriteFixtures)
		}
	}
	t.Logf("%d cases, %d of sound: first words %v", len(cases), len(speech), firstWords)
}

// qwenPictures are the Qwen judge's fixture pictures as a prompt names them, each with the half the Qwen suite asks the
// colour of (its image-n-half case): image-0 is split left and right, image-1 top and bottom.
var qwenPictures = []struct {
	file map[string]any
	half string
}{
	{map[string]any{"kind": "image", "path": "docs/verification/fixtures/qwen27/image-0.png", "artifact": "file:sha256:cb9549be184fba6e3969811657310cf63d38b108befcc0f50345a335e07b4e7e"}, "left"},
	{map[string]any{"kind": "image", "path": "docs/verification/fixtures/qwen27/image-1.png", "artifact": "file:sha256:1465c1b16f7722254fac80a13532251298cf1ebcc7279f34df7f8d55b62d6053"}, "top"},
}

// utterance is a LibriSpeech utterance: its samples at 16 kHz and the first word of its transcript.
type utterance struct {
	samples []float32
	first   string
}

// librispeechUtterances are the first gemmaSpeechCases utterances of the parquet file that run at most
// gemmaLongestUtterance seconds and whose transcripts open with distinct words of gemmaShortestWord letters or more.
func librispeechUtterances(t *testing.T, parquet string) []utterance {
	t.Helper()
	transcripts := map[uint64]string{}
	testcheck.NoError(t, dataset.ReadParquetTextRows(t.Context(), parquet, "text", gemmaRowsRead, func(row uint64, text string) error {
		transcripts[row] = text
		return nil
	}))
	var chosen []utterance
	testcheck.NoError(t, dataset.ReadParquetTextRows(t.Context(), parquet, "audio.bytes", gemmaRowsRead, func(row uint64, flac string) error {
		if len(chosen) == gemmaSpeechCases {
			return nil
		}
		words := strings.FieldsFunc(strings.ToLower(transcripts[row]), func(r rune) bool { return !unicode.IsLetter(r) && r != '\'' })
		if len(words) == 0 || len(words[0]) < gemmaShortestWord || slices.ContainsFunc(chosen, func(held utterance) bool { return held.first == words[0] }) {
			return nil
		}
		decoded, _, err := media.DecodeAudio(t.Context(), []byte(flac), gemmaSampleRate*gemmaLongestUtterance)
		if err != nil {
			// Longer than the cases take.
			return nil
		}
		chosen = append(chosen, utterance{samples: decoded.Samples, first: words[0]})
		return nil
	}))
	if len(chosen) < gemmaSpeechCases {
		t.Fatalf("%d LibriSpeech utterances fit the sound cases, want %d", len(chosen), gemmaSpeechCases)
	}
	return chosen
}

// lettered is choices as a case lists them: "A = first; B = second; ...".
func lettered(choices []string) string {
	parts := make([]string, len(choices))
	for index, choice := range choices {
		parts[index] = gemmaChoices[index] + " = " + choice
	}
	return strings.Join(parts, "; ")
}

// promptJSON is a prompt as recipe verify reads it.
func promptJSON(t *testing.T, prompt map[string]any) string {
	t.Helper()
	return string(testcheck.Must(json.Marshal(prompt))(t))
}
