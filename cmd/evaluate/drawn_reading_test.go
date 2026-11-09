package main

import (
	"context"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// scriptedReader reads each picture as its script says, row by row.
type scriptedReader struct {
	rows   map[string][]string
	reader evaluation.MediaRuntime
}

func (s scriptedReader) read(_ context.Context, name string, _ image.Image) (textReading, error) {
	reading := textReading{Picture: name, Reader: s.reader}
	for _, text := range s.rows[name] {
		reading.Rows = append(reading.Rows, textRow{Kind: "text", Text: text})
	}
	return reading, nil
}

// TestDrawnTextScoresTheCharactersReadBack holds -read-suite's reading of a drawn-text suite: each picture a case asks
// about is read once from the suite's picture directory, its rows kept as the reading of that picture with the reader's
// identities, so the suite's score is the share of the asked characters read back. A picture read whole scores 1 and
// one read with a character wrong scores below it.
func TestDrawnTextScoresTheCharactersReadBack(t *testing.T) {
	t.Parallel()
	items := testcheck.Must(evaluation.PickDrawnText(2, 7))(t)
	pictures := t.TempDir()
	images, rows := map[string]string{}, map[string][]string{}
	for index, item := range items {
		name := item.ID + ".png"
		images[item.ID] = name
		file := testcheck.Must(os.Create(filepath.Join(pictures, name)))(t)
		testcheck.NoError(t, png.Encode(file, image.NewGray(image.Rect(0, 0, 1, 1))))
		testcheck.NoError(t, file.Close())
		for _, question := range item.Questions {
			rows[name] = append(rows[name], question.Answer)
		}
		if index == 1 {
			rows[name][0] = "#" + rows[name][0][1:]
		}
	}
	suite := testcheck.Must(evaluation.BuildDrawnText(items, images, "judge"))(t)
	reader := evaluation.MediaRuntime{
		Model: testutil.ArtifactID(t, artifact.KindModel, "reader"), Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "inference"),
		Projection: testutil.ArtifactID(t, artifact.KindRecipe, "projection"),
	}
	testcheck.NoError(t, readSuite(t.Context(), scriptedReader{rows: rows, reader: reader}, &suite, pictures))
	if suite.Readings == nil || suite.Readings.Reader != reader || len(suite.Readings.Read) != len(items) ||
		suite.Readings.Read[items[0].ID+".png"] != strings.Join(rows[items[0].ID+".png"], "\n") {
		t.Fatalf("the suite's readings are %+v, want each picture's rows under its name with the reader", suite.Readings)
	}
	cases := map[string][]evaluation.GeneratedAnswerCase{}
	for _, testCase := range suite.Cases {
		cases[testCase.Group] = append(cases[testCase.Group], testCase)
	}
	whole := suite.Readings.Metrics(cases[items[0].ID])[0].Value
	short := suite.Readings.Metrics(cases[items[1].ID])[0].Value
	t.Logf("the picture read whole scores %.4f, the one with a character wrong %.4f", whole, short)
	if whole != 1 || short >= whole {
		t.Fatalf("the picture read whole scores %v and the one with a character wrong %v", whole, short)
	}
	testcheck.Must(evaluation.CompileGeneratedAnswer(suite))(t)
}
