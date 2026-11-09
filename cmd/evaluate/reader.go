package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelcli"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/sampling"
)

// How a reader reads a picture's text: Unlimited-OCR's document parsing, greedy, with the n-gram ban of its reference
// decode (references/fixtures/unlimitedocr_golden.json), whose tiles were not split.
const (
	readerQuestion = "document parsing."
	// readerRepeatSpan and readerRepeatWindow are the reference decode's ban: no run of 35 tokens repeats within 128.
	readerRepeatSpan   = 35
	readerRepeatWindow = 128
	// readerTokens bounds one reading: the reference page's 29 rows took 811 tokens.
	readerTokens = 4096
	// boxSides are a row box's edges: left, top, right and bottom.
	boxSides = 4
)

// textReader reads the text a picture holds with an OCR model through its projector: a second instrument beside a
// suite's judge, so the reading and the judging are not one model's.
type textReader struct {
	projectionRuntime
}

// textRow is one row a reader read: its kind (title, text and so on), its box on the page in the reader's thousandths,
// and its text.
type textRow struct {
	Kind string        `json:"kind"`
	Box  [boxSides]int `json:"box"`
	Text string        `json:"text"`
}

// textReading is what a reader read off one picture, row by row, with who read it: the reader's model, recipe and
// projection.
type textReading struct {
	Picture string                  `json:"picture"`
	Rows    []textRow               `json:"rows"`
	Reader  evaluation.MediaRuntime `json:"reader"`
}

// openTextReader opens the OCR model at modelPath and its projector at projectorPath on the card of device, under the
// model's active projection recipe.
func openTextReader(ctx context.Context, store *overgodb.Store, repository, modelPath, projectorPath string, device int) (*textReader, error) {
	model, _, err := modelcli.ResolveModel(ctx, store, modelPath)
	if err != nil {
		return nil, err
	}
	reserve, bind := modelcli.ModelReserve(ctx)
	vision, err := projector.OpenActiveSession(ctx, store, model, projectorPath, projector.OpenOptions{CUDA: true, DeviceOrdinal: device, MemoryReserve: reserve, DisableDynamicTiles: true})
	if err != nil {
		return nil, err
	}
	runner, err := modelcli.OpenRunner(ctx, repository, modelPath, inference.OpenOptions{DeviceOrdinal: device})
	if err != nil {
		return nil, errors.Join(err, vision.Close())
	}
	bind(runner)
	reader := &textReader{projectionRuntime{runner: runner, vision: vision}}
	if reader.identity, err = projectionIdentity(ctx, store, reader.projectionRuntime); err != nil {
		return nil, errors.Join(err, reader.Close())
	}
	return reader, nil
}

// Close releases the projector and the model.
func (r *textReader) Close() error { return errors.Join(r.vision.Close(), r.runner.Close()) }

// read reads the text of picture, named name in what it records.
func (r *textReader) read(ctx context.Context, name string, picture image.Image) (textReading, error) {
	built, err := r.vision.BuildImagePrompt(ctx, r.runner, picture, "", readerQuestion, false)
	if err != nil {
		return textReading{}, fmt.Errorf("evaluate: project %s for its reading: %w", name, err)
	}
	ids, projected, err := inference.CompileProjectedInputs(built, r.runner.Spec().EmbeddingLength)
	if err != nil {
		return textReading{}, err
	}
	config := r.runner.RuntimePolicy().Interactive.Sampling.Config()
	config.Temperature, config.TopK, config.TopP = 0, 1, 1
	config.NoRepeatNgramSize, config.NgramWindow = readerRepeatSpan, readerRepeatWindow
	sampler, err := sampling.New(config)
	if err != nil {
		return textReading{}, err
	}
	generated, _, err := r.runner.Generate(ctx, "", inference.GenerateOptions{
		MaxNewTokens: readerTokens, Sampler: sampler, DeviceGreedy: true, PromptTokenIDs: ids, ProjectedInputs: &projected,
	})
	if err != nil {
		return textReading{}, fmt.Errorf("evaluate: read %s: %w", name, err)
	}
	text, err := r.runner.Detokenize(generated[len(ids):], inference.RenderText)
	if err != nil {
		return textReading{}, err
	}
	return textReading{Picture: name, Rows: readRows(text), Reader: r.identity}, nil
}

// readRows splits a reading into its rows, each written "kind [x1, y1, x2, y2]text"; a line that holds no box is a row
// of its own text with no kind.
func readRows(text string) []textRow {
	var rows []textRow
	for line := range strings.Lines(strings.TrimSpace(text)) {
		line = strings.TrimRight(line, "\r\n")
		row := textRow{Text: line}
		opened, closed := strings.Index(line, "["), strings.Index(line, "]")
		if opened > 0 && closed > opened {
			fields := strings.Split(line[opened+1:closed], ",")
			box, boxed := [boxSides]int{}, len(fields) == boxSides
			for index, field := range fields {
				value, err := strconv.Atoi(strings.TrimSpace(field))
				boxed = boxed && err == nil
				if boxed {
					box[index] = value
				}
			}
			if boxed {
				row = textRow{Kind: strings.TrimSpace(line[:opened]), Box: box, Text: line[closed+1:]}
			}
		}
		rows = append(rows, row)
	}
	return rows
}

// writeReadings reads each picture of directory, in name order, with the reader at modelPath and projectorPath, and
// prints each reading as a line of JSON.
func writeReadings(ctx context.Context, repository, directory, modelPath, projectorPath string, device int) error {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return err
	}
	repository, err = dataroot.StoreRoot(repository)
	if err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	reader, err := openTextReader(ctx, store, repository, modelPath, projectorPath, device)
	if err != nil {
		return err
	}
	read := 0
	for _, entry := range entries {
		if !slices.Contains(pictureExtensions, strings.ToLower(filepath.Ext(entry.Name()))) {
			continue
		}
		file, err := os.Open(filepath.Join(directory, entry.Name()))
		if err != nil {
			return errors.Join(err, reader.Close())
		}
		picture, _, err := image.Decode(file)
		if err = errors.Join(err, file.Close()); err != nil {
			return errors.Join(fmt.Errorf("evaluate: decode %s: %w", entry.Name(), err), reader.Close())
		}
		reading, err := reader.read(ctx, entry.Name(), picture)
		if err != nil {
			return errors.Join(err, reader.Close())
		}
		line, err := json.Marshal(reading)
		if err != nil {
			return errors.Join(err, reader.Close())
		}
		fmt.Println(string(line))
		read++
	}
	fmt.Fprintf(os.Stderr, "evaluate: read %d pictures of %s\n", read, directory)
	return reader.Close()
}
