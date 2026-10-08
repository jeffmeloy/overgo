package seq2seq

import (
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/textgeneration"
	"overgo/internal/tokenizer"
	"overgo/internal/workflowruntime"
)

var generatedTextContract = artifact.JSONContract(artifact.KindFile, "overgo.seq2seq-text.v1")

// Generator binds one model to its artifact tokenizer: the piece table that
// reads ids back as text, and, for a tokenizer model that declares byte-pair
// merges, the merges that segment text; a model that declares none is
// segmented by the table as a unigram one.
type Generator struct {
	model     *Model
	tokenizer *tokenizer.Unigram
	merges    *pieceMerges
	selected  uint64
}

// toolsMarker is the piece that parts a request from the tools it may call.
// The model's own code is given the two apart and builds the encoder's input
// as the request's ids, the marker's id and the tools' ids, each text
// segmented by itself; a request in one text is cut at the marker and built
// the same way, so the ids are those the model was trained on.
const toolsMarker = "<tools>"

// maxSourceTokens is the encoder input the model's own code builds at most,
// and markerAndTool what a request leaves of it for the marker and one id of
// the tools.
const (
	maxSourceTokens = 1024
	markerAndTool   = 2
)

// segment gives one text's ids by the tokenizer model's own rule.
func (g *Generator) segment(text string) ([]int, error) {
	if g.merges != nil {
		return g.merges.encode(text)
	}
	return g.tokenizer.Encode(text)
}

// source builds the encoder's input for one request text as the model's own
// code builds it, cutting the request short before the tools and the tools
// to what room is left. With rename it first rewrites the tools' names in
// snake case, as that code does for a request it serves, and returns each new
// name's original for the answer to be written back in; a training example
// is segmented as its author wrote it.
func (g *Generator) source(text string, rename bool) ([]int, map[string]string, error) {
	marker, marked := g.tokenizer.PieceID(toolsMarker)
	request, tools, parted := strings.Cut(text, toolsMarker)
	if !marked || !parted {
		ids, err := g.segment(text)
		return ids, nil, err
	}
	var names map[string]string
	if rename {
		tools, names = normalizeTools(strings.TrimSpace(tools))
	}
	asked, err := g.segment(request)
	if err != nil {
		return nil, nil, err
	}
	listed, err := g.segment(tools)
	if err != nil {
		return nil, nil, err
	}
	asked = asked[:min(len(asked), maxSourceTokens-markerAndTool)]
	listed = listed[:min(len(listed), maxSourceTokens-len(asked)-1)]
	return slices.Concat(asked, []int{marker}, listed), names, nil
}

// OutputTokens is the number of tokens the generator's last selection
// decoded. The select stage is the incremental decode over the encoded
// source and the tokens' reading back as text, apart from the encode and
// prepare stages, so its wall is the decode's.
func (g *Generator) OutputTokens() uint64 { return g.selected }

func LoadGenerator(directory string) (*Generator, error) {
	model, err := Load(directory)
	if err != nil {
		return nil, err
	}
	pieces, unknown, err := tokenizer.ReadSentencePieceModel(filepath.Join(directory, "tokenizer.model"))
	if err != nil {
		return nil, fmt.Errorf("seq2seq: load tokenizer: %w", err)
	}
	table, err := tokenizer.NewUnigram(pieces, unknown)
	if err != nil {
		return nil, err
	}
	generator := &Generator{model: model, tokenizer: table}
	merges, declared, err := loadPieceMerges(filepath.Join(directory, "tokenizer.model"), pieces, unknown)
	if err != nil {
		return nil, fmt.Errorf("seq2seq: load tokenizer: %w", err)
	}
	if declared {
		generator.merges = merges
	}
	return generator, nil
}

// Model is the loaded model the generator decodes with and a trainer trains.
func (g *Generator) Model() *Model { return g.model }

func (g *Generator) encodeRequest(request textgeneration.Request) (encodedRequest, error) {
	if g == nil || g.model == nil || g.tokenizer == nil {
		return encodedRequest{}, errors.New("seq2seq: generator is unavailable")
	}
	if err := textgeneration.Validate(request); err != nil {
		return encodedRequest{}, err
	}
	source, names, err := g.source(request.Text, true)
	if err != nil {
		return encodedRequest{}, err
	}
	encoded, err := g.model.encodeTokens(source, request.MaxTokens)
	encoded.names = names
	return encoded, err
}

func (g *Generator) prepareGeneration(encoded encodedRequest) (textSelector, error) {
	selector, err := g.model.prepareTokens(encoded)
	if err != nil {
		return nil, err
	}
	return textGeneration{tokens: selector, generator: g, names: encoded.names}, nil
}

// Generate answers one request directly: it encodes the text, prepares the
// decode and reads back the text the decode selects, the stages a recipe's
// runtime binds one after another.
func (g *Generator) Generate(request textgeneration.Request) (string, error) {
	encoded, err := g.encodeRequest(request)
	if err != nil {
		return "", err
	}
	selector, err := g.prepareGeneration(encoded)
	if err != nil {
		return "", err
	}
	return selector.selectText()
}

// RegisterRuntime binds one loaded generator to its text stages.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, generator *Generator) error {
	if generator == nil {
		return errors.New("seq2seq: incomplete runtime binding")
	}
	return workflowruntime.RegisterJSONPipeline(
		runtime, modelID, generatedTextContract,
		modelrecipe.ModuleSeq2SeqEncode, generator.encodeRequest,
		modelrecipe.ModuleSeq2SeqPrepare, generator.prepareGeneration,
		modelrecipe.ModuleSeq2SeqSelect,
		func(selector textSelector) (string, error) {
			if selector == nil {
				return "", errors.New("seq2seq: missing text selector")
			}
			return selector.selectText()
		},
	)
}
