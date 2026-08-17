// Package scratchmodel derives a causal model from corpus facts.
package scratchmodel

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
	"strings"
	"unicode/utf8"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

// ScratchArchitecture is the constructed topology's registered name in the
// executor's architecture-profile vocabulary: constructed and inherited
// components are named, validated and policy-driven through the same registry.
const ScratchArchitecture = "adaptive-causal"

type CorpusFacts struct {
	Documents []string
	Seed      int64
	Steps     int
}

// DerivationProfile owns corpus-to-topology policy.
type DerivationProfile struct {
	Version            string  `json:"version"`
	SplitDenominator   int     `json:"split_denominator"`
	StableSplitMinimum int     `json:"stable_split_minimum"`
	MinimumLayers      int     `json:"minimum_layers"`
	MinimumMLPFactor   int     `json:"minimum_mlp_factor"`
	MLPBudget          int     `json:"mlp_budget"`
	Epsilon            float64 `json:"epsilon"`
	MuonMomentum       float64 `json:"muon_momentum"`
}

func AdaptiveDerivationProfile() DerivationProfile {
	return DerivationProfile{
		Version:          "adaptive-corpus-derivation-v1",
		SplitDenominator: 10, StableSplitMinimum: 30,
		MinimumLayers: 2, MinimumMLPFactor: 2, MLPBudget: 256,
		Epsilon: 1e-8, MuonMomentum: 29.0 / 31.0,
	}
}

func (p DerivationProfile) validate() error {
	if strings.TrimSpace(p.Version) == "" || strings.ContainsAny(p.Version, "\x00\r\n") ||
		p.SplitDenominator <= 0 || p.StableSplitMinimum <= 0 ||
		p.MinimumLayers <= 0 || p.MinimumMLPFactor <= 0 || p.MLPBudget <= 0 ||
		p.Epsilon <= 0 || math.IsNaN(p.Epsilon) || math.IsInf(p.Epsilon, 0) ||
		p.MuonMomentum <= 0 || p.MuonMomentum >= 1 || math.IsNaN(p.MuonMomentum) || math.IsInf(p.MuonMomentum, 0) {
		return errors.New("scratch model: invalid derivation profile")
	}
	return nil
}

type Split struct {
	Train      []string `json:"train"`
	Validation []string `json:"validation"`
	Test       []string `json:"test"`
}

type Config struct {
	VocabSize       int            `json:"vocab_size"`
	BlockSize       int            `json:"block_size"`
	Embedding       int            `json:"n_embd"`
	HeadDim         int            `json:"head_dim"`
	HeadCount       int            `json:"n_head"`
	LayerCount      int            `json:"n_layer"`
	MLPWidth        int            `json:"mlp_width"`
	AttentionWindow int            `json:"attn_window"`
	BaseLR          float64        `json:"base_lr"`
	InitStd         float64        `json:"init_std"`
	Epsilon         float64        `json:"eps"`
	MuonMomentum    float64        `json:"muon_momentum"`
	Characters      []string       `json:"uchars"`
	BOS             int            `json:"BOS"`
	CharacterIndex  map[string]int `json:"char_to_idx"`
	EstimatedParams int            `json:"estimated_n_params"`
}

type Initializer string

const (
	InitializerUniform Initializer = "uniform"
	InitializerZero    Initializer = "zero"
)

type Parameter struct {
	Name        string      `json:"name"`
	Role        string      `json:"role"`
	Rows        int         `json:"rows"`
	Cols        int         `json:"cols"`
	Initializer Initializer `json:"initializer"`
	TiedTo      string      `json:"tied_to,omitempty"`
	Digest      string      `json:"sha256"`
}

type Construction struct {
	id           artifact.ID
	dataset      artifact.ID
	splitID      artifact.ID
	config       Config
	split        Split
	parameters   []Parameter
	weights      []float32
	bindings     map[string]parameterBinding
	seed         int64
	authority    trainingprogram.ScratchConstruction
	optimizer    optimizer.Plan
	program      trainingprogram.TrainingProgram
	processor    artifact.ID
	architecture model.ArchitectureProfile
}

type parameterBinding struct{ start, end int }

func Compile(facts CorpusFacts, profile DerivationProfile) (Construction, error) {
	if len(facts.Documents) < 3 || facts.Steps <= 0 {
		return Construction{}, errors.New("scratch model: invalid corpus facts")
	}
	if err := profile.validate(); err != nil {
		return Construction{}, err
	}
	for _, document := range facts.Documents {
		if !utf8.ValidString(document) || utf8.RuneCountInString(document) == 0 {
			return Construction{}, errors.New("scratch model: corpus document is empty or not UTF-8")
		}
	}

	datasetID, err := artifact.JSONID(artifact.KindDataset, facts.Documents)
	if err != nil {
		return Construction{}, err
	}
	split := splitDocuments(facts.Documents, facts.Seed, profile)
	splitID, err := artifact.JSONID(artifact.KindDatasetShard, split)
	if err != nil {
		return Construction{}, err
	}
	config := deriveConfig(split.Train, facts.Steps, profile)
	weights, bindings, parameters := initialize(config, facts.Seed)
	optimizerPlan, program, err := compileSharedProgram(parameters, bindings, len(weights))
	if err != nil {
		return Construction{}, err
	}

	derivationProfile, err := artifact.JSONID(artifact.KindProfile, struct {
		Profile DerivationProfile `json:"profile"`
		Steps   int               `json:"steps"`
	}{Profile: profile, Steps: facts.Steps})

	if err != nil {
		return Construction{}, err
	}
	topologyProfile, err := artifact.JSONID(artifact.KindProfile, struct {
		Version string `json:"version"`
		Config  Config `json:"config"`
	}{Version: "adaptive-causal-topology-v1", Config: config})

	if err != nil {
		return Construction{}, err
	}
	tokenizerID, err := artifact.JSONID(artifact.KindTokenizer, struct {
		Version    string         `json:"version"`
		Characters []string       `json:"characters"`
		BOS        int            `json:"bos"`
		Index      map[string]int `json:"index"`
	}{"adaptive-rune-tokenizer-v1", config.Characters, config.BOS, config.CharacterIndex})

	if err != nil {
		return Construction{}, err
	}
	processorID, err := artifact.JSONID(artifact.KindProfile, "adaptive-rune-document-v1")
	if err != nil {
		return Construction{}, err
	}
	manifestID, err := artifact.JSONID(artifact.KindTensorInventory, struct {
		Version    string      `json:"version"`
		Parameters []Parameter `json:"parameters"`
	}{Version: "adaptive-causal-parameters-v1", Parameters: parameters})

	if err != nil {
		return Construction{}, err
	}
	initializerProfile, err := artifact.JSONID(artifact.KindProfile, struct {
		Version string  `json:"version"`
		Seed    int64   `json:"seed"`
		Scale   float64 `json:"scale"`
	}{Version: "adaptive-uniform-zero-v1", Seed: facts.Seed, Scale: config.InitStd})

	if err != nil {
		return Construction{}, err
	}
	modelID, err := artifact.JSONID(artifact.KindModel, struct {
		Topology artifact.ID `json:"topology"`
		Manifest artifact.ID `json:"manifest"`
	}{topologyProfile, manifestID})

	if err != nil {
		return Construction{}, err
	}
	recipeID, err := artifact.JSONID(artifact.KindRecipe, struct {
		Dataset artifact.ID `json:"dataset"`
		Split   artifact.ID `json:"split"`
		Model   artifact.ID `json:"model"`
	}{datasetID, splitID, modelID})

	if err != nil {
		return Construction{}, err
	}
	rngAlgorithm, err := artifact.JSONID(artifact.KindProfile, "go-math-rand-v1")
	if err != nil {
		return Construction{}, err
	}
	baseSeed := uint64(facts.Seed)
	authority, err := trainingprogram.CompileScratchConstruction(trainingprogram.ScratchSpec{
		Recipe: recipeID, Dataset: datasetID, Split: splitID,
		DerivationProfile: derivationProfile, TopologyProfile: topologyProfile,
		Tokenizer: tokenizerID, ParameterManifest: manifestID,
		InitializerProfile: initializerProfile, InitializedModel: modelID,
		RNGStreams: []trainingprogram.RNGStreamSpec{
			{Name: "augmentation", Algorithm: rngAlgorithm, Seed: baseSeed + 3},
			{Name: "data", Algorithm: rngAlgorithm, Seed: baseSeed + 2},
			{Name: "init", Algorithm: rngAlgorithm, Seed: baseSeed},
			{Name: "split", Algorithm: rngAlgorithm, Seed: baseSeed},
		},
	})
	if err != nil {
		return Construction{}, err
	}
	architecture, registered := model.LookupArchitecture(ScratchArchitecture)
	if !registered {
		return Construction{}, fmt.Errorf("scratch model: architecture %q is not registered", ScratchArchitecture)
	}
	if err := model.ValidateArchitectureProfile(architecture); err != nil {
		return Construction{}, err
	}
	return Construction{
		id: modelID, dataset: datasetID, splitID: splitID, config: cloneConfig(config),
		split: cloneSplit(split), parameters: slices.Clone(parameters), weights: weights, bindings: bindings, seed: facts.Seed,
		authority: authority, optimizer: optimizerPlan, program: program,
		processor: processorID, architecture: architecture,
	}, nil
}

func (c Construction) ID() artifact.ID      { return c.id }
func (c Construction) Dataset() artifact.ID { return c.dataset }
func (c Construction) SplitID() artifact.ID { return c.splitID }

// Architecture emits the constructed topology in the executor's registered
// profile vocabulary -- the interchangeability contract with inherited models.
func (c Construction) Architecture() model.ArchitectureProfile        { return c.architecture }
func (c Construction) Split() Split                                   { return cloneSplit(c.split) }
func (c Construction) Parameters() []Parameter                        { return slices.Clone(c.parameters) }
func (c Construction) Authority() trainingprogram.ScratchConstruction { return c.authority }
func (c Construction) OptimizerPlan() optimizer.Plan                  { return c.optimizer }
func (c Construction) Program() trainingprogram.TrainingProgram       { return c.program }
func (c Construction) Processor() artifact.ID                         { return c.processor }

func (c Construction) Weights(name string) ([]float32, bool) {
	values, ok := c.weightView(name)
	return slices.Clone(values), ok
}

func (c Construction) weightView(name string) ([]float32, bool) {
	binding, ok := c.bindings[name]
	if !ok || binding.start < 0 || binding.start > binding.end || binding.end > len(c.weights) {
		return nil, false
	}
	return c.weights[binding.start:binding.end], true
}

func (c Construction) Tokens(document string) ([]int, error) {
	tokens := make([]int, 1, utf8.RuneCountInString(document)+2)
	tokens[0] = c.config.BOS
	for _, character := range document {
		index, ok := c.config.CharacterIndex[string(character)]
		if !ok {
			return nil, fmt.Errorf("scratch model: character %q absent from tokenizer", character)
		}
		tokens = append(tokens, index)
	}
	return append(tokens, c.config.BOS), nil
}

func splitDocuments(documents []string, seed int64, profile DerivationProfile) Split {
	indices := make([]int, len(documents))
	for index := range indices {
		indices[index] = index
	}
	rand.New(rand.NewSource(seed)).Shuffle(len(indices), func(left, right int) {
		indices[left], indices[right] = indices[right], indices[left]
	})
	validationCount := max(1, len(indices)/profile.SplitDenominator)
	testCount := max(1, len(indices)/profile.SplitDenominator)
	if len(indices)/3 >= profile.StableSplitMinimum {
		validationCount = max(validationCount, profile.StableSplitMinimum)
		testCount = max(testCount, profile.StableSplitMinimum)
	}
	if validationCount+testCount >= len(indices) {
		validationCount, testCount = 1, 1
	}
	trainEnd := len(indices) - validationCount - testCount
	validationEnd := len(indices) - testCount
	materialize := func(source []int) []string {
		result := make([]string, len(source))
		for index, document := range source {
			result[index] = documents[document]
		}
		return result
	}
	return Split{
		Train:      materialize(indices[:trainEnd]),
		Validation: materialize(indices[trainEnd:validationEnd]),
		Test:       materialize(indices[validationEnd:]),
	}
}

func deriveConfig(documents []string, steps int, profile DerivationProfile) Config {
	counts := map[rune]int{}
	maxLength, tokenCount := 0, 0
	for _, document := range documents {
		length := utf8.RuneCountInString(document)
		maxLength = max(maxLength, length)
		tokenCount += length
		for _, character := range document {
			counts[character]++
		}
	}
	characters := make([]string, 0, len(counts))
	for character := range counts {
		characters = append(characters, string(character))
	}
	sort.Strings(characters)
	index := make(map[string]int, len(characters))
	for position, character := range characters {
		index[character] = position
	}
	vocabSize := len(characters) + 1
	bitsNeeded := ceilLog2(max(vocabSize, 2)) + ceilLog2(max(maxLength+2, 2))
	blockSize := max(bitsNeeded, maxLength+1)
	contextRank := int(math.Ceil(math.Sqrt(float64(max(maxLength, 1)))))
	embedding := max(bitsNeeded, 1<<ceilLog2(max(bitsNeeded, contextRank)))
	headDim := max(1, int(math.Round(math.Log2(float64(max(embedding, 1))))))
	headCount := embedding / headDim
	for headCount > 1 && embedding%headCount != 0 {
		headCount--
	}
	headCount = max(1, headCount)
	headDim = embedding / headCount
	effectiveVocab := effectiveVocabulary(counts, tokenCount)
	mlpFactor := int(math.Ceil(math.Sqrt(float64(vocabSize)) + effectiveVocab/float64(embedding)))
	mlpFactor = max(profile.MinimumMLPFactor, min(mlpFactor, int(math.Round(float64(profile.MLPBudget)/math.Sqrt(float64(vocabSize))))))
	layerCount := deriveLayerCount(steps, blockSize, embedding, tokenCount, profile.MinimumLayers)
	estimated := 2*vocabSize*embedding + blockSize*embedding + layerCount*(4*embedding*embedding+2*embedding*(embedding*mlpFactor))
	return Config{
		VocabSize: vocabSize, BlockSize: blockSize, Embedding: embedding,
		HeadDim: headDim, HeadCount: headCount, LayerCount: layerCount,
		MLPWidth: embedding * mlpFactor, AttentionWindow: min(blockSize, embedding),
		BaseLR: 1 / math.Sqrt(float64(estimated)), InitStd: 1 / math.Sqrt(float64(embedding)),
		Epsilon: profile.Epsilon, MuonMomentum: profile.MuonMomentum, Characters: characters, BOS: len(characters),
		CharacterIndex: index, EstimatedParams: estimated,
	}
}

func deriveLayerCount(steps, blockSize, embedding, tokenCount, minimumLayers int) int {
	rho := float64(steps*blockSize*embedding) / float64(tokenCount)
	request := int(math.Floor(math.Sqrt(rho)))
	scale := 1
	for scale <= request/2 {
		scale *= 2
	}
	return min(max(minimumLayers, minimumLayers*scale), max(minimumLayers, embedding/2))
}

func effectiveVocabulary(counts map[rune]int, total int) float64 {
	var entropy float64
	for _, count := range counts {
		probability := float64(count) / float64(total)
		entropy -= probability * math.Log2(probability)
	}
	return math.Exp2(entropy)
}

type parameterShape struct {
	name, role  string
	rows, cols  int
	initializer Initializer
	digest      string
}

func initialize(config Config, seed int64) ([]float32, map[string]parameterBinding, []Parameter) {
	rng := rand.New(rand.NewSource(seed))
	parameterCount := 2*config.VocabSize*config.Embedding + config.BlockSize*config.Embedding +
		config.BlockSize + config.LayerCount*config.HeadCount +
		config.LayerCount*(4*config.Embedding*config.Embedding+2*config.MLPWidth*config.Embedding)
	weights := make([]float32, 0, parameterCount)
	bindings := make(map[string]parameterBinding, 5+4*config.LayerCount)
	shapes := make([]parameterShape, 0, 5+4*config.LayerCount)
	add := func(name, role string, rows, cols int, initializer Initializer) {
		values := make([]float32, rows*cols)
		digestValues := make([]float64, len(values))
		if initializer == InitializerUniform {
			for index := range values {
				digestValues[index] = (rng.Float64()*2 - 1) * config.InitStd
				values[index] = float32(digestValues[index])
			}
		}
		start := len(weights)
		weights = append(weights, values...)
		bindings[name] = parameterBinding{start: start, end: len(weights)}
		shapes = append(shapes, parameterShape{name, role, rows, cols, initializer, digestFloats(digestValues)})
	}
	add("wte", "token-embedding", config.VocabSize, config.Embedding, InitializerUniform)
	add("lm_head", "output-projection", config.VocabSize, config.Embedding, InitializerUniform)
	add("wpe", "position-embedding", config.BlockSize, config.Embedding, InitializerUniform)
	add("pos_bias", "position-bias", 1, config.BlockSize, InitializerZero)
	add("lt", "layer-temperature", config.LayerCount, config.HeadCount, InitializerZero)
	for layer := 0; layer < config.LayerCount; layer++ {
		prefix := fmt.Sprintf("l%d.", layer)
		add(prefix+"wqkv", "attention-qkv", 3*config.Embedding, config.Embedding, InitializerUniform)
		add(prefix+"wo", "attention-output", config.Embedding, config.Embedding, InitializerUniform)
		add(prefix+"w1", "mlp-expand", config.MLPWidth, config.Embedding, InitializerUniform)
		add(prefix+"w2", "mlp-contract", config.Embedding, config.MLPWidth, InitializerUniform)
	}
	sort.Slice(shapes, func(left, right int) bool { return shapes[left].name < shapes[right].name })
	parameters := make([]Parameter, len(shapes))
	for index, shape := range shapes {
		parameters[index] = Parameter{
			Name: shape.name, Role: shape.role, Rows: shape.rows, Cols: shape.cols,
			Initializer: shape.initializer, Digest: shape.digest,
		}
	}
	return weights, bindings, parameters
}

func compileSharedProgram(parameters []Parameter, bindings map[string]parameterBinding, parameterCount int) (optimizer.Plan, trainingprogram.TrainingProgram, error) {
	ordered := slices.Clone(parameters)
	sort.Slice(ordered, func(left, right int) bool {
		return bindings[ordered[left].Name].start < bindings[ordered[right].Name].start
	})
	groups := make([]optimizer.GroupSpec, len(ordered))
	manifest := make([]trainingprogram.ParameterSpec, len(ordered))
	for index, parameter := range ordered {
		binding, ok := bindings[parameter.Name]
		if !ok {
			return optimizer.Plan{}, trainingprogram.TrainingProgram{}, fmt.Errorf("scratch model: parameter %q has no slab binding", parameter.Name)
		}
		groups[index] = optimizer.GroupSpec{
			Name: parameter.Name, Start: binding.start, End: binding.end,
			Rows: parameter.Rows, Cols: parameter.Cols,
		}
		manifest[index] = trainingprogram.ParameterSpec{
			Name: parameter.Name, Rows: parameter.Rows, Cols: parameter.Cols, Trainable: true,
		}
	}
	plan, err := optimizer.CompilePlan(parameterCount, groups)
	if err != nil {
		return optimizer.Plan{}, trainingprogram.TrainingProgram{}, err
	}
	program, err := trainingprogram.CompileTrainingProgram(trainingprogram.ProgramSpec{
		Objective: trainingprogram.ObjectiveTokenPrediction,
		Operators: []trainingprogram.OperatorSpec{
			{ID: scratchOperatorBatch, Phase: trainingprogram.PhaseBatch},
			{ID: scratchOperatorForward, Phase: trainingprogram.PhaseForward},
			{ID: scratchOperatorLossVJP, Phase: trainingprogram.PhaseBackward},
			{ID: scratchOperatorMuon, Phase: trainingprogram.PhaseOptimize},
			{ID: scratchOperatorEvaluate, Phase: trainingprogram.PhaseEvaluate},
		},
		Parameters: manifest,
		Optimizer:  plan,
	})
	return plan, program, err
}

const (
	scratchOperatorBatch    = "token-batch"
	scratchOperatorForward  = "adaptive-causal-forward"
	scratchOperatorLossVJP  = "adaptive-causal-loss-vjp"
	scratchOperatorMuon     = "muon"
	scratchOperatorEvaluate = "held-out-cross-entropy"
)

func digestFloats(values []float64) string {
	digest := sha256.New()
	var encoded [8]byte
	for _, value := range values {
		binary.LittleEndian.PutUint64(encoded[:], math.Float64bits(value))
		digest.Write(encoded[:])
	}
	return hex.EncodeToString(digest.Sum(nil))
}

func ceilLog2(value int) int { return int(math.Ceil(math.Log2(float64(value)))) }

func cloneConfig(config Config) Config {
	config.Characters = slices.Clone(config.Characters)
	source := config.CharacterIndex
	config.CharacterIndex = make(map[string]int, len(config.CharacterIndex))
	for character, index := range source {
		config.CharacterIndex[character] = index
	}
	return config
}

func cloneSplit(split Split) Split {
	return Split{Train: slices.Clone(split.Train), Validation: slices.Clone(split.Validation), Test: slices.Clone(split.Test)}
}

func cloneWeights(source map[string][]float64) map[string][]float64 {
	result := make(map[string][]float64, len(source))
	for name, values := range source {
		result[name] = slices.Clone(values)
	}
	return result
}
