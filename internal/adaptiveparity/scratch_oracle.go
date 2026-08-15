package adaptiveparity

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	ScratchOracleSchema    = "overgo/adaptive-scratch-oracle/v1"
	ScratchOracleMediaType = "application/vnd.overgo.adaptive-scratch-oracle+json"
)

type ScratchConfig struct {
	VocabSize        int            `json:"vocab_size"`
	BlockSize        int            `json:"block_size"`
	Embedding        int            `json:"n_embd"`
	HeadDim          int            `json:"head_dim"`
	Heads            int            `json:"n_head"`
	Layers           int            `json:"n_layer"`
	MLPWidth         int            `json:"mlp_width"`
	AttentionWindow  int            `json:"attn_window"`
	BaseLearningRate float64        `json:"base_lr"`
	Initialization   float64        `json:"init_std"`
	Epsilon          float64        `json:"eps"`
	MuonMomentum     float64        `json:"muon_momentum"`
	Characters       []string       `json:"uchars"`
	BOS              int            `json:"BOS"`
	CharacterToIndex map[string]int `json:"char_to_idx"`
	EstimatedParams  int            `json:"estimated_n_params"`
}

type ScratchGroupOracle struct {
	Name               string `json:"name"`
	Rows               int    `json:"rows"`
	Cols               int    `json:"cols"`
	WeightSHA256       string `json:"weight_sha256"`
	GradientSHA256     string `json:"gradient_sha256"`
	Gradient1e12SHA256 string `json:"gradient_1e12_sha256"`
	UpdateSHA256       string `json:"update_sha256"`
	Update1e8SHA256    string `json:"update_1e8_sha256"`
}

type ScratchOracle struct {
	Schema            string               `json:"schema"`
	SourceCommit      string               `json:"source_commit"`
	SourceEntrypoints []string             `json:"source_entrypoints"`
	Seed              int64                `json:"seed"`
	Steps             int                  `json:"steps"`
	Documents         []string             `json:"documents"`
	Train             []string             `json:"train"`
	Validation        []string             `json:"validation"`
	Test              []string             `json:"test"`
	Config            ScratchConfig        `json:"config"`
	ParameterCount    int                  `json:"parameter_count"`
	Tokens            [][]int              `json:"tokens"`
	BatchSize         int                  `json:"batch_size"`
	FirstLoss         float64              `json:"first_loss"`
	LossHistory       []float64            `json:"loss_history"`
	FinalValLoss      float64              `json:"final_val_loss"`
	LossTolerance     float64              `json:"loss_tolerance"`
	Groups            []ScratchGroupOracle `json:"groups"`
	ID                artifact.ID          `json:"-"`
}

var scratchOracleCodec = artifact.DocumentCodec[ScratchOracle]{
	Name: "adaptive scratch oracle",
	Contract: artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: ScratchOracleMediaType, Schema: ScratchOracleSchema,
	},
	Decode: func(data []byte, oracle *ScratchOracle) error { return strictjson.DecodeBytes(data, oracle) },
	Encode: func(oracle ScratchOracle) ([]byte, error) {
		oracle.ID = artifact.ID{}
		return json.Marshal(oracle)
	},
	Canonicalize: validateScratchOracle,
	Clone:        cloneScratchOracle,
	Identity:     func(oracle ScratchOracle) artifact.ID { return oracle.ID },
	SetIdentity:  func(oracle *ScratchOracle, id artifact.ID) { oracle.ID = id },
}

func ParseScratchOracle(data []byte) (ScratchOracle, error) {
	return scratchOracleCodec.Parse(data)
}

func NormalizeScratchOracle(data []byte) (ScratchOracle, []byte, error) {
	return scratchOracleCodec.Normalize(data)
}

func (o ScratchOracle) Content() (artifact.Content, error) {
	return scratchOracleCodec.Content(o)
}

func (o ScratchOracle) CorpusID() (artifact.ID, error) {
	return artifact.JSONID(artifact.KindDataset, o.Documents)
}

func (o ScratchOracle) SplitID() (artifact.ID, error) {
	return artifact.JSONID(artifact.KindDatasetShard, struct {
		Train      []string `json:"train"`
		Validation []string `json:"validation"`
		Test       []string `json:"test"`
	}{Train: o.Train, Validation: o.Validation, Test: o.Test})

}

func validateScratchOracle(oracle *ScratchOracle) error {
	if oracle == nil || oracle.Schema != ScratchOracleSchema || !validCommit(oracle.SourceCommit) {
		return errors.New("adaptive scratch oracle: invalid header")
	}
	if len(oracle.SourceEntrypoints) == 0 || !slices.IsSorted(oracle.SourceEntrypoints) {
		return errors.New("adaptive scratch oracle: invalid source entrypoints")
	}
	for _, entrypoint := range oracle.SourceEntrypoints {
		if strings.TrimSpace(entrypoint) == "" || strings.ContainsAny(entrypoint, "\\\r\x00") {
			return errors.New("adaptive scratch oracle: invalid source entrypoint")
		}
	}
	if oracle.Steps <= 0 || oracle.BatchSize <= 0 || len(oracle.Documents) < 3 ||
		len(oracle.Train) == 0 || len(oracle.Validation) == 0 || len(oracle.Test) == 0 {
		return errors.New("adaptive scratch oracle: invalid run geometry")
	}
	if len(oracle.Train)+len(oracle.Validation)+len(oracle.Test) != len(oracle.Documents) ||
		!sameDocumentMultiset(oracle.Documents, oracle.Train, oracle.Validation, oracle.Test) {
		return errors.New("adaptive scratch oracle: split does not partition documents")
	}
	if err := validateScratchConfig(oracle.Config); err != nil {
		return err
	}
	if len(oracle.Tokens) != len(oracle.Train) {
		return errors.New("adaptive scratch oracle: token batch count differs from train split")
	}
	for index := range oracle.Train {
		if err := validateScratchTokens(oracle.Train[index], oracle.Tokens[index], oracle.Config); err != nil {
			return fmt.Errorf("adaptive scratch oracle: train row %d: %w", index, err)
		}
	}
	if len(oracle.LossHistory) != oracle.Steps || oracle.FirstLoss != oracle.LossHistory[0] ||
		!finite(oracle.FirstLoss) || !finite(oracle.FinalValLoss) || !positiveFinite(oracle.LossTolerance) {
		return errors.New("adaptive scratch oracle: invalid loss trajectory")
	}
	for _, loss := range oracle.LossHistory {
		if !finite(loss) {
			return errors.New("adaptive scratch oracle: non-finite loss")
		}
	}
	if len(oracle.Groups) == 0 {
		return errors.New("adaptive scratch oracle: parameter groups absent")
	}
	params := 0
	previous := ""
	for _, group := range oracle.Groups {
		if group.Name == "" || group.Name <= previous || group.Rows <= 0 || group.Cols <= 0 ||
			!sha256Hex(group.WeightSHA256) || !sha256Hex(group.GradientSHA256) ||
			!sha256Hex(group.Gradient1e12SHA256) || !sha256Hex(group.UpdateSHA256) || !sha256Hex(group.Update1e8SHA256) {
			return errors.New("adaptive scratch oracle: invalid parameter group")
		}
		if group.Rows > math.MaxInt/group.Cols {
			return errors.New("adaptive scratch oracle: parameter count overflow")
		}
		params += group.Rows * group.Cols
		previous = group.Name
	}
	if oracle.ParameterCount <= 0 || params != oracle.ParameterCount {
		return fmt.Errorf("adaptive scratch oracle: groups cover %d parameters, manifest reports %d", params, oracle.ParameterCount)
	}
	return nil
}

func validateScratchConfig(config ScratchConfig) error {
	if config.VocabSize <= 1 || config.BlockSize <= 1 || config.Embedding <= 0 || config.HeadDim <= 0 ||
		config.Heads <= 0 || config.Layers <= 0 || config.MLPWidth <= 0 || config.AttentionWindow <= 0 ||
		config.AttentionWindow > config.BlockSize || config.Embedding%config.Heads != 0 ||
		config.HeadDim != config.Embedding/config.Heads || config.BOS != len(config.Characters) ||
		config.VocabSize != len(config.Characters)+1 || len(config.CharacterToIndex) != len(config.Characters) ||
		config.EstimatedParams <= 0 || !positiveFinite(config.BaseLearningRate) ||
		!positiveFinite(config.Initialization) || !positiveFinite(config.Epsilon) ||
		config.MuonMomentum <= 0 || config.MuonMomentum >= 1 || !finite(config.MuonMomentum) {
		return errors.New("adaptive scratch oracle: invalid derived config")
	}
	if !slices.IsSorted(config.Characters) {
		return errors.New("adaptive scratch oracle: characters are not sorted")
	}
	for index, character := range config.Characters {
		if character == "" || config.CharacterToIndex[character] != index {
			return errors.New("adaptive scratch oracle: character map differs")
		}
	}
	return nil
}

func validateScratchTokens(document string, tokens []int, config ScratchConfig) error {
	runes := []rune(document)
	if len(tokens) != len(runes)+2 {
		return fmt.Errorf("token framing differs: runes=%d tokens=%d", len(runes), len(tokens))
	}
	if tokens[0] != config.BOS || tokens[len(tokens)-1] != config.BOS {
		return fmt.Errorf("token framing differs: runes=%d tokens=%d first=%d last=%d bos=%d", len(runes), len(tokens), tokens[0], tokens[len(tokens)-1], config.BOS)
	}
	for index, value := range runes {
		want, ok := config.CharacterToIndex[string(value)]
		if !ok || tokens[index+1] != want {
			return errors.New("token mapping differs")
		}
	}
	return nil
}

func sameDocumentMultiset(all []string, partitions ...[]string) bool {
	counts := make(map[string]int, len(all))
	for _, document := range all {
		if strings.TrimSpace(document) == "" {
			return false
		}
		counts[document]++
	}
	for _, partition := range partitions {
		for _, document := range partition {
			counts[document]--
		}
	}
	for _, count := range counts {
		if count != 0 {
			return false
		}
	}
	return true
}

func sha256Hex(value string) bool {
	if len(value) != 64 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func positiveFinite(value float64) bool {
	return value > 0 && finite(value)
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func cloneScratchOracle(oracle ScratchOracle) ScratchOracle {
	oracle.Documents = slices.Clone(oracle.Documents)
	oracle.SourceEntrypoints = slices.Clone(oracle.SourceEntrypoints)
	oracle.Train = slices.Clone(oracle.Train)
	oracle.Validation = slices.Clone(oracle.Validation)
	oracle.Test = slices.Clone(oracle.Test)
	oracle.Config.Characters = slices.Clone(oracle.Config.Characters)
	oracle.Config.CharacterToIndex = cloneMap(oracle.Config.CharacterToIndex)
	tokens := oracle.Tokens
	oracle.Tokens = make([][]int, len(tokens))
	for index := range tokens {
		oracle.Tokens[index] = slices.Clone(tokens[index])
	}
	oracle.LossHistory = slices.Clone(oracle.LossHistory)
	oracle.Groups = slices.Clone(oracle.Groups)
	return oracle
}

func cloneMap[K comparable, V any](source map[K]V) map[K]V {
	if source == nil {
		return nil
	}
	result := make(map[K]V, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}
