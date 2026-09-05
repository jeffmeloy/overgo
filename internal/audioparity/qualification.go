package audioparity

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/binaryschema"
	"overgo/internal/hfbpe"
	"overgo/internal/modelartifact"
	"overgo/internal/pathidentity"
	"overgo/internal/strictjson"
	"overgo/internal/tensor"
)

const (
	// QualificationMediaType identifies an audio artifact loadability decision.
	QualificationMediaType = "application/vnd.overgo.audio-artifact-qualification+json"
	// QualificationSchema identifies the audio artifact qualification contract.
	QualificationSchema  = "overgo/audio-artifact-qualification/v1"
	audioRecipeMediaType = "application/vnd.overgo.audio-architecture-recipe+json"
	audioRecipeSchema    = "overgo/audio-architecture-recipe/v1"
)

var (
	//go:embed recipes/granite5asr.json
	graniteRecipeJSON  []byte
	qualificationCodec = artifact.JSONDocumentCodec(
		"audio artifact qualification", artifact.KindEvidence, QualificationMediaType, QualificationSchema,
		canonicalizeQualification,
		func(value AudioArtifactQualification) artifact.ID { return value.ID },
		func(value *AudioArtifactQualification, id artifact.ID) { value.ID = id },
		cloneQualification,
	)
	audioRecipeContract = artifact.DocumentContract{
		Kind: artifact.KindProfile, MediaType: audioRecipeMediaType, Schema: audioRecipeSchema,
	}
)

var qualificationChecks = []string{
	"architecture-recipe", "container", "oracle", "preprocessor", "tensor-schema", "tokenizer",
}

type recipeComponent struct {
	Name string                 `json:"name"`
	Role artifact.ComponentRole `json:"role"`
	Path string                 `json:"path"`
}

type graniteConfig struct {
	VocabSize             uint64   `json:"vocab_size"`
	PadTokenID            uint64   `json:"pad_token_id"`
	HiddenSize            uint64   `json:"hidden_size"`
	IntermediateSize      uint64   `json:"intermediate_size"`
	NumHiddenLayers       uint64   `json:"num_hidden_layers"`
	NumAttentionHeads     uint64   `json:"num_attention_heads"`
	NumKeyValueHeads      uint64   `json:"num_key_value_heads"`
	HeadDim               uint64   `json:"head_dim"`
	ContextSize           uint64   `json:"context_size"`
	ConvKernelSize        uint64   `json:"conv_kernel_size"`
	ConvExpansionFactor   uint64   `json:"conv_expansion_factor"`
	MaxPositionEmbeddings uint64   `json:"max_position_embeddings"`
	NumMelBins            uint64   `json:"num_mel_bins"`
	SubsampleLayers       []uint64 `json:"subsample_layers"`
}

type granitePreprocessor struct {
	SampleRate     uint64  `json:"sample_rate"`
	NFFT           uint64  `json:"n_fft"`
	WinLength      uint64  `json:"win_length"`
	HopLength      uint64  `json:"hop_length"`
	NMels          uint64  `json:"n_mels"`
	StackFactor    uint64  `json:"stack_factor"`
	Deltas         bool    `json:"deltas"`
	DeltaWinLength uint64  `json:"delta_win_length"`
	LogmelFloorDB  float64 `json:"logmel_floor_db"`
}

type audioArchitectureRecipe struct {
	ID                      artifact.ID                `json:"-"`
	Version                 uint16                     `json:"version"`
	Family                  string                     `json:"family"`
	Capability              string                     `json:"capability"`
	Format                  modelartifact.TensorFormat `json:"format"`
	ModelType               string                     `json:"model_type"`
	Architecture            string                     `json:"architecture"`
	TokenizerClass          string                     `json:"tokenizer_class"`
	ProcessorClass          string                     `json:"processor_class"`
	FeatureExtractorClass   string                     `json:"feature_extractor_class"`
	ReferenceImplementation string                     `json:"reference_implementation"`
	Components              []recipeComponent          `json:"components"`
	Config                  graniteConfig              `json:"config"`
	Preprocessor            granitePreprocessor        `json:"preprocessor"`
}

// AudioArtifactQualification is the content-addressed decision that an exact
// model artifact either satisfies or fails its executable audio recipe.
type AudioArtifactQualification struct {
	ID              artifact.ID                `json:"-"`
	Version         uint16                     `json:"version"`
	Model           artifact.ID                `json:"model"`
	Election        artifact.ID                `json:"election"`
	Recipe          artifact.ID                `json:"recipe"`
	TensorInventory artifact.ID                `json:"tensor_inventory"`
	Family          string                     `json:"family"`
	Format          modelartifact.TensorFormat `json:"format"`
	Loadable        bool                       `json:"loadable"`
	Passed          []string                   `json:"passed"`
	Refusals        []string                   `json:"refusals,omitempty"`
}

// QualifyAudioArtifact validates an already parsed model inventory and its
// installed assets. Container format comes from the parser, never a filename.
func QualifyAudioArtifact(
	ctx context.Context,
	election Election,
	inventory modelartifact.Inventory,
	modelRoot string,
) (AudioArtifactQualification, error) {
	if ctx == nil {
		return AudioArtifactQualification{}, errors.New("audio parity: nil qualification context")
	}
	if err := ctx.Err(); err != nil {
		return AudioArtifactQualification{}, err
	}
	recipe, _, err := graniteRecipe()
	if err != nil {
		return AudioArtifactQualification{}, err
	}
	root, err := pathidentity.Canonical(modelRoot)
	if err != nil {
		return AudioArtifactQualification{}, err
	}
	passed, refusals := qualifyAudioArtifact(election, inventory, root, recipe)
	qualification := AudioArtifactQualification{
		Version: artifact.InitialDocumentVersion, Model: inventory.Manifest.ID,
		Election: election.ID, Recipe: recipe.ID, TensorInventory: inventory.TensorInventory.ID,
		Family: election.Family, Format: inventory.TensorInventory.Format,
		Loadable: len(refusals) == 0, Passed: passed, Refusals: refusals,
	}
	return qualificationCodec.NewInitial(qualification)
}

// AugmentBatch binds a successful qualification, its recipe, and complete
// tensor inventory into the election's atomic artifact publication.
func (q AudioArtifactQualification) AugmentBatch(
	batch *artifact.Batch,
	inventory modelartifact.Inventory,
	previous *artifact.ID,
) error {
	if batch == nil || !q.Loadable || inventory.Manifest.ID != q.Model ||
		inventory.TensorInventory.ID != q.TensorInventory {
		return errors.New("audio parity: invalid loadable qualification publication")
	}
	if err := qualificationCodec.ValidateIdentity(q); err != nil {
		return err
	}
	_, recipeContent, err := graniteRecipe()
	if err != nil {
		return err
	}
	tensorContent, err := inventory.TensorInventory.Content()
	if err != nil {
		return err
	}
	qualificationContent, err := qualificationCodec.Content(q)
	if err != nil {
		return err
	}
	batch.Contents = append(batch.Contents, recipeContent, tensorContent, qualificationContent)
	batch.Lineage = append(batch.Lineage, inventory.TensorInventory.Lineage()...)
	batch.Lineage = append(batch.Lineage, artifact.DependencyLineage(
		q.ID, q.Model, q.Election, q.Recipe, q.TensorInventory,
	)...)
	batch.Aliases = append(batch.Aliases, artifact.AliasBinding{
		Name: q.Alias(), Target: q.ID, Previous: artifact.CloneID(previous),
	})
	return batch.Validate()
}

// Alias returns the stable loadability binding for this exact model.
func (q AudioArtifactQualification) Alias() string {
	return "audio/qualification/loadable/" + q.Model.DigestHex()
}

func qualifyAudioArtifact(
	election Election,
	inventory modelartifact.Inventory,
	root string,
	recipe audioArchitectureRecipe,
) ([]string, []string) {
	passed := make([]string, 0, len(qualificationChecks))
	refusals := make([]string, 0)
	pass := func(check string, err error) {
		if err == nil {
			passed = append(passed, check)
			return
		}
		refusals = append(refusals, check+": "+err.Error())
	}
	pass("container", validateQualificationContainer(election, inventory, recipe))
	pass("tensor-schema", validateGraniteTensorSchema(inventory.TensorInventory, recipe))
	pass("architecture-recipe", validateGraniteConfiguration(root, recipe))
	pass("tokenizer", validateGraniteTokenizer(root, recipe))
	pass("preprocessor", validateGranitePreprocessor(root, recipe))
	pass("oracle", validateQualificationOracle(election, recipe))
	return passed, refusals
}

func validateQualificationContainer(
	election Election,
	inventory modelartifact.Inventory,
	recipe audioArchitectureRecipe,
) error {
	if err := inventory.Manifest.Validate(); err != nil {
		return err
	}
	if err := inventory.TensorInventory.ValidateIdentity(); err != nil {
		return err
	}
	if inventory.Manifest.ID != election.Model.ID || inventory.TensorInventory.Owner != election.Model.ID {
		return errors.New("manifest or tensor inventory owner differs from the election")
	}
	if inventory.TensorInventory.Format != recipe.Format {
		return fmt.Errorf("parsed format %q is incompatible with recipe format %q", inventory.TensorInventory.Format, recipe.Format)
	}
	for _, requirement := range recipe.Components {
		found := false
		for _, component := range inventory.Manifest.Components {
			if component.Name == requirement.Name && component.Role == requirement.Role {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("required %s component %q is absent", requirement.Role, requirement.Name)
		}
		bound := false
		for _, file := range election.Files {
			if file.Path == requirement.Path {
				bound = true
				break
			}
		}
		if !bound {
			return fmt.Errorf("required component path %q is not election-bound", requirement.Path)
		}
	}
	return nil
}

func validateGraniteTensorSchema(inventory modelartifact.TensorInventoryDocument, recipe audioArchitectureRecipe) error {
	expected := graniteTensorRequirements(recipe)
	if len(inventory.Tensors) != len(expected) {
		return fmt.Errorf("tensor count=%d, want %d", len(inventory.Tensors), len(expected))
	}
	for index, want := range expected {
		got := inventory.Tensors[index]
		if got.Name != want.Name || !slices.Equal(got.Shape, want.Shape) {
			return fmt.Errorf("tensor[%d]=%s%v, want %s%v", index, got.Name, got.Shape, want.Name, want.Shape)
		}
	}
	return nil
}

func graniteTensorRequirements(recipe audioArchitectureRecipe) []modelartifact.TensorFact {
	config := recipe.Config
	inputFeatures := config.NumMelBins * recipe.Preprocessor.StackFactor
	if recipe.Preprocessor.Deltas {
		inputFeatures += inputFeatures
	}
	convInner := config.HiddenSize * config.ConvExpansionFactor
	twiceConvInner := convInner + convInner
	relativePositionRows := config.MaxPositionEmbeddings + config.MaxPositionEmbeddings
	relativePositionRows++
	requirements := []modelartifact.TensorFact{
		{Name: "encoder.input_linear.bias", Shape: []uint64{config.HiddenSize}},
		{Name: "encoder.input_linear.weight", Shape: []uint64{config.HiddenSize, inputFeatures}},
	}
	for layer := range config.NumHiddenLayers {
		prefix := "encoder.layers." + strconv.FormatUint(layer, binaryschema.DecimalRadix) + "."
		add := func(name string, shape ...uint64) {
			requirements = append(requirements, modelartifact.TensorFact{Name: prefix + name, Shape: shape})
		}
		for _, block := range []string{"feed_forward1", "feed_forward2"} {
			add(block+".linear1.bias", config.IntermediateSize)
			add(block+".linear1.weight", config.IntermediateSize, config.HiddenSize)
			add(block+".linear2.bias", config.HiddenSize)
			add(block+".linear2.weight", config.HiddenSize, config.IntermediateSize)
		}
		for _, name := range []string{"norm_conv", "norm_feed_forward1", "norm_feed_forward2", "norm_out", "norm_self_att"} {
			add(name+".bias", config.HiddenSize)
			add(name+".weight", config.HiddenSize)
		}
		add("self_attn.k_proj.weight", config.HiddenSize, config.HiddenSize)
		add("self_attn.o_proj.bias", config.HiddenSize)
		add("self_attn.o_proj.weight", config.HiddenSize, config.HiddenSize)
		add("self_attn.q_proj.weight", config.HiddenSize, config.HiddenSize)
		add("self_attn.rel_pos_emb.weight", relativePositionRows, config.HeadDim)
		add("self_attn.v_proj.weight", config.HiddenSize, config.HiddenSize)
		add("conv.depthwise_conv.weight", convInner, tensor.SingletonExtent, config.ConvKernelSize)
		add("conv.norm.bias", convInner)
		add("conv.norm.num_batches_tracked")
		add("conv.norm.running_mean", convInner)
		add("conv.norm.running_var", convInner)
		add("conv.norm.weight", convInner)
		add("conv.pointwise_lin1.bias", twiceConvInner)
		add("conv.pointwise_lin1.weight", twiceConvInner, config.HiddenSize)
		add("conv.pointwise_lin2.bias", config.HiddenSize)
		add("conv.pointwise_lin2.weight", config.HiddenSize, convInner)
	}
	requirements = append(requirements,
		modelartifact.TensorFact{Name: "encoder.out.bias", Shape: []uint64{config.VocabSize}},
		modelartifact.TensorFact{Name: "encoder.out.weight", Shape: []uint64{config.VocabSize, config.HiddenSize}},
		modelartifact.TensorFact{Name: "encoder.out_mid.bias", Shape: []uint64{config.HiddenSize}},
		modelartifact.TensorFact{Name: "encoder.out_mid.weight", Shape: []uint64{config.HiddenSize, config.VocabSize}},
	)
	slices.SortFunc(requirements, func(left, right modelartifact.TensorFact) int {
		return strings.Compare(left.Name, right.Name)
	})
	return requirements
}

func validateGraniteConfiguration(root string, recipe audioArchitectureRecipe) error {
	var value struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
		VocabSize     uint64   `json:"vocab_size"`
		PadTokenID    uint64   `json:"pad_token_id"`
		EncoderConfig struct {
			HiddenSize            uint64   `json:"hidden_size"`
			IntermediateSize      uint64   `json:"intermediate_size"`
			NumHiddenLayers       uint64   `json:"num_hidden_layers"`
			NumAttentionHeads     uint64   `json:"num_attention_heads"`
			NumKeyValueHeads      uint64   `json:"num_key_value_heads"`
			HeadDim               uint64   `json:"head_dim"`
			ContextSize           uint64   `json:"context_size"`
			ConvKernelSize        uint64   `json:"conv_kernel_size"`
			ConvExpansionFactor   uint64   `json:"conv_expansion_factor"`
			MaxPositionEmbeddings uint64   `json:"max_position_embeddings"`
			NumMelBins            uint64   `json:"num_mel_bins"`
			SubsampleLayers       []uint64 `json:"subsample_layers"`
		} `json:"encoder_config"`
	}
	if err := readQualificationJSON(root, "config.json", &value); err != nil {
		return err
	}
	got := graniteConfig{
		VocabSize: value.VocabSize, PadTokenID: value.PadTokenID,
		HiddenSize: value.EncoderConfig.HiddenSize, IntermediateSize: value.EncoderConfig.IntermediateSize,
		NumHiddenLayers: value.EncoderConfig.NumHiddenLayers, NumAttentionHeads: value.EncoderConfig.NumAttentionHeads,
		NumKeyValueHeads: value.EncoderConfig.NumKeyValueHeads, HeadDim: value.EncoderConfig.HeadDim,
		ContextSize: value.EncoderConfig.ContextSize, ConvKernelSize: value.EncoderConfig.ConvKernelSize,
		ConvExpansionFactor:   value.EncoderConfig.ConvExpansionFactor,
		MaxPositionEmbeddings: value.EncoderConfig.MaxPositionEmbeddings,
		NumMelBins:            value.EncoderConfig.NumMelBins, SubsampleLayers: value.EncoderConfig.SubsampleLayers,
	}
	if value.ModelType != recipe.ModelType || !slices.Equal(value.Architectures, []string{recipe.Architecture}) ||
		!reflect.DeepEqual(got, recipe.Config) {
		return errors.New("config.json differs from the pinned architecture recipe")
	}
	return nil
}

func validateGraniteTokenizer(root string, recipe audioArchitectureRecipe) error {
	var config struct {
		TokenizerClass string `json:"tokenizer_class"`
		ProcessorClass string `json:"processor_class"`
	}
	if err := readQualificationJSON(root, "tokenizer_config.json", &config); err != nil {
		return err
	}
	if config.TokenizerClass != recipe.TokenizerClass || config.ProcessorClass != recipe.ProcessorClass {
		return errors.New("tokenizer configuration differs from the recipe")
	}
	if _, err := hfbpe.Load(root); err != nil {
		return errors.New("tokenizer.json is not executable by the Hugging Face BPE owner")
	}
	return nil
}

func validateGranitePreprocessor(root string, recipe audioArchitectureRecipe) error {
	var preprocessor granitePreprocessor
	if err := readQualificationJSON(root, "preprocessor_config.json", &preprocessor); err != nil {
		return err
	}
	if !reflect.DeepEqual(preprocessor, recipe.Preprocessor) {
		return errors.New("preprocessor configuration differs from the recipe")
	}
	var processor struct {
		ProcessorClass   string `json:"processor_class"`
		FeatureExtractor struct {
			FeatureExtractorType string `json:"feature_extractor_type"`
			SamplingRate         uint64 `json:"sampling_rate"`
			NumMelBins           uint64 `json:"num_mel_bins"`
			NFFT                 uint64 `json:"n_fft"`
			WinLength            uint64 `json:"win_length"`
			HopLength            uint64 `json:"hop_length"`
		} `json:"feature_extractor"`
	}
	if err := readQualificationJSON(root, "processor_config.json", &processor); err != nil {
		return err
	}
	if processor.ProcessorClass != recipe.ProcessorClass ||
		processor.FeatureExtractor.FeatureExtractorType != recipe.FeatureExtractorClass ||
		processor.FeatureExtractor.SamplingRate != recipe.Preprocessor.SampleRate ||
		processor.FeatureExtractor.NumMelBins != recipe.Preprocessor.NMels ||
		processor.FeatureExtractor.NFFT != recipe.Preprocessor.NFFT ||
		processor.FeatureExtractor.WinLength != recipe.Preprocessor.WinLength ||
		processor.FeatureExtractor.HopLength != recipe.Preprocessor.HopLength {
		return errors.New("processor configuration differs from the recipe")
	}
	return nil
}

func validateQualificationOracle(election Election, recipe audioArchitectureRecipe) error {
	if election.Family != recipe.Family || election.Capability != recipe.Capability ||
		election.Runtime.Implementation != recipe.ReferenceImplementation || election.Runtime.Device != "cpu" {
		return errors.New("election does not bind the recipe's CPU reference implementation")
	}
	summary := election.Summary()
	if summary.Fixtures == 0 || summary.EquivalentMatches != summary.Fixtures {
		return errors.New("election does not provide complete pinned oracle parity")
	}
	return nil
}

func readQualificationJSON(root, name string, value any) error {
	path, err := pathidentity.Canonical(filepath.Join(root, name))
	if err != nil {
		return fmt.Errorf("audio parity: qualification asset %q is unavailable", name)
	}
	contained, err := pathidentity.Contains(root, path)
	if err != nil || !contained {
		return fmt.Errorf("audio parity: qualification asset %q escapes model root", name)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("audio parity: qualification asset %q is unavailable", name)
	}
	if err := json.Unmarshal(data, value); err != nil {
		return fmt.Errorf("audio parity: parse %s: %w", name, err)
	}
	return nil
}

func graniteRecipe() (audioArchitectureRecipe, artifact.Content, error) {
	var recipe audioArchitectureRecipe
	if err := strictjson.DecodeBytes(graniteRecipeJSON, &recipe); err != nil {
		return audioArchitectureRecipe{}, artifact.Content{}, err
	}
	if err := validateAudioRecipe(recipe); err != nil {
		return audioArchitectureRecipe{}, artifact.Content{}, err
	}
	encoded, err := json.Marshal(recipe)
	if err != nil {
		return audioArchitectureRecipe{}, artifact.Content{}, err
	}
	content, err := audioRecipeContract.ContentBytes(encoded)
	if err != nil {
		return audioArchitectureRecipe{}, artifact.Content{}, err
	}
	recipe.ID = content.Descriptor.ID
	return recipe, content, nil
}

func validateAudioRecipe(recipe audioArchitectureRecipe) error {
	if recipe.Version != artifact.InitialDocumentVersion || recipe.Family == "" || recipe.Capability != "offline-asr" ||
		recipe.Format != modelartifact.TensorFormatSafetensors || recipe.ModelType == "" || recipe.Architecture == "" ||
		recipe.TokenizerClass == "" || recipe.ProcessorClass == "" || recipe.FeatureExtractorClass == "" ||
		recipe.ReferenceImplementation == "" ||
		len(recipe.Components) == 0 || recipe.Config.NumHiddenLayers == 0 || recipe.Config.HiddenSize == 0 ||
		recipe.Preprocessor.SampleRate == 0 || !recipe.Preprocessor.Deltas {
		return errors.New("audio parity: invalid audio architecture recipe")
	}
	seenNames, seenPaths := map[string]bool{}, map[string]bool{}
	for _, component := range recipe.Components {
		if component.Name == "" || component.Role == artifact.ComponentInvalid || component.Path == "" ||
			seenNames[component.Name] || seenPaths[component.Path] {
			return errors.New("audio parity: invalid recipe component")
		}
		seenNames[component.Name], seenPaths[component.Path] = true, true
	}
	return nil
}

func canonicalizeQualification(value *AudioArtifactQualification) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Model.Kind() != artifact.KindModel ||
		value.Election.Kind() != artifact.KindEvidence || value.Recipe.Kind() != artifact.KindProfile ||
		value.TensorInventory.Kind() != artifact.KindTensorInventory || value.Family == "" ||
		(value.Format != modelartifact.TensorFormatSafetensors && value.Format != modelartifact.TensorFormatGGUF) {
		return errors.New("audio parity: invalid qualification envelope")
	}
	slices.Sort(value.Passed)
	slices.Sort(value.Refusals)
	if !uniqueStrings(value.Passed) || !uniqueStrings(value.Refusals) {
		return errors.New("audio parity: qualification results are not unique")
	}
	loadable := len(value.Refusals) == 0 && slices.Equal(value.Passed, qualificationChecks)
	if value.Loadable != loadable {
		return errors.New("audio parity: qualification loadability differs from derived checks")
	}
	return nil
}

func uniqueStrings(values []string) bool {
	for index, value := range values {
		if strings.TrimSpace(value) != value || value == "" || index > 0 && value == values[index-1] {
			return false
		}
	}
	return true
}

func cloneQualification(value AudioArtifactQualification) AudioArtifactQualification {
	value.Passed = slices.Clone(value.Passed)
	value.Refusals = slices.Clone(value.Refusals)
	return value
}
