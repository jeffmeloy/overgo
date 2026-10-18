package modelintake

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/hfgguf"
	"overgo/internal/hfrepo"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/strictjson"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

var importContract = artifact.JSONContract(artifact.KindFile, "overgo/model-import/v1")

// ImportInspection holds source identities and structural import facts.
// Numerical verification and serving activation require their existing evidence owners.
type ImportInspection struct {
	Inventory modelartifact.Inventory
	Content   artifact.Content
	profiles  []modelrecipe.ProfileDocument
}

type importDescription struct {
	Version           uint16               `json:"version"`
	Model             artifact.ID          `json:"model"`
	TensorInventory   artifact.ID          `json:"tensor_inventory"`
	Sources           []artifact.Component `json:"sources"`
	Components        []importComponent    `json:"components"`
	StoredTensorBytes uint64               `json:"stored_tensor_bytes"`
	Qualification     string               `json:"qualification"`
}

type importComponent struct {
	Declaration       modelartifact.PipelineComponent `json:"declaration"`
	Purpose           hfgguf.LanguageOutput           `json:"purpose,omitempty"`
	Profile           *artifact.ID                    `json:"profile,omitempty"`
	Spec              *model.Spec                     `json:"spec,omitempty"`
	Bindings          []hfgguf.TensorBinding          `json:"bindings,omitempty"`
	Operations        []string                        `json:"operations,omitempty"`
	Policies          []importPolicy                  `json:"declared_policies,omitempty"`
	MappedTensorBytes uint64                          `json:"mapped_tensor_bytes"`
	Unsupported       string                          `json:"unsupported,omitempty"`
}

// InspectImport describes a local HF model, pipeline or GGUF using registered mappings.
// Hashing verifies source identity while tensor payloads remain unmaterialized.
// Empty output derives the contract from the declared model class.
func InspectImport(ctx context.Context, path string, output hfgguf.LanguageOutput) (ImportInspection, error) {
	if ctx == nil {
		return ImportInspection{}, errors.New("model intake: inspection requires a context")
	}
	if err := ctx.Err(); err != nil {
		return ImportInspection{}, err
	}
	if output != "" && output != hfgguf.LanguageHidden && output != hfgguf.LanguageTokens {
		return ImportInspection{}, fmt.Errorf("model intake: unsupported language output %q", output)
	}
	info, err := os.Stat(path)
	if err != nil {
		return ImportInspection{}, err
	}
	if !info.IsDir() {
		inventory, component, profile, err := inspectGGUFImport(ctx, path, output)
		if err != nil {
			return ImportInspection{}, err
		}
		return finishImportInspection(ctx, inventory, []importComponent{component}, []modelrecipe.ProfileDocument{profile})
	}
	inventory, err := modelartifact.FromHFPathContext(ctx, path)
	if err != nil {
		return ImportInspection{}, err
	}
	declarations := []modelartifact.PipelineComponent{{Name: "."}}
	if _, err := os.Stat(filepath.Join(path, "model_index.json")); err == nil {
		declarations, err = modelartifact.ReadPipelineComponents(path)
		if err != nil {
			return ImportInspection{}, err
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return ImportInspection{}, err
	}
	var components []importComponent
	var profiles []modelrecipe.ProfileDocument
	for _, declaration := range declarations {
		if err := ctx.Err(); err != nil {
			return ImportInspection{}, err
		}
		component, profile, err := inspectImportComponent(filepath.Join(path, declaration.Name), declaration, output)
		if err != nil {
			return ImportInspection{}, fmt.Errorf("model intake: component %q: %w", declaration.Name, err)
		}
		component.Policies, err = componentImportPolicies(inventory, declaration.Name)
		if err != nil {
			return ImportInspection{}, err
		}
		components = append(components, component)
		profiles = append(profiles, profile)
	}
	return finishImportInspection(ctx, inventory, components, profiles)
}

func finishImportInspection(ctx context.Context, inventory modelartifact.Inventory, components []importComponent, profiles []modelrecipe.ProfileDocument) (ImportInspection, error) {
	if err := ctx.Err(); err != nil {
		return ImportInspection{}, err
	}
	description := importDescription{Version: artifact.InitialDocumentVersion, Model: inventory.Manifest.ID, TensorInventory: inventory.TensorInventory.ID, Sources: inventory.Manifest.Components, Components: components, Qualification: "structural-only"}
	for _, fact := range inventory.TensorInventory.Tensors {
		var ok bool
		description.StoredTensorBytes, ok = checked.Add64(description.StoredTensorBytes, fact.Bytes)
		if !ok {
			return ImportInspection{}, errors.New("model intake: stored tensor byte count overflows")
		}
	}
	inspection := ImportInspection{Inventory: inventory}
	seen := map[artifact.ID]bool{}
	for _, profile := range profiles {
		if profile.ID != (artifact.ID{}) && !seen[profile.ID] {
			inspection.profiles = append(inspection.profiles, profile)
			seen[profile.ID] = true
		}
	}
	var err error
	inspection.Content, err = artifact.JSONContent(importContract, description)
	return inspection, err
}

func inspectImportComponent(path string, declaration modelartifact.PipelineComponent, output hfgguf.LanguageOutput) (importComponent, modelrecipe.ProfileDocument, error) {
	component := importComponent{Declaration: declaration}
	unsupported := func(reason string) (importComponent, modelrecipe.ProfileDocument, error) {
		component.Unsupported = reason
		return component, modelrecipe.ProfileDocument{}, nil
	}
	if _, err := os.Stat(filepath.Join(path, "config.json")); errors.Is(err, os.ErrNotExist) {
		return unsupported("component policy is inventoried but has no registered executable import mapping")
	} else if err != nil {
		return component, modelrecipe.ProfileDocument{}, err
	}
	identity, err := hfrepo.InspectIdentity(path)
	if err != nil {
		return component, modelrecipe.ProfileDocument{}, err
	}
	if declaration.Library != "" && declaration.Library != "transformers" {
		return unsupported("component library has no registered import mapping")
	}
	switch identity.ModelType {
	case "llama", "qwen2", "qwen3_5":
	default:
		return unsupported(fmt.Sprintf("model type %q has no registered import mapping", identity.ModelType))
	}
	if declaration.Class != "" && !slices.Contains(identity.Architectures, declaration.Class) {
		return component, modelrecipe.ProfileDocument{}, errors.New("pipeline class disagrees with model architectures")
	}
	if output == "" {
		output = hfgguf.LanguageTokens
		if slices.Contains(identity.Architectures, "Qwen3_5Model") {
			output = hfgguf.LanguageHidden
		}
	}
	repository, err := hfrepo.Open(path)
	if err != nil {
		return component, modelrecipe.ProfileDocument{}, err
	}
	catalog, inspectErr := hfgguf.InspectLanguageModel(repository, output)
	if err := errors.Join(inspectErr, repository.Close()); err != nil {
		return component, modelrecipe.ProfileDocument{}, err
	}
	return describeImportLanguage(component, catalog.Spec, catalog.Weights, catalog.Bindings(), output)
}

func describeImportLanguage(component importComponent, spec model.Spec, weights model.Weights, bindings []hfgguf.TensorBinding, output hfgguf.LanguageOutput) (importComponent, modelrecipe.ProfileDocument, error) {
	plan, err := model.CompileModelPlanWithProfile(spec, weights, spec.Profile())
	if err != nil {
		return component, modelrecipe.ProfileDocument{}, err
	}
	if plan.Forward().Operation != model.ForwardOperationCached || plan.Forward().AlternateStates() {
		component.Unsupported = "compiled forward contract has no uncached language inspection mapping"
		return component, modelrecipe.ProfileDocument{}, nil
	}
	component.Operations, err = importLanguageOperations(spec, weights, output, plan)
	if err != nil {
		return component, modelrecipe.ProfileDocument{}, err
	}
	profile, err := modelrecipe.NewProfileDocument(spec.Profile())
	if err != nil {
		return component, modelrecipe.ProfileDocument{}, err
	}
	spec.Name = ""
	component.Purpose, component.Profile, component.Spec = output, &profile.ID, &spec
	component.Bindings = bindings
	for _, binding := range component.Bindings {
		var ok bool
		component.MappedTensorBytes, ok = checked.Add64(component.MappedTensorBytes, binding.TargetBytes)
		if !ok {
			return component, modelrecipe.ProfileDocument{}, errors.New("mapped tensor byte count overflows")
		}
	}
	return component, profile, nil
}

func importLanguageOperations(spec model.Spec, weights model.Weights, output hfgguf.LanguageOutput, plan model.ModelPlan) ([]string, error) {
	builder := tensor.NewBuilder()
	bind := func(info *gguf.TensorInfo) (*tensor.Tensor, error) {
		if info == nil || info.Name == "" {
			return nil, nil
		}
		shape, err := tensor.NewShape(info.Shape[:info.Dimensions]...)
		if err != nil {
			return nil, err
		}
		return builder.Input(info.Name, dtype.F32, shape), builder.Err()
	}
	embedding, err := bind(&weights.TokenEmbedding)
	if err != nil {
		return nil, err
	}
	current := builder.GetRows(embedding, []uint32{0})
	if scale := spec.InputEmbeddingScale(); scale != 1 {
		current = builder.Scale(current, scale)
	}
	for index := range plan.LayerCount() {
		layerWeights := model.LayerGraphWeights{}
		source := weights.Layers[index]
		if err := model.BindLayerGraphInventory(&source, &layerWeights, func(_ model.LayerBindingSlot, info *gguf.TensorInfo) (*tensor.Tensor, error) { return bind(info) }); err != nil {
			return nil, err
		}
		context := model.CachedBlockContext{Builder: builder, Input: current, Positions: []uint32{0}, Layer: uint32(index), Recurrent: source.Recurrent, CacheWrite: tensor.CacheWriteConcat, Sequences: 1}
		if source.Recurrent {
			schema, err := plan.CacheSchema(uint32(index), 1)
			if err != nil {
				return nil, err
			}
			context.PastKey = builder.Input("convolution-state", dtype.F32, schema.Primary.Key.Value.Shape)
			context.PastValue = builder.Input("recurrent-state", dtype.F32, schema.Primary.Value.Value.Shape)
		}
		program, err := plan.LayerProgram(uint32(index))
		if err != nil {
			return nil, err
		}
		result, err := program.Build(context, layerWeights)
		if err != nil {
			return nil, err
		}
		current = result.Output
	}
	if plan.Terminal().Normalization != model.OutputNormAbsent {
		norm, err := bind(&weights.OutputNorm)
		if err != nil {
			return nil, err
		}
		bias, err := bind(weights.OutputNormBias)
		if err != nil {
			return nil, err
		}
		current = plan.Normalization().Apply(builder, current, norm, bias)
	}
	if output == hfgguf.LanguageTokens {
		head := embedding
		if plan.Terminal().OutputHead == model.OutputHeadDedicated {
			head, err = bind(weights.Output)
			if err != nil {
				return nil, err
			}
		}
		current = builder.MulMat(head, current)
		bias, err := bind(weights.OutputBias)
		if err != nil {
			return nil, err
		}
		if bias != nil {
			current = builder.Add(current, bias)
		}
		if scale := spec.OutputLogitMultiplier(); scale != 1 {
			current = builder.Scale(current, scale)
		}
		if spec.FinalLogitSoftcap != 0 {
			current = builder.Scale(builder.Tanh(builder.Scale(current, 1/spec.FinalLogitSoftcap)), spec.FinalLogitSoftcap)
		}
	}
	if err := builder.Err(); err != nil {
		return nil, err
	}
	nodes, err := tensor.Topological(current)
	if err != nil {
		return nil, err
	}
	operations := map[string]bool{}
	for _, node := range nodes {
		if _, known := tensor.DescribeOperation(node.Op); !known {
			return nil, fmt.Errorf("model intake: undeclared operation %v", node.Op)
		}
		operations[node.Op.String()] = true
	}
	names := make([]string, 0, len(operations))
	for name := range operations {
		names = append(names, name)
	}
	slices.Sort(names)
	return names, nil
}

// RegisterImport publishes inspected source facts without a recipe or activation.
func RegisterImport(ctx context.Context, store artifact.Repository, inspection ImportInspection) error {
	if err := importContract.ValidateContent(inspection.Content, inspection.Content.Descriptor.ID); err != nil {
		return err
	}
	var description importDescription
	if err := strictjson.DecodeBytes(inspection.Content.Data, &description); err != nil {
		return err
	}
	if description.Version != artifact.InitialDocumentVersion || description.Qualification != "structural-only" ||
		description.Model != inspection.Inventory.Manifest.ID || description.TensorInventory != inspection.Inventory.TensorInventory.ID ||
		!slices.Equal(description.Sources, inspection.Inventory.Manifest.Components) {
		return errors.New("model intake: inspection does not bind the supplied source inventory")
	}
	if err := inspection.Inventory.VerifySources(ctx); err != nil {
		return err
	}
	batch, err := inspection.Inventory.Batch("model/import/" + inspection.Content.Descriptor.ID.String())
	if err != nil {
		return err
	}
	batch.Contents = append(batch.Contents, inspection.Content)
	parents := []artifact.ID{inspection.Inventory.Manifest.ID, inspection.Inventory.TensorInventory.ID}
	for _, profile := range inspection.profiles {
		contents, lineage, err := modelrecipe.ProfilePublicationFacts(profile)
		if err != nil {
			return err
		}
		batch.Contents = append(batch.Contents, contents...)
		batch.Lineage = append(batch.Lineage, lineage...)
		parents = append(parents, profile.ID)
	}
	batch.Lineage = append(batch.Lineage, artifact.UniqueDependencyLineage(inspection.Content.Descriptor.ID, parents...)...)
	if err := modelartifact.PreserveRawTextDescriptors(ctx, store, &batch); err != nil {
		return err
	}
	return modelrecipe.PublishFacts(ctx, store, batch)
}
