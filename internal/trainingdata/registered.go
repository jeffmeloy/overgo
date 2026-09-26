package trainingdata

import (
	"context"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/recipecontract"
)

// OpenRegistered opens the dataset registered under name in the store's
// catalog for inspection: every record through its modality's processor, a
// decoded audio record bounded to maxSamples. A registration commits no split.
func OpenRegistered(ctx context.Context, reader artifact.Reader, name string, maxSamples uint64) (*Dataset, error) {
	catalog, found, err := dataset.ResolveCatalog(ctx, reader)
	if err != nil {
		return nil, err
	}
	index := slices.IndexFunc(catalog.Catalog, func(entry dataset.CatalogEntry) bool { return entry.Name == name })
	if !found || index < 0 {
		return nil, fmt.Errorf("dataset %q is not registered", name)
	}
	modality := recipecontract.Modality(catalog.Catalog[index].Modality)
	var process Processor
	switch modality {
	case recipecontract.ModalityText:
		process = Passthrough(RoleInput, modality, EncodingUTF8)
	case recipecontract.ModalityImage:
		process = ImageProcessor(RoleInput)
	case recipecontract.ModalityAudio:
		process = AudioProcessor(RoleInput, maxSamples)
	default:
		return nil, fmt.Errorf("dataset %q is %s; an inspection reads text, image and audio datasets", name, modality)
	}
	processor, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/dataset-inspection/"+string(modality)+"/v1"))
	if err != nil {
		return nil, err
	}
	signature := recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{modality}, Outputs: []recipecontract.Modality{modality}}
	return MaterializeWhole(ctx, reader, Authority{
		Dataset: catalog.Catalog[index].Dataset, Processors: []artifact.ID{processor}, Signature: signature,
	}, []ProcessorBinding{{Artifact: processor, Modalities: []recipecontract.Modality{modality}, Process: process}})
}
