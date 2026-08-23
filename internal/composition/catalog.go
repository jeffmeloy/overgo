// Catalog assembly: committed decompositions joined with measurements.
package composition

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/repodb"
	"overgo/internal/tensorstats"
)

// LoadCatalog builds the retrieval catalog from the store: every committed
// component decomposition (lexical signal) joined with committed tensor
// measurements (distributional signal). One owner for the join -- retrieval,
// proposal, and ranking training all index the same components.
func LoadCatalog(ctx context.Context, store *repodb.Store) ([]CatalogComponent, error) {
	statistics := map[string]tensorstats.Characterization{}
	measurementResult, err := store.Query(ctx, repodb.Query{
		Kind: artifact.KindTensorInventory, MaxResults: store.QueryExtent(),
		Projection: repodb.ProjectArtifacts | repodb.ProjectContentData,
	})
	if err != nil {
		return nil, err
	}
	for _, descriptor := range measurementResult.Artifacts {
		if descriptor.MediaType != modelartifact.TensorMeasurementMediaType {
			continue
		}
		content, ok := measurementResult.Content(descriptor.ID)
		if !ok {
			continue
		}
		document, err := modelartifact.ParseTensorMeasurementDocument(content)
		if err != nil {
			continue
		}
		inventory, ok, err := modelartifact.ReadTensorInventoryDocument(ctx, store, document.Inventory)
		if err != nil || !ok {
			continue
		}
		for _, measurement := range document.Measurements {
			statistics[inventory.Model.String()+"\x00"+measurement.Name] = measurement.Characterization
		}
	}
	components := make([]CatalogComponent, 0)
	decompositionResult, err := store.Query(ctx, repodb.Query{
		Kind: artifact.KindTensorSet, MaxResults: store.QueryExtent(),
		Projection: repodb.ProjectArtifacts | repodb.ProjectContentData,
	})
	if err != nil {
		return nil, err
	}
	for _, descriptor := range decompositionResult.Artifacts {
		if descriptor.MediaType != modelartifact.ComponentDecompositionMediaType {
			continue
		}
		content, ok := decompositionResult.Content(descriptor.ID)
		if !ok {
			continue
		}
		decomposition, err := modelartifact.ParseComponentDecomposition(content)
		if err != nil {
			continue
		}
		for _, component := range decomposition.Components {
			entry := CatalogComponent{
				Model: decomposition.Model, Name: component.Name, Contract: component.Contract,
			}
			if characterization, ok := statistics[decomposition.Model.String()+"\x00"+component.Name]; ok {
				entry.Statistics = &characterization
			}
			components = append(components, entry)
		}
	}
	if len(components) == 0 {
		return nil, errors.New("composition: no committed component decompositions to index")
	}
	return components, nil
}
