// Catalog assembly: committed decompositions joined with measurements.
package composition

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/tensorstats"
)

// LoadCatalog builds the retrieval catalog from the store: every committed
// component decomposition (lexical signal) joined with committed tensor
// measurements (distributional signal). One owner for the join -- retrieval,
// proposal, and ranking training all index the same components.
func LoadCatalog(ctx context.Context, store *overgodb.Store) ([]CatalogComponent, error) {
	statistics := map[string]tensorstats.Characterization{}
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindTensorInventory, MediaType: modelartifact.TensorMeasurementMediaType,
			Schema: modelartifact.TensorMeasurementSchema,
		}}, Order: overgodb.DocumentOldestFirst,
	}, modelartifact.ParseTensorMeasurementDocument, func(_ overgodb.DocumentView, document modelartifact.TensorMeasurementDocument) error {
		inventory, ok, err := modelartifact.ReadTensorInventoryDocument(ctx, store, document.Inventory)
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
		for _, measurement := range document.Measurements {
			statistics[inventory.Owner.String()+"\x00"+measurement.Name] = measurement.Characterization
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	components := make([]CatalogComponent, 0)
	_, err = overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindTensorSet, MediaType: modelartifact.ComponentDecompositionMediaType,
			Schema: modelartifact.ComponentDecompositionSchema,
		}}, Order: overgodb.DocumentOldestFirst,
	}, modelartifact.ParseComponentDecomposition, func(_ overgodb.DocumentView, decomposition modelartifact.ComponentDecompositionDocument) error {
		for _, component := range decomposition.Components {
			entry := CatalogComponent{
				Model: decomposition.Model, Name: component.Name, Contract: component.Contract,
			}
			if characterization, ok := statistics[decomposition.Model.String()+"\x00"+component.Name]; ok {
				entry.Statistics = &characterization
			}
			components = append(components, entry)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(components) == 0 {
		return nil, errors.New("composition: no committed component decompositions to index")
	}
	return components, nil
}
