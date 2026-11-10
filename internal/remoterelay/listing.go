package remoterelay

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"overgo/internal/overgodb"
	"overgo/internal/remoteprovider"
)

// modelsPath is the provider's model listing beside its chat completions.
const modelsPath = "/models"

// maxListingBytes bounds a provider's model listing: a few thousand
// models with names stay under a megabyte; the bound refuses a
// misbehaving provider rather than reading without end.
const maxListingBytes = 4 << 20

// Listed is one model a provider lists: its id, a display name when the
// listing carries one, and the context length it declares (zero when
// unstated).
type Listed struct {
	ID            string `json:"id"`
	Name          string `json:"name,omitzero"`
	ContextLength uint32 `json:"context_length,omitzero"`
}

// Listing is what a model listing goes with: the endpoint and either the key in hand, which goes there and nowhere else,
// or the variable a declaration bound to it.
type Listing struct {
	Endpoint       string
	KeyEnvironment string
	Key            string
}

// ListModels asks a provider for the models it serves under a key that goes to no other endpoint: the one in hand, or the
// value of a variable a declaration bound to this endpoint, the limit bounding the active models read. A key's absence
// refuses the listing by the variable's name.
func ListModels(ctx context.Context, store *overgodb.Store, limit int, listing Listing, client *http.Client) ([]Listed, error) {
	if listing.Key != "" {
		if err := remoteprovider.KeyTransport(listing.Endpoint); err != nil {
			return nil, err
		}
		return listModels(ctx, remoteprovider.CanonicalEndpoint(listing.Endpoint), listing.Key, client)
	}
	// A variable is read only where the operator named it, though a declaration made before that rule binds it.
	if err := remoteprovider.Admitted(ctx, store, listing.KeyEnvironment); err != nil {
		return nil, err
	}
	provider, err := remoteprovider.Bound(ctx, store, limit, listing.Endpoint, listing.KeyEnvironment)
	if err != nil {
		return nil, err
	}
	key, err := remoteprovider.Key(provider)
	if err != nil {
		return nil, err
	}
	return listModels(ctx, provider.Endpoint, key, client)
}

// listModels GETs the endpoint's models route: the OpenAI-compatible listing carries each model's id, and OpenRouter's adds
// its name and context length.
func listModels(ctx context.Context, endpoint, key string, client *http.Client) ([]Listed, error) {
	client = cmp.Or(client, http.DefaultClient)
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint+modelsPath, nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Authorization", "Bearer "+key)
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, maxListingBytes))
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("remote relay: %s answered %d to the model listing: %s", endpoint, response.StatusCode, strings.TrimSpace(string(body)))
	}
	var listing struct {
		Data []Listed `json:"data"`
	}
	if err := json.Unmarshal(body, &listing); err != nil {
		return nil, fmt.Errorf("remote relay: %s model listing: %w", endpoint, err)
	}
	models := make([]Listed, 0, len(listing.Data))
	for _, model := range listing.Data {
		if strings.TrimSpace(model.ID) != "" {
			models = append(models, model)
		}
	}
	return models, nil
}
