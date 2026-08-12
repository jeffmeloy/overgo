package artifact

import "encoding/json"

const JSONMediaType = "application/json"

// JSONContract defines one typed JSON artifact schema.
func JSONContract(kind Kind, schema string) DocumentContract {
	return DocumentContract{Kind: kind, MediaType: JSONMediaType, Schema: schema}
}

// JSONContent encodes a bounded content-addressed JSON document.
func JSONContent(contract DocumentContract, value any) (Content, error) {
	payload, err := json.Marshal(value)
	if err != nil {
		return Content{}, err
	}
	id, err := contract.Identify(payload)
	if err != nil {
		return Content{}, err
	}
	return contract.Content(id, payload)
}
