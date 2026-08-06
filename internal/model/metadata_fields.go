package model

import "llamacpp2go/internal/gguf"

type metadataField[T any] struct {
	key         string
	destination *T
}

func metadataDestination[T any](key string, destination *T) metadataField[T] {
	return metadataField[T]{key: key, destination: destination}
}

func readRequiredMetadataFields[T any](
	values map[string]gguf.Value,
	prefix string,
	valueType gguf.ValueType,
	fields ...metadataField[T],
) error {
	for _, field := range fields {
		value, err := required[T](values, prefix+field.key, valueType)
		if err != nil {
			return err
		}
		*field.destination = value
	}
	return nil
}
