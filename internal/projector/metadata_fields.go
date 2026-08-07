package projector

import "overgo/internal/gguf"

type metadataIntField struct {
	key    string
	target *int
}

func readMetadataIntFields(file *gguf.File, fields ...metadataIntField) error {
	for _, field := range fields {
		value, err := metadataUint32(file, field.key)
		if err != nil {
			return err
		}
		*field.target = int(value)
	}
	return nil
}
