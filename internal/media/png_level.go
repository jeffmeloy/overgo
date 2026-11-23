package media

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"hash/crc32"
)

// pngLevel is the deflate level a picture's PNG is written at: ComfyUI's SaveImage's compress_level (nodes.py, the Moss
// install's), which the MOSS Ultra workflow saves its pictures at. image/png's default, level 6, had taken 211 to 264 ms
// a 1024 x 576 MOSS picture, about half of each section's decode part, where level 4 takes 55 to 66 ms for 4% more bytes
// (png-level); the pixels are the same at any level.
const pngLevel = 4

// pngFilters are PNG's five row filters, in their type's order: none, sub, up, average and Paeth.
const pngFilters = 5

// writeRGBPNG is an 8-bit RGB PNG of rows, height rows of width pixels, deflated at pngLevel, each row filtered by the
// filter whose bytes sum least in absolute value, as image/png and libpng choose it.
func writeRGBPNG(rows []byte, width, height int) ([]byte, error) {
	var out bytes.Buffer
	out.WriteString("\x89PNG\r\n\x1a\n")
	chunk := func(kind string, data []byte) {
		var length, sum [4]byte
		binary.BigEndian.PutUint32(length[:], uint32(len(data)))
		out.Write(length[:])
		crc := crc32.NewIEEE()
		crc.Write([]byte(kind))
		crc.Write(data)
		out.WriteString(kind)
		out.Write(data)
		binary.BigEndian.PutUint32(sum[:], crc.Sum32())
		out.Write(sum[:])
	}
	const bitDepth, truecolor = 8, 2
	header := make([]byte, 13)
	binary.BigEndian.PutUint32(header[0:], uint32(width))
	binary.BigEndian.PutUint32(header[4:], uint32(height))
	header[8], header[9] = bitDepth, truecolor
	chunk("IHDR", header)
	var compressed bytes.Buffer
	writer, err := zlib.NewWriterLevel(&compressed, pngLevel)
	if err != nil {
		return nil, err
	}
	stride := width * RGBChannels
	previous := make([]byte, stride)
	var filtered [pngFilters][]byte
	for kind := range filtered {
		filtered[kind] = make([]byte, stride+1)
		filtered[kind][0] = byte(kind)
	}
	for y := range height {
		row := rows[y*stride : (y+1)*stride]
		best, least := 0, -1
		for kind := range pngFilters {
			sum, line := 0, filtered[kind][1:]
			for i, value := range row {
				var left, upLeft byte
				if i >= RGBChannels {
					left, upLeft = row[i-RGBChannels], previous[i-RGBChannels]
				}
				up := previous[i]
				switch kind {
				case 1:
					value -= left
				case 2:
					value -= up
				case 3:
					value -= byte((int(left) + int(up)) / 2)
				case 4:
					value -= paeth(left, up, upLeft)
				}
				line[i] = value
				sum += absoluteFiltered(value)
			}
			if least < 0 || sum < least {
				best, least = kind, sum
			}
		}
		if _, err := writer.Write(filtered[best]); err != nil {
			return nil, err
		}
		copy(previous, row)
	}
	if err := writer.Close(); err != nil {
		return nil, err
	}
	chunk("IDAT", compressed.Bytes())
	chunk("IEND", nil)
	return out.Bytes(), nil
}

// absoluteFiltered is a filtered byte's distance from zero, read as signed, as the filter choice weighs it.
func absoluteFiltered(value byte) int {
	if signed := int(int8(value)); signed < 0 {
		return -signed
	}
	return int(int8(value))
}

// paeth is the Paeth predictor of a byte from the bytes to its left, above it and above its left.
func paeth(left, up, upLeft byte) byte {
	estimate := int(left) + int(up) - int(upLeft)
	distance := func(of byte) int {
		if d := estimate - int(of); d >= 0 {
			return d
		}
		return int(of) - estimate
	}
	toLeft, toUp, toUpLeft := distance(left), distance(up), distance(upLeft)
	switch {
	case toLeft <= toUp && toLeft <= toUpLeft:
		return left
	case toUp <= toUpLeft:
		return up
	}
	return upLeft
}
