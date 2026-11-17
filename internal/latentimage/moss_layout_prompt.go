package latentimage

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode"

	"overgo/internal/checked"
)

// The MOSS Ultra workflow's prompt mode 3 makes MOSS's layout prompt from boxes, as ComfyUI's MOSSImageLayoutPrompt
// does (fixtures/comfy/moss-ultra/requests/text-to-image.json, 707:702:435), held to that node on the same boxes
// (moss-ultra-sections/prompt-strings). It reads text as Python reads it: str.strip's whitespace, \d's digits.

// pythonSpace is whether r is whitespace to Python's re without the ASCII flag (str.isspace): Unicode's spaces and
// the four information separators, U+001C to U+001F.
func pythonSpace(r rune) bool { return unicode.IsSpace(r) || r >= 0x1C && r <= 0x1F }

// LayoutBoxes are a request's boxes, which make its prompt MOSS's layout prompt as the workflow's mode 3 makes it
// (layoutPrompt): the boxes as ComfyUI's Create Bounding Boxes gives them, and the numbers, from 1, of the object boxes
// that are characters, as its character_boxes input takes them ("1, 3").
type LayoutBoxes struct {
	Boxes      []LayoutBox `json:"boxes"`
	Characters string      `json:"character_boxes,omitzero"`
}

// LayoutBox is one box as Create Bounding Boxes gives it: its kind (text or another), its caption, a text box's text,
// its palette, and its edges [y0, x0, y1, x1] from 0 to 1000.
type LayoutBox struct {
	Type    string    `json:"type,omitzero"`
	Desc    string    `json:"desc,omitzero"`
	Text    string    `json:"text,omitzero"`
	Palette []string  `json:"color_palette,omitzero"`
	Box     []float64 `json:"bbox"`
}

// layoutEdges is a box's edge count, yxyx.
const layoutEdges = 4

// validate refuses a box without its four finite edges, which ComfyUI's node unpacks.
func (l LayoutBoxes) validate() error {
	for index, box := range l.Boxes {
		finite := len(box.Box) == layoutEdges
		for _, edge := range box.Box {
			finite = finite && checked.Finite64(edge)
		}
		if !finite {
			return fmt.Errorf("latent image: layout box %d needs four finite edges [y0, x0, y1, x1] from 0 to 1000", index+1)
		}
	}
	return nil
}

// layoutPrompt is MOSSImageLayoutPrompt: the prompt as it is where there are no boxes, else MOSS's layout JSON: the
// prompt trimmed as its summary, each box an item labelled text, character (its number, from 1, among characters) or
// object, its caption, a text box's text, its palette, and its edges in pixels about the canvas centre, the near edges
// floored and the far ones ceiled after rounding to 9 places, written compactly as Python's json.dumps writes it.
func layoutPrompt(prompt string, boxes LayoutBoxes, width, height int) string {
	if len(boxes.Boxes) == 0 {
		return prompt
	}
	numbered := map[int]bool{}
	for _, field := range strings.FieldsFunc(boxes.Characters, func(r rune) bool { return !unicode.IsDigit(r) }) {
		numbered[decimalValue(field)] = true
	}
	var out strings.Builder
	out.WriteString(`{"summary":{"caption":`)
	writePythonJSONString(&out, strings.TrimFunc(prompt, pythonSpace))
	out.WriteString(`},"items":[`)
	for index, element := range boxes.Boxes {
		if index > 0 {
			out.WriteByte(',')
		}
		label := "object"
		if element.Type == "text" {
			label = "text"
		} else if numbered[index+1] {
			label = "character"
		}
		out.WriteString(`{"label":`)
		writePythonJSONString(&out, label)
		out.WriteString(`,"caption":`)
		writePythonJSONString(&out, element.Desc)
		if label == "text" {
			out.WriteString(`,"text":`)
			writePythonJSONString(&out, element.Text)
		}
		if len(element.Palette) > 0 {
			out.WriteString(`,"palette":[`)
			for p, colour := range element.Palette {
				if p > 0 {
					out.WriteByte(',')
				}
				writePythonJSONString(&out, colour)
			}
			out.WriteByte(']')
		}
		edge := func(value float64, side int, ceil bool) int {
			scaled := pythonRound9(value * float64(side) / 1000)
			if ceil {
				return int(math.Ceil(scaled)) - side/2
			}
			return int(math.Floor(scaled)) - side/2
		}
		y0, x0, y1, x1 := element.Box[0], element.Box[1], element.Box[2], element.Box[3]
		fmt.Fprintf(&out, `,"bbox":[%d,%d,%d,%d]}`, edge(y0, height, false), edge(x0, width, false), edge(y1, height, true), edge(x1, width, true))
	}
	out.WriteString(`],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}`)
	return out.String()
}

// decimalValue is Python's int of a run of decimal digits, \d's Unicode digits of any script among them: each digit's
// value is its place in its script's run of ten, Unicode keeping each script's digits as a run from zero. A value past
// any box's number saturates.
func decimalValue(digits string) int {
	value := 0
	for _, r := range digits {
		zero := r
		for unicode.IsDigit(zero - 1) {
			zero--
		}
		value = min(value*10+int(r-zero)%10, math.MaxInt32)
	}
	return value
}

// pythonRound9 is Python's round(value, 9): the value correctly rounded to 9 decimal places.
func pythonRound9(value float64) float64 {
	rounded, _ := strconv.ParseFloat(strconv.FormatFloat(value, 'f', 9, 64), 64)
	return rounded
}

// writePythonJSONString writes s as Python's json.dumps writes a string with ensure_ascii off: quotes, backslash and
// the control characters escaped, every other character as it is.
func writePythonJSONString(out *strings.Builder, s string) {
	out.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			out.WriteString(`\"`)
		case '\\':
			out.WriteString(`\\`)
		case '\n':
			out.WriteString(`\n`)
		case '\r':
			out.WriteString(`\r`)
		case '\t':
			out.WriteString(`\t`)
		case '\b':
			out.WriteString(`\b`)
		case '\f':
			out.WriteString(`\f`)
		default:
			if r < 0x20 {
				fmt.Fprintf(out, `\u%04x`, r)
			} else {
				out.WriteRune(r)
			}
		}
	}
	out.WriteByte('"')
}
