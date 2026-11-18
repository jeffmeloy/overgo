package latentimage

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"unicode"

	"overgo/internal/checked"
)

// The MOSS Ultra workflow's prompt mode 3 (fixtures/comfy/moss-ultra/requests/text-to-image.json, subgraph 707:702)
// takes its prompt from KJNodes' Ideogram 4 prompt builder (450), whose compact caption JSON it takes apart into a
// prompt of eight fields (JsonExtractString 488-495 behind RegexExtract 486 and 487, StringFormat 496) and whose
// elements it takes through Create Bounding Boxes on the canvas (485, 434), laying both out as MOSS's layout prompt
// (MOSSImageLayoutPrompt 435). The user's own prompt is not read. LayoutDraft is the builder's fields and regions, and
// its prompt the string those nodes make of them (moss-ultra-sections/mode-three).

// The builder's styles: none, photo (with its camera) and art style.
const (
	draftStyleNone  = "none"
	draftStylePhoto = "photo"
	draftStyleArt   = "art_style"
)

// draftPaletteColours is how many colours a region keeps, as the builder and Create Bounding Boxes keep them.
const draftPaletteColours = 5

// draftGrid is the 0 to 1000 grid the builder and Create Bounding Boxes measure a box's edges in.
const draftGrid = 1000

// LayoutDraft is a request's prompt mode 3 as the workflow's Ideogram 4 builder takes it: an overview of the picture,
// its background, a style (none, photo or art_style; none where the request names none) with the photo's camera or the
// art style, its aesthetics, lighting, medium and palette, the regions drawn on the canvas, and the numbers, from 1, of
// the object regions that are characters ("1, 3"), which the layout node takes.
type LayoutDraft struct {
	Description string         `json:"high_level_description,omitzero"`
	Background  string         `json:"background,omitzero"`
	Style       string         `json:"style,omitzero"`
	Photo       string         `json:"photo,omitzero"`
	ArtStyle    string         `json:"art_style,omitzero"`
	Aesthetics  string         `json:"aesthetics,omitzero"`
	Lighting    string         `json:"lighting,omitzero"`
	Medium      string         `json:"medium,omitzero"`
	Palette     []string       `json:"palette,omitzero" controls:"fields"`
	Regions     []LayoutRegion `json:"regions,omitzero" controls:"fields"`
	Characters  string         `json:"character_boxes,omitzero"`
}

// LayoutRegion is one region as the builder's editor holds it: its corner and extent as shares of the canvas (an
// extent below 0 drawn back from the corner), its kind (text, or an object), a text region's text, its caption and its
// palette.
type LayoutRegion struct {
	X       float64  `json:"x"`
	Y       float64  `json:"y"`
	W       float64  `json:"w"`
	H       float64  `json:"h"`
	Type    string   `json:"type,omitzero"`
	Text    string   `json:"text,omitzero"`
	Desc    string   `json:"desc,omitzero"`
	Palette []string `json:"palette,omitzero" controls:"fields"`
}

// style is the draft's style as the builder reads it: none where it names none.
func (d LayoutDraft) style() string {
	if d.Style == "" {
		return draftStyleNone
	}
	return d.Style
}

// validate refuses a style the builder does not offer and a region whose corner or extent is no finite number.
func (d LayoutDraft) validate() error {
	switch d.style() {
	case draftStyleNone, draftStylePhoto, draftStyleArt:
	default:
		return fmt.Errorf("latent image: layout style %q is none of none, photo and art_style", d.Style)
	}
	for index, region := range d.Regions {
		if !checked.Finite64(region.X) || !checked.Finite64(region.Y) || !checked.Finite64(region.W) || !checked.Finite64(region.H) {
			return fmt.Errorf("latent image: layout region %d needs a finite corner and extent", index+1)
		}
	}
	return nil
}

// errLayoutBesidePrompt refuses a request giving a prompt beside a layout, whose fields are its prompt.
var errLayoutBesidePrompt = errors.New("latent image: a layout's fields are its prompt; give the one or the other")

// prompt is the layout prompt the workflow's mode 3 makes of the draft on a canvas of width by height: its text laid
// out with its regions as Create Bounding Boxes gives them.
func (d LayoutDraft) prompt(width, height int) string {
	elements := make([]layoutElement, len(d.Regions))
	for index, region := range d.Regions {
		elements[index] = region.element(width, height)
	}
	return layoutPrompt(d.text(), elements, width, height, d.Characters)
}

// text is the builder's fields as the extracts read them from its caption, in StringFormat 496's eight lines.
func (d LayoutDraft) text() string {
	var scene, camera, artStyle, aesthetics, lighting, medium, palette string
	// The builder writes an overview that is no more than whitespace not at all.
	if strings.TrimFunc(d.Description, pythonSpace) != "" {
		scene = d.Description
	}
	if style := d.style(); style != draftStyleNone {
		aesthetics, lighting, medium = d.Aesthetics, d.Lighting, d.Medium
		if style == draftStylePhoto {
			camera = d.Photo
		} else {
			artStyle = d.ArtStyle
		}
		// JsonExtractString gives a list as Python's str gives it.
		if colours := upperColours(d.Palette); len(colours) > 0 {
			palette = pythonListRepr(colours)
		}
	}
	return "Scene: " + scene + "\nBackground: " + d.Background + "\nAesthetics: " + aesthetics + "\nLighting: " + lighting +
		"\nCamera: " + camera + "\nMedium: " + medium + "\nArt style: " + artStyle + "\nPalette: " + palette
}

// element is the region as Create Bounding Boxes gives it back on a canvas of width by height: the builder's box on
// the 0 to 1000 grid (draftBox), rounded to the canvas's pixels and back to the grid, its kind text or obj, a text
// region's text, its caption, and its first five colours upper-cased.
func (r LayoutRegion) element(width, height int) layoutElement {
	builder := draftBox(r.X, r.Y, r.W, r.H)
	// Create Bounding Boxes reads the builder's box as pixels (elements_to_boxes) and the pixels as shares
	// (boxes_to_regions), Python rounding half to even.
	ymin, xmin, ymax, xmax := builder[0]/draftGrid, builder[1]/draftGrid, builder[2]/draftGrid, builder[3]/draftGrid
	x := math.RoundToEven(math.Min(xmin, xmax) * float64(width))
	y := math.RoundToEven(math.Min(ymin, ymax) * float64(height))
	w := math.RoundToEven(math.Abs(xmax-xmin) * float64(width))
	h := math.RoundToEven(math.Abs(ymax-ymin) * float64(height))
	element := layoutElement{Type: "obj", Desc: r.Desc,
		Box: draftBox(x/float64(max(width, 1)), y/float64(max(height, 1)), w/float64(max(width, 1)), h/float64(max(height, 1)))}
	if r.Type == "text" {
		element.Type, element.Text = "text", r.Text
	}
	if colours := upperColours(r.Palette); len(colours) > 0 {
		element.Palette = colours[:min(len(colours), draftPaletteColours)]
	}
	return element
}

// draftBox is a box of shares as the builder and Create Bounding Boxes put it on the 0 to 1000 grid: each edge rounded
// half to even and held to the grid, the lesser of each pair first, [ymin, xmin, ymax, xmax].
func draftBox(x, y, w, h float64) []float64 {
	grid := func(value float64) float64 { return max(0, min(draftGrid, math.RoundToEven(value*draftGrid))) }
	ymin, xmin, ymax, xmax := grid(y), grid(x), grid(y+h), grid(x+w)
	return []float64{min(ymin, ymax), min(xmin, xmax), max(ymin, ymax), max(xmin, xmax)}
}

// upperColours is a palette as the builder and Create Bounding Boxes normalize it: each colour upper-cased, empty ones
// dropped.
func upperColours(palette []string) []string {
	var colours []string
	for _, colour := range palette {
		if colour != "" {
			colours = append(colours, strings.ToUpper(colour))
		}
	}
	return colours
}

// pythonListRepr is a list of strings as Python's str writes it: ['a', 'b'].
func pythonListRepr(values []string) string {
	quoted := make([]string, len(values))
	for index, value := range values {
		quoted[index] = pythonRepr(value)
	}
	return "[" + strings.Join(quoted, ", ") + "]"
}

// pythonRepr is a string as Python's repr writes it: in single quotes unless it holds one and no double quote, the
// backslash, the quote and the common control characters escaped, and every character Python does not print as its
// \x, \u or \U escape.
func pythonRepr(value string) string {
	quote := '\''
	if strings.ContainsRune(value, '\'') && !strings.ContainsRune(value, '"') {
		quote = '"'
	}
	var out strings.Builder
	out.WriteRune(quote)
	for _, r := range value {
		switch {
		case r == '\\' || r == quote:
			out.WriteRune('\\')
			out.WriteRune(r)
		case r == '\t':
			out.WriteString(`\t`)
		case r == '\n':
			out.WriteString(`\n`)
		case r == '\r':
			out.WriteString(`\r`)
		case unicode.IsPrint(r):
			out.WriteRune(r)
		case r < 0x100:
			fmt.Fprintf(&out, `\x%02x`, r)
		case r < 0x10000:
			fmt.Fprintf(&out, `\u%04x`, r)
		default:
			fmt.Fprintf(&out, `\U%08x`, r)
		}
	}
	out.WriteRune(quote)
	return out.String()
}
