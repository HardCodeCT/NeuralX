// ============================================================================
// motion_detector_known_limitation.cpp
// (excerpt from motion_detector.h's DetectMotion() documentation, verbatim)
//
// This is the actual comment sitting above DetectMotion() in the real
// codebase -- not a paraphrase. Reproduced here because the comment IS the
// content worth showing: a documented, tested limitation with a concrete
// measured example, written the way a design doc or code review comment
// should be, rather than glossed over.
// ============================================================================

// outCopyRects: cleared and filled with verified moved-block rectangles
// (already coalesced -- adjacent blocks sharing an identical motion
// vector are merged into one rect each, not emitted individually).
// outLeftoverBox: the sub-region of dirtyBox NOT explained by any copy
// rect (i.e. genuinely new pixel content, or motion detection was
// skipped/found nothing) -- still needs the existing raw-pixel path.
// May be BoundingBox::empty() if every block was explained.
//
// ---- KNOWN, TESTED LIMITATION: outLeftoverBox is a single bounding
// box, same representation the pre-existing diff_detector.h dirty
// rectangle already uses. This means the WIRE-BANDWIDTH win from
// motion detection is inconsistent, even when the underlying pixel
// explanation rate is high: if the unexplained pixels form a pattern
// that touches multiple edges of dirtyBox (a common case -- e.g. a
// scrolled region's true content boundary sitting a few pixels off
// from a 16px block boundary means blocks straddling that boundary
// can never validate a single rigid-translation match, and if that
// happens along a full edge, the bounding box snaps back out to
// dirtyBox's full extent even though the interior was mostly
// explained), outLeftoverBox stops shrinking even though most pixels
// WERE genuinely explained -- verified directly with a synthetic
// scroll test: 91% of pixels explained by motion, but the leftover
// bounding box came back unchanged because the unexplained ~9% formed
// a border-touching band. The copy rects themselves are always
// correct in this situation (that's not in question, they're
// pixel-verified); what's inconsistent is specifically how much wire
// bandwidth gets saved on any given frame. A tighter representation
// (a short list of leftover rectangles instead of one, or a coarse
// bitmap) would fix this but is a separate, larger change to both the
// wire protocol and the receiver's parsing -- out of scope here. ----
//
// Returns true if motion detection actually ran (false only when the
// search region exceeded kMaxSearchRegionPixels and was skipped
// entirely -- in that case outCopyRects is empty and outLeftoverBox ==
// dirtyBox unchanged, i.e. behaves exactly like the pre-motion-detection
// pipeline).
bool DetectMotion(const uint8_t *prevFull, const uint8_t *currFull, int strideBytes,
                   int fullW, int fullH, const diffdet::BoundingBox &dirtyBox,
                   std::vector<wire::CopyRect> &outCopyRects,
                   diffdet::BoundingBox &outLeftoverBox);
