/*
 * SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "log_macros.h"
#include "common_util.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

extern "C" {
#include "time_counter.h"
#include "wrapper.h"
#include "landmark_wrapper.h"
#include "palm_postprocess.h"
#include "palm_preprocess.h"
#include "landmark_preprocess.h"
#include "landmark_postprocess.h"
#include "./ai_application/ruhmi_conversion_results/sub_0000_tensors.h"
#include "./ai_application/ruhmi_conversion_results/sub_0002_tensors.h"
#include "./ai_application/ruhmi_landmark_results/sub_0000__landmark_tensors.h"
extern uint8_t sub_0000_arena[];
extern uint8_t sub_0002_arena[];
#define sub_0000__landmark_arena sub_0000_arena  /* shared SRAM — never run simultaneously */
void update_detection_result(uint16_t index, signed short  x, signed short  y, signed short  w, signed short  h);
#include "landmark_display.h"

/* Forward-declared rather than #include "console_output.h", which would
 * redeclare sprintf_buffer (already visible via common_util.h) with a
 * different language linkage and fail to compile. */
vision_ai_app_err_t print_to_console(char * p_data);

/* Camera RGB565 buffer — declared in camera_layer.h */
extern uint8_t camera_capture_image_rgb565[];

/* Palm preprocess meta — declared in camera_display_thread_entry.c */
extern palm_preprocess_meta_t g_palm_preprocess_meta;
}

#define IMAGE_DATA_SIZE  (AI_INPUT_IMAGE_WIDTH * AI_INPUT_IMAGE_HEIGHT * AI_INPUT_IMAGE_BYTE_PER_PIXEL)
extern int8_t model_buffer_int8[IMAGE_DATA_SIZE];

/* Maximum number of hands to run landmark on (limit to reduce SDRAM bus load) */
#define MAX_LANDMARK_HANDS  2

/* Frame geometry as floats, so none of the maths below silently promotes the
 * camera_size_list_t enum. */
#define CAM_W_F     ((float)CAM_VGA_WIDTH)      /* 640 */
#define CAM_H_F     ((float)CAM_VGA_HEIGHT)     /* 480 */
#define CAM_REF_F   (CAM_W_F)                   /* larger of the two, matches sqn_rr_size's reference */

/* Palm model quantization parameters (from RUHMI perf eval) */
#define PALM_NUM_ANCHORS       2016
#define PALM_SCORES_SCALE      0.00390625f
#define PALM_SCORES_ZP         (-128)
#define PALM_BOXES_SCALE       0.0070548211f
#define PALM_BOXES_ZP          (-80)
#define PALM_KP_SCALE          0.005439954f
#define PALM_KP_ZP             (-106)

/* Static buffers for dequantized outputs — keep in SRAM to avoid SDRAM bus contention with LCD */
static float dequant_scores[PALM_NUM_ANCHORS];
static float dequant_boxes[PALM_NUM_ANCHORS * 4];
static float dequant_kp0[PALM_NUM_ANCHORS * 2];
static float dequant_kp2[PALM_NUM_ANCHORS * 2];
static palm_box_t palm_results[MAX_LANDMARK_HANDS];

/* Landmark model INT8 input tensor (224×224×3 = 150528 bytes)
 * Moved to SDRAM to free SRAM for the shared arena.
 * Only accessed by CPU briefly (preprocess + memcpy), not during NPU inference. */
static int8_t landmark_input_buf[LANDMARK_INPUT_SIZE] BSP_ALIGN_VARIABLE(32) BSP_PLACE_IN_SECTION(BSP_UNINIT_SECTION_PREFIX ".sdram_noinit");


/* ===========================================================================
 * 1 Euro filter (Casiez et al., CHI 2012)
 *
 * A plain EMA forces a choice between "steady when still" and "responsive when
 * moving". The 1 Euro filter avoids it by making the cutoff frequency track
 * the estimated speed of the signal: heavy smoothing while the input sits
 * still, almost none while it moves fast.
 *
 * All state is caller-owned; zero-initialised state means "not started yet",
 * so memset(0) is a complete reset.
 * ======================================================================== */
typedef struct
{
    float x_prev;
    float dx_prev;
    int   initialized;
} one_euro_t;

static inline float one_euro_alpha(float cutoff, float dt)
{
    const float tau = 1.0f / (2.0f * 3.14159265358979323846f * cutoff);
    return 1.0f / (1.0f + (tau / dt));
}

static inline float one_euro_filter(one_euro_t * p_f, float x, float dt,
                                    float min_cutoff, float beta, float d_cutoff)
{
    if (!p_f->initialized)
    {
        p_f->x_prev      = x;
        p_f->dx_prev     = 0.0f;
        p_f->initialized = 1;
        return x;
    }

    /* Derivative, itself low-passed so quantisation noise in the input does
     * not leak into the speed estimate and open the cutoff up. */
    const float dx      = (x - p_f->x_prev) / dt;
    const float alpha_d = one_euro_alpha(d_cutoff, dt);
    const float dx_hat  = (alpha_d * dx) + ((1.0f - alpha_d) * p_f->dx_prev);
    p_f->dx_prev = dx_hat;

    /* Cutoff rises with speed, then an ordinary first-order low-pass. */
    const float cutoff = min_cutoff + (beta * fabsf(dx_hat));
    const float alpha  = one_euro_alpha(cutoff, dt);
    const float x_hat  = (alpha * x) + ((1.0f - alpha) * p_f->x_prev);
    p_f->x_prev = x_hat;

    return x_hat;
}

/* Tuning. Everything fed to the filters below is normalised to the frame, so
 * "speed" is in frames/second — a hand crossing half the frame in 0.3 s runs
 * at about 1.7. min_cutoff dominates while the hand is still (lower = steadier
 * but slower to settle); beta dominates while it moves (higher = less lag).
 *
 * Measured at the fingertip, in display pixels, against a simulated palm
 * keypoint stream (0.5 LSB sensor noise, then INT8 output quantisation,
 * 20 fps):
 *
 *     (min_cutoff, beta)      still jitter    lag while sweeping
 *     unfiltered                  5.24 px           2.8 px
 *     (0.5,  1.0)                 1.72 px          29.9 px
 *     (0.5,  4.0)                 2.31 px          13.6 px
 *     (0.5,  8.0)  <- default     2.74 px           8.6 px
 *     (0.5, 16.0)                 3.29 px           5.4 px
 *
 * Raise beta if the landmarks trail the hand, lower it if they still shimmer
 * while the hand is held still. Size gets a gentler setting than the centre
 * because a hand's apparent size changes far more slowly than its position.
 *
 * Note the table above is for a SINGLE filter stage. There are two here in
 * series — the crop rectangle, then the displayed points — so their lags add.
 * That is why the betas below sit at the responsive end of the table rather
 * than at its measured optimum: half the smoothing budget is already spent by
 * the time the second stage runs. The point stage is pushed furthest because
 * its input has already been cleaned up by the rectangle stage, so it only has
 * the landmark model's own quantisation noise left to remove (~1.11 px/LSB)
 * and does not need much authority to do it. */
/* min_cutoff was lowered from the 0.4-0.5 above to the values below to cut
 * resting jitter. The two coefficients act at different times, so this does
 * not cost motion lag: while the hand moves, the beta term dominates and the
 * cutoff opens far past min_cutoff regardless; min_cutoff only sets the floor
 * the filter settles to once the hand stops.
 *
 * Steady-state noise through a first-order low-pass is input * sqrt(a/(2-a)):
 *
 *   points, stepped every frame (dt 38 ms)
 *     0.50 Hz -> a 0.107 -> 24% of raw noise      (was)
 *     0.25 Hz -> a 0.056 -> 17% of raw noise      (now, ~29% less jitter)
 *
 *   crop rectangle, stepped once per detector run (dt ~114 ms, so the same
 *   cutoff smooths far less here — worth lowering further)
 *     0.50 Hz -> a 0.264 -> 39% of raw noise      (was)
 *     0.20 Hz -> a 0.125 -> 26% of raw noise      (now, ~34% less jitter)
 *
 * The cost is settling time after motion stops: about 1/(2*pi*min_cutoff),
 * i.e. 0.6 s for the points and 0.8 s for the rectangle. Lower these further
 * for a steadier hold, raise them if the landmarks feel like they creep into
 * place after the hand comes to rest. */
/* beta was lowered from 20/16/8 to the values below, trading motion tracking
 * for steadiness at rest. What it buys, and why min_cutoff could not:
 *
 * A static target (a printed hand, so any movement is the system's) still
 * produced two distinct kinds of variation. The continuous part was already
 * small — w within 188..199, h within 209..224, a few px of standard
 * deviation, which min_cutoff governs and which is no longer the problem.
 * On top of that sat discrete events roughly every 1.2 s where h dropped
 * 16..23 px for several frames and came back, while w barely moved: one
 * fingertip landmark jumping and returning, because tiny frame-to-frame
 * differences in the crop flip the INT8 model between two answers.
 *
 * min_cutoff does nothing for those. A 20 px step reads as fast motion to the
 * speed term, so the cutoff opens and the filter passes it: at beta 20 about
 * 57% of the jump landed in the first frame. At beta 6 that is about 30%.
 *
 * Note this only damps the jump, it cannot block it — a step that persists
 * for several frames eventually passes any first-order low-pass. Removing
 * those outright needs a median long enough to span the event, which costs
 * proportionally more lag; this is the cheaper half of that trade.
 *
 * Raise these back towards 16-20 if fast movement now trails the hand too
 * much; the resting jitter comes back with it. */
#define FILT_CENTER_MIN_CUTOFF   (0.2f)
#define FILT_CENTER_BETA         (6.0f)
#define FILT_SIZE_MIN_CUTOFF     (0.15f)
#define FILT_SIZE_BETA           (3.0f)
#define FILT_ROT_MIN_CUTOFF      (0.2f)
#define FILT_ROT_BETA            (6.0f)
#define FILT_POINT_MIN_CUTOFF    (0.25f)
#define FILT_POINT_BETA          (6.0f)

/* Low-pass on the internal speed estimate. This is what decides how fast the
 * filter notices a movement has STARTED: at ~15 fps and 1 Hz, the speed
 * estimate only moves ~30% per frame, so it takes 3-4 frames to open the
 * cutoff up — visible as the landmarks lagging at the start of a fast sweep
 * and then catching up. 2 Hz roughly halves that reaction time. */
#define FILT_D_CUTOFF            (2.0f)

/* Frame interval assumed before a real one has been measured. */
#define FILT_NOMINAL_DT_S        (0.05f)

/* A detection further than this (normalised to the frame) from a slot's
 * previous position is a different hand, not the same one having moved, so it
 * must not inherit that slot's filter history. */
#define SLOT_MATCH_MAX_DIST      (0.25f)


/* How often the palm detector runs, in AI frames. On the frames in between,
 * each hand keeps the crop rectangle from the most recent detector run.
 *
 * Measured cost split on this board: palm detection ~38 ms, landmark ~26 ms.
 * Running the detector every frame therefore costs 64 ms per frame; at 1 in 2
 * that averages 45 ms, at 1 in 3 about 39 ms. Since end-to-end latency is what
 * makes the landmarks trail a moving hand, that is the largest lever available
 * short of changing the architecture.
 *
 * Why a stale rectangle is safe here, unlike deriving it from the landmarks:
 * it is still a real detector measurement, just an old one, so nothing feeds
 * back on itself and there is nothing to compound. The only cost is that the
 * hand is off-centre in the crop. The crop is 2.6x the palm (half-width ~165
 * px), and one skipped frame is ~26 ms, i.e. ~26 px at a brisk 1000 px/s — so
 * the hand stays comfortably inside. If it ever does not, the landmark model's
 * confidence drops, that slot is dropped, and the next detector run
 * re-acquires it; the failure mode is a lost frame, not a runaway.
 *
 * With MP_ROI_TRACKING on this governs only how quickly a NEW hand entering
 * the frame is picked up — a hand already tracked no longer needs the detector
 * — so it is set far longer than the 3 it had to be when the detector supplied
 * every crop. At 10 the worst case for spotting a second hand is about 0.4 s,
 * while the average frame cost drops from (64 + 26*2)/3 = 39 ms to
 * (64 + 26*9)/10 = 30 ms. Acquiring the first hand, and recovering a dropped
 * track, both bypass the interval entirely.
 *
 * Set to 1 to restore detection on every frame. */
#define PALM_DETECT_INTERVAL     (10u)

/* Smallest fraction of the crop that may lie inside the camera frame before
 * this hand's landmarks are refused.
 *
 * A hand leaving the frame is not a tracking failure -- the loop stays on its
 * fixed point throughout, measured at ratio 96..101% all the way out -- it is
 * missing information. The pixels are simply not there, so the model reports
 * the smaller hand it can actually see, the crop follows it down, and the
 * drawn skeleton compresses against the border. Nothing downstream can undo
 * that; the only choice is what to display once the input is incomplete, and
 * showing a squashed hand is worse than showing none.
 *
 * The threshold comes from a capture of a hand moving out to the right,
 * pairing the visible fraction of the crop against the width of the drawn box
 * as a share of its in-frame value:
 *
 *      visible   0.98   0.85   0.78   0.74   0.71   0.68   0.65   0.63
 *      box width  100%    91%    86%    81%    72%    59%    47%    36%
 *
 * Compression is not yet obvious at 0.78 and unmistakable by 0.68, so 0.75
 * drops the overlay while the box is still ~80% of full size, i.e. before the
 * squash reads as a defect. Lower it toward 0.65 to keep tracking further out
 * at the cost of showing some of the squash; raise it to drop sooner. */
#define MP_EDGE_MIN_VISIBLE      (0.75f)

/* The console prints this next to the timings so a captured log states which
 * tuning produced it. Two logs taken across a retune are otherwise hard to
 * tell apart, and a stale terminal scrollback is easy to mistake for a fresh
 * capture. Defined here rather than next to the filter constants because it
 * has to sit after every macro it stringifies — an undefined one would
 * silently stringify to its own name instead of its value. */
#define HT_STR2(x)               #x
#define HT_STR(x)                HT_STR2(x)

extern "C" const char * hand_tuning_tag(void)
{
    return "pt "      HT_STR(FILT_POINT_MIN_CUTOFF)  "/" HT_STR(FILT_POINT_BETA)
           "  rect "  HT_STR(FILT_CENTER_MIN_CUTOFF) "/" HT_STR(FILT_CENTER_BETA)
           "  det 1/" HT_STR(PALM_DETECT_INTERVAL)
           "  edge "   HT_STR(MP_EDGE_MIN_VISIBLE);
}

/* Window used to average the reported pipeline time.
 *
 * With the detector no longer running every frame, the per-frame time
 * legitimately alternates (~64 ms on detector frames, ~26 ms in between),
 * which makes the raw figure unreadable on screen. Averaging over a whole
 * number of detector cycles means the window always holds the same mix of
 * both kinds of frame, so the number only moves when the actual workload
 * does. */
#define PIPE_TIME_AVG_WINDOW     (PALM_DETECT_INTERVAL * 4u)


/* ===========================================================================
 * Per-hand state
 *
 * Two earlier attempts at closed-loop tracking failed and were reverted; the
 * rule they used to turn landmarks back into the next crop was invented rather
 * than taken from the model's convention, and both had their only fixed point
 * at zero, so the crop shrank until the hand was lost. See the block above
 * mp_roi_from_landmarks() for the rule the model was actually trained on.
 *
 * The loop gain of that rule was then measured on this board before it was
 * allowed to drive anything: with the hand held still, so the variation in the
 * crop came from palm-detector noise and was therefore exogenous, the ROI it
 * produced responded with a slope of 0.44 (n=60, 95% CI roughly 0.30..0.56).
 * Below 1 means the loop contracts — a disturbance decays by more than half
 * each frame — which is the property the earlier formulas lacked.
 *
 * A second capture with the hand moving toward and away from the camera shows
 * a slope near 1.0, which looks like a contradiction and is not: there the
 * hand's real size drives both the crop and the ROI, so the correlation is
 * confounded rather than causal. Only the still capture isolates the loop.
 * ======================================================================== */
typedef struct
{
    bool  has_rect;     /* holds a usable crop rect from the last detector run */
    float prev_cx;      /* its raw (unsmoothed) centre, for slot matching */
    float prev_cy;

    /* The smoothed rectangle actually used for the crop, reused as-is on
     * frames where the detector does not run. */
    palm_rotated_hand_t used_rect;

    /* Closed-loop tracking: the ROI derived from this hand's own landmarks
     * last frame, and whether it is currently trusted. Only used when
     * MP_ROI_TRACKING is on. */
    bool                tracking;
    palm_rotated_hand_t track_roi;
    float               ref_size;   /* last detector-derived size, for the clamp */

    /* Crop-rectangle filters. Rotation is filtered as a sin/cos pair rather
     * than as an angle: the pair is continuous across the +-pi wrap, where a
     * raw angle would jump by 2*pi and drag the filter across the frame. */
    one_euro_t f_cx;
    one_euro_t f_cy;
    one_euro_t f_size;
    one_euro_t f_rsin;
    one_euro_t f_rcos;

    /* Display smoothing for the 21 points. */
    one_euro_t f_px[LANDMARK_NUM_POINTS];
    one_euro_t f_py[LANDMARK_NUM_POINTS];
} hand_slot_t;

static hand_slot_t g_hand[MAX_LANDMARK_HANDS];

/*********************************************************************************************************************
 *  hand_slot_reset: drop a slot's history, so a hand that appears there later
 *  starts clean instead of being dragged in from wherever the previous one was.
***********************************************************************************************************************/
static void hand_slot_reset(int slot)
{
    memset(&g_hand[slot], 0, sizeof(g_hand[slot]));
}

/*********************************************************************************************************************
 *  hand_rect_smooth: run this frame's crop rectangle through the slot's
 *  filters, in place.
***********************************************************************************************************************/
static void hand_rect_smooth(hand_slot_t * p_slot, palm_rotated_hand_t * p_rect, float dt_s)
{
    p_rect->sqn_rr_center_x = one_euro_filter(&p_slot->f_cx, p_rect->sqn_rr_center_x, dt_s,
                                              FILT_CENTER_MIN_CUTOFF, FILT_CENTER_BETA, FILT_D_CUTOFF);
    p_rect->sqn_rr_center_y = one_euro_filter(&p_slot->f_cy, p_rect->sqn_rr_center_y, dt_s,
                                              FILT_CENTER_MIN_CUTOFF, FILT_CENTER_BETA, FILT_D_CUTOFF);
    p_rect->sqn_rr_size     = one_euro_filter(&p_slot->f_size, p_rect->sqn_rr_size, dt_s,
                                              FILT_SIZE_MIN_CUTOFF, FILT_SIZE_BETA, FILT_D_CUTOFF);

    const float rs = one_euro_filter(&p_slot->f_rsin, sinf(p_rect->rotation), dt_s,
                                     FILT_ROT_MIN_CUTOFF, FILT_ROT_BETA, FILT_D_CUTOFF);
    const float rc = one_euro_filter(&p_slot->f_rcos, cosf(p_rect->rotation), dt_s,
                                     FILT_ROT_MIN_CUTOFF, FILT_ROT_BETA, FILT_D_CUTOFF);

    /* The filtered pair is no longer unit length, but atan2 only uses the
     * ratio, so the recovered angle is still correct. Guard the degenerate
     * case where both components filter to ~0. */
    if ((fabsf(rs) > 1e-6f) || (fabsf(rc) > 1e-6f))
    {
        p_rect->rotation = atan2f(rs, rc);
    }
}

/*********************************************************************************************************************
 *  landmark_points_smooth: display smoothing of the 21 points, in place.
 *  Points are normalised by the same reference dimension before filtering so
 *  the tuning constants above stay in frames/second units.
***********************************************************************************************************************/
static void landmark_points_smooth(hand_slot_t * p_slot, landmark_result_t * p_lm, float dt_s)
{
    for (int k = 0; k < LANDMARK_NUM_POINTS; k++)
    {
        const float nx = one_euro_filter(&p_slot->f_px[k], (float)p_lm->pts[k].x / CAM_REF_F, dt_s,
                                         FILT_POINT_MIN_CUTOFF, FILT_POINT_BETA, FILT_D_CUTOFF);
        const float ny = one_euro_filter(&p_slot->f_py[k], (float)p_lm->pts[k].y / CAM_REF_F, dt_s,
                                         FILT_POINT_MIN_CUTOFF, FILT_POINT_BETA, FILT_D_CUTOFF);

        p_lm->pts[k].x = (int32_t)lrintf(nx * CAM_REF_F);
        p_lm->pts[k].y = (int32_t)lrintf(ny * CAM_REF_F);
    }
}

/*********************************************************************************************************************
 *  landmark_bounding_box: axis-aligned box (camera pixels) enclosing the 21
 *  points, used as the on-screen detection box.
 *
 *  Deriving the box from the smoothed landmarks rather than from the palm
 *  detector keeps the box and the skeleton consistent — they are then the same
 *  data, so the box cannot disagree with what is drawn inside it, and it
 *  inherits the same smoothing.
***********************************************************************************************************************/
static void landmark_bounding_box(const landmark_result_t * lm,
                                  float * out_x1, float * out_y1,
                                  float * out_w,  float * out_h)
{
    float min_x = (float)lm->pts[0].x;
    float max_x = min_x;
    float min_y = (float)lm->pts[0].y;
    float max_y = min_y;

    for (int k = 1; k < LANDMARK_NUM_POINTS; k++)
    {
        const float px = (float)lm->pts[k].x;
        const float py = (float)lm->pts[k].y;

        if (px < min_x) min_x = px;
        if (px > max_x) max_x = px;
        if (py < min_y) min_y = py;
        if (py > max_y) max_y = py;
    }

    /* Small margin so the fingertips are not sitting exactly on the edge. */
    const float margin = 8.0f;
    min_x -= margin; max_x += margin;
    min_y -= margin; max_y += margin;

    /* Deliberately NOT clamped to the frame.
     *
     * The landmark model extrapolates points past the border as a hand leaves
     * the frame, which is correct and useful. Clamping the box to the frame
     * pins the leading edge to the border while the trailing edge keeps
     * following the hand, so the box collapses against the edge instead of
     * sliding out of view -- it looks like the hand is being squashed against
     * a wall. The display clips overlays to the camera area (see d2_cliprect
     * in do_detection_screen), so an off-frame box now renders correctly: it
     * simply leaves the screen.
     *
     * The bound below is only about what the renderer can represent. d2_point
     * is a signed 16-bit value used as 4-bit fixed point, so display
     * coordinates must stay within roughly +-2000 px; at the 1.25x display
     * scaling that is +-1600 camera px, and +-600 leaves ample headroom. */
    const float bound = 600.0f;

    if (min_x < -bound) min_x = -bound;
    if (min_y < -bound) min_y = -bound;
    if (max_x > (CAM_W_F + bound)) max_x = CAM_W_F + bound;
    if (max_y > (CAM_H_F + bound)) max_y = CAM_H_F + bound;

    /* The display treats a top-left that casts to exactly (0,0) as "no box"
     * and skips it, so keep the origin off that value. The shift is at most
     * one pixel. */
    if ((min_x > -1.0f) && (min_x < 1.0f)) min_x = 1.0f;
    if ((min_y > -1.0f) && (min_y < 1.0f)) min_y = 1.0f;

    *out_x1 = min_x;
    *out_y1 = min_y;
    *out_w  = (max_x > min_x) ? (max_x - min_x) : 1.0f;
    *out_h  = (max_y > min_y) ? (max_y - min_y) : 1.0f;
}


/* ===========================================================================
 * MediaPipe upstream landmarks -> ROI. MEASUREMENT ONLY — nothing downstream
 * reads it.
 *
 * Two attempts at closed-loop tracking failed here, both because the rule for
 * turning landmarks back into the next crop rectangle was invented rather than
 * taken from the model's own convention. The first was 180 deg out and
 * collapsed within frames; the second was corrected against an open-loop
 * calibration and still shrank monotonically (176 -> 145 px, lose, reacquire,
 * repeat) — a calibration taken at one operating point does not survive the
 * loop moving that operating point.
 *
 * This is the real rule, transcribed from upstream MediaPipe
 * (hand_landmark_landmarks_to_roi.pbtxt: HandLandmarksToRectCalculator, then
 * RectTransformationCalculator with scale 2.0, shift_y -0.1, square_long). It
 * differs from both guesses in ways that matter:
 *
 *   - twelve landmarks {0,1,2,3,5,6,9,10,13,14,17,18} — wrist and finger bases,
 *     NOT the fingertips, which is exactly where this INT8 model's intermittent
 *     jumps live, so those jumps cannot move the ROI at all;
 *   - the size is the long side of their rotated bounding box, not a two-point
 *     distance through an empirical constant that amplifies any error in it;
 *   - a shift_y of -0.1, a systematic offset neither attempt had an equivalent
 *     of — precisely the kind of bias a feedback loop compounds.
 *
 * And the reason this one has grounds to be stable where the others did not:
 * the model was trained on crops produced by this rule, so this rule is the
 * loop's fixed point. Feed a crop in the training convention, the landmarks
 * come back accurate, and re-deriving the rectangle reproduces the same crop.
 * An invented formula has no reason to close on itself like that.
 *
 * Measured here beside the rectangle actually in use, so the relationship can
 * be established before anything is allowed to depend on it.
 * ======================================================================== */
#define MP_ROI_MEASURE_ONLY      (0)

/* Feed the rule back in: a hand whose landmarks came out valid supplies its own
 * crop for the next frame, and the palm detector is only consulted for slots
 * that are not tracking. Set to 0 to fall back to detector-only crops.
 *
 * Justified by the measured loop gain of 0.44 above, but not relying on it:
 * MP_TRACK_MIN/MAX_VS_REF below clamp the tracked size against the last size
 * the detector produced for that slot, so even a pathological frame cannot
 * walk the crop away. A landmark result below the confidence threshold drops
 * the track outright and hands the slot back to the detector. */
#define MP_ROI_TRACKING          (1)

/* Bounds for a tracked crop, as a fraction of the last detector-derived size
 * for that slot. Wide enough not to fight the loop settling into its fixed
 * point (measured 1.0x to 1.2x of the detector size depending on distance),
 * tight enough that a runaway is impossible. */
#define MP_TRACK_MIN_VS_REF      (0.6f)
#define MP_TRACK_MAX_VS_REF      (1.8f)

/*********************************************************************************************************************
 *  crop_visible_fraction: how much of a crop rectangle falls inside the camera
 *  frame, as a fraction of its own area.
 *
 *  The crop is a rotated square, but rotation is ignored here: the measure
 *  only feeds a threshold, and the axis-aligned square is both the cheaper and
 *  the more conservative reading of it. Evaluated on the crop rather than on
 *  the hand because the crop is what the model is fed, and it is known before
 *  the model runs -- so a hand that fails this test costs no inference.
***********************************************************************************************************************/
static float crop_visible_fraction(const palm_rotated_hand_t * p_rect)
{
    const float side = p_rect->sqn_rr_size * CAM_REF_F;

    if (side <= 0.0f)
    {
        return 0.0f;
    }

    const float half = side * 0.5f;
    const float cx   = p_rect->sqn_rr_center_x * CAM_W_F;
    const float cy   = p_rect->sqn_rr_center_y * CAM_H_F;

    float x1 = cx - half;
    float x2 = cx + half;
    float y1 = cy - half;
    float y2 = cy + half;

    if (x1 < 0.0f)     x1 = 0.0f;
    if (y1 < 0.0f)     y1 = 0.0f;
    if (x2 > CAM_W_F)  x2 = CAM_W_F;
    if (y2 > CAM_H_F)  y2 = CAM_H_F;

    const float vw = (x2 > x1) ? (x2 - x1) : 0.0f;
    const float vh = (y2 > y1) ? (y2 - y1) : 0.0f;

    return (vw * vh) / (side * side);
}

#if (MP_ROI_MEASURE_ONLY == 1)
/* Print every Nth valid hand, to keep the console readable. */
#define MP_ROI_LOG_STRIDE        (3)
#endif

#if ((MP_ROI_MEASURE_ONLY == 1) || (MP_ROI_TRACKING == 1))
static void mp_roi_from_landmarks(const landmark_result_t * lm, palm_rotated_hand_t * out)
{
    /* Wrist plus the finger base joints; deliberately no fingertips. */
    static const uint8_t k_subset[12] = { 0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18 };

    /* Rotation: wrist -> mean of the index/middle/ring MCPs, brought to a
     * target angle of pi/2. */
    const float x0 = (float)lm->pts[0].x;
    const float y0 = (float)lm->pts[0].y;
    const float x1 = ((float)lm->pts[5].x + (float)lm->pts[9].x + (float)lm->pts[13].x) / 3.0f;
    const float y1 = ((float)lm->pts[5].y + (float)lm->pts[9].y + (float)lm->pts[13].y) / 3.0f;

    float rot = 1.5707963f - atan2f(-(y1 - y0), x1 - x0);
    while (rot >  3.14159265f) rot -= 6.28318531f;
    while (rot < -3.14159265f) rot += 6.28318531f;

    /* Axis-aligned centre of the subset, used as the rotation origin. */
    float ax_min = (float)lm->pts[k_subset[0]].x, ax_max = ax_min;
    float ay_min = (float)lm->pts[k_subset[0]].y, ay_max = ay_min;

    for (int i = 1; i < 12; i++)
    {
        const float px = (float)lm->pts[k_subset[i]].x;
        const float py = (float)lm->pts[k_subset[i]].y;

        if (px < ax_min) ax_min = px;
        if (px > ax_max) ax_max = px;
        if (py < ay_min) ay_min = py;
        if (py > ay_max) ay_max = py;
    }

    const float aacx = 0.5f * (ax_min + ax_max);
    const float aacy = 0.5f * (ay_min + ay_max);

    /* Bounding box of the subset once rotated into the hand's own frame. */
    const float c_neg = cosf(-rot);
    const float s_neg = sinf(-rot);

    float rx_min = 0.0f, rx_max = 0.0f, ry_min = 0.0f, ry_max = 0.0f;

    for (int i = 0; i < 12; i++)
    {
        const float dx = (float)lm->pts[k_subset[i]].x - aacx;
        const float dy = (float)lm->pts[k_subset[i]].y - aacy;
        const float rx = (dx * c_neg) - (dy * s_neg);
        const float ry = (dx * s_neg) + (dy * c_neg);

        if (i == 0)
        {
            rx_min = rx_max = rx;
            ry_min = ry_max = ry;
        }
        else
        {
            if (rx < rx_min) rx_min = rx;
            if (rx > rx_max) rx_max = rx;
            if (ry < ry_min) ry_min = ry;
            if (ry > ry_max) ry_max = ry;
        }
    }

    const float box_w = rx_max - rx_min;
    const float box_h = ry_max - ry_min;
    const float pcx   = 0.5f * (rx_min + rx_max);
    const float pcy   = 0.5f * (ry_min + ry_max);

    /* Rotate the projected centre back into image coordinates. */
    const float c_pos = cosf(rot);
    const float s_pos = sinf(rot);

    float cx = (pcx * c_pos) - (pcy * s_pos) + aacx;
    float cy = (pcx * s_pos) + (pcy * c_pos) + aacy;

    /* RectTransformationCalculator applies the shift first, against the
     * pre-scale size, then square_long, then the scale. */
    const float shift_y = -0.1f;
    cx += -box_h * shift_y * s_pos;
    cy +=  box_h * shift_y * c_pos;

    float side = (box_w > box_h) ? box_w : box_h;   /* square_long */
    side *= 2.0f;                                    /* scale_x = scale_y = 2.0 */

    out->sqn_rr_center_x = cx / CAM_W_F;
    out->sqn_rr_center_y = cy / CAM_H_F;
    out->sqn_rr_size     = side / CAM_REF_F;
    out->rotation        = rot;
}
#endif /* MP_ROI_MEASURE_ONLY || MP_ROI_TRACKING */


/*********************************************************************************************************************
 *  main_loop_palm_detection function: runs palm detection inference and post-processing.
 *  @param[IN]   None
 *  @retval      true   successful execution.
 *               false  handler call failed.
***********************************************************************************************************************/
bool main_loop_palm_detection()
{
    /* Start pipeline timer (covers pre-copy + inference + post-processing + landmark) */
    volatile uint32_t pipeline_start = TimeCounter_CurrentCountGet();

    /* Frame interval for the point filters, which step every frame. Clamped so
     * the first frame or a stall (debugger halt, dropped frames) cannot feed a
     * nonsense dt in. */
    static uint32_t s_prev_start = 0;
    static bool     s_have_prev  = false;

    float dt_s = FILT_NOMINAL_DT_S;
    if (s_have_prev)
    {
        const uint32_t dt_ms = TimeCounter_CountValueConvertToMs(s_prev_start, (uint32_t)pipeline_start);
        if ((dt_ms >= 1u) && (dt_ms <= 500u))
        {
            dt_s = (float)dt_ms * 0.001f;
        }
    }
    s_prev_start = (uint32_t)pipeline_start;
    s_have_prev  = true;

    /* Separate interval for the crop-rectangle filters, which only step on
     * detector frames — every PALM_DETECT_INTERVAL frames, not every frame.
     *
     * Feeding them the per-frame dt would understate the elapsed time by that
     * factor, and since alpha = 1/(1 + tau/dt), a dt that is 3x too small
     * makes the filter roughly 3x heavier than intended. The rectangle would
     * then trail a moving hand, leaving the hand off-centre in the crop and
     * pushing the fingertips towards the crop border, which is where the
     * landmark model is least reliable. */
    static uint32_t s_prev_detect_start = 0;
    static bool     s_have_prev_detect  = false;

    float det_dt_s = FILT_NOMINAL_DT_S * (float)PALM_DETECT_INTERVAL;
    if (s_have_prev_detect)
    {
        const uint32_t det_dt_ms = TimeCounter_CountValueConvertToMs(s_prev_detect_start, (uint32_t)pipeline_start);
        if ((det_dt_ms >= 1u) && (det_dt_ms <= 1500u))
        {
            det_dt_s = (float)det_dt_ms * 0.001f;
        }
    }

    palm_rotated_hand_t hands[MAX_LANDMARK_HANDS];
    bool hand_valid[MAX_LANDMARK_HANDS] = { false };

    /* Decide whether the palm detector runs this frame. It always runs while
     * no hand is held, so acquiring a hand is never delayed; the interval only
     * applies to keeping one that is already in hand. */
    static uint32_t s_since_detect = PALM_DETECT_INTERVAL;

#if (MP_ROI_TRACKING == 1)
    /* With tracking closed, a tracked hand no longer needs the detector at all
     * — it supplies its own crop. The detector's remaining job is finding
     * hands that are not being tracked, which is why the interval can be much
     * longer than it had to be when every crop came from it.
     *
     * Two cases still demand it immediately, so that lengthening the interval
     * costs nothing where it would be felt:
     *   - nothing is tracked, i.e. acquiring the first hand;
     *   - a slot is holding a hand whose track just dropped, and is otherwise
     *     about to fall back on a stale palm rectangle.
     * An empty slot is neither, and only wants the periodic scan for a hand
     * that may have entered the frame. */
    bool detect_now = true;
    bool have_room  = true;

    {
        int  tracked = 0;
        bool stale   = false;

        for (int s = 0; s < MAX_LANDMARK_HANDS; s++)
        {
            if (g_hand[s].tracking) tracked++;
            else if (g_hand[s].has_rect) stale = true;
        }

        detect_now = (tracked == 0) || stale;

        /* With every slot tracking, the periodic scan has nothing left to
         * find, so it is pure cost. Divergence is still caught: a track that
         * leaves the MP_TRACK_*_VS_REF band clears its tracking flag, which
         * frees a slot and brings the detector straight back. */
        have_room = (tracked < MAX_LANDMARK_HANDS);
    }

    const bool run_detector =
        detect_now ||
        (have_room && ((s_since_detect + 1u) >= PALM_DETECT_INTERVAL));
#else
    int live_rects = 0;
    for (int s = 0; s < MAX_LANDMARK_HANDS; s++)
    {
        if (g_hand[s].has_rect) live_rects++;
    }

    const bool run_detector = (live_rects == 0) ||
                              ((s_since_detect + 1u) >= PALM_DETECT_INTERVAL);
#endif

#if (MP_ROI_TRACKING == 1)
    /* A slot that produced valid landmarks last frame supplies its own crop
     * for this one, whether or not the detector runs. The rectangle comes from
     * twelve landmarks the fingertip jumps cannot reach, so it is both fresher
     * and steadier than the detector's — measured at half the frame-to-frame
     * spread of the palm-derived one. */
    for (int s = 0; s < MAX_LANDMARK_HANDS; s++)
    {
        if (g_hand[s].tracking)
        {
            hands[s]      = g_hand[s].track_roi;
            hand_valid[s] = true;
        }
    }
#endif

    if (run_detector)
    {
        s_since_detect      = 0u;
        s_prev_detect_start = (uint32_t)pipeline_start;
        s_have_prev_detect  = true;

        /* Copy the AI input image to tensor arena */
        memcpy((void*)mera_input_ptr(), (const void*)model_buffer_int8, IMAGE_DATA_SIZE);

#if (BSP_CFG_DCACHE_ENABLED == 1)
        /* Write the freshly copied input back to RAM before the NPU reads it.
         *
         * The NPU is a bus master and does not snoop the CPU's write-back
         * D-cache, so without this it reads whatever RAM held before the
         * memcpy. The copy is 110 KB, far larger than the L1 D-cache, so most
         * lines have already been evicted naturally -- but the last ~16-32 KB
         * written, i.e. the bottom of the image, would still be stale. That is
         * why the symptom is subtle: detection degrades in the lower part of
         * the frame only.
         *
         * Only the input tensor needs cleaning; nothing else in this arena was
         * written by the CPU this frame. RunModel() already handles its two
         * internal hand-offs (NPU1 output -> CPU, CPU -> NPU2) on its own.
         *
         * This fix originated here and was carried back to the landmark-only
         * project, which had the same bug. */
        SCB_CleanDCache_by_Addr((uint32_t*)mera_input_ptr(), (int32_t)IMAGE_DATA_SIZE);
#endif

        /* Execute AI inference */
        mera_invoke();

#if (BSP_CFG_DCACHE_ENABLED == 1)
        /* Invalidate only NPU arenas so CPU sees fresh output (don't invalidate entire cache in RTOS!) */
        SCB_InvalidateDCache_by_Addr((uint32_t*)sub_0000_arena, (int32_t)kArenaSize_sub_0000);
        SCB_InvalidateDCache_by_Addr((uint32_t*)sub_0002_arena, (int32_t)kArenaSize_sub_0002);
#endif

        /* Get output pointers */
        int8_t* out_scores = mera_output_scores_ptr();   /* [2016]   */
        int8_t* out_boxes  = mera_output_boxes_ptr();     /* [2016,4] */
        int8_t* out_kp0    = mera_output_kp0_ptr();       /* [2016,2] */
        int8_t* out_kp2    = mera_output_kp2_ptr();       /* [2016,2] */

        /* Dequantize all outputs to float */
        palm_dequantize_i8(out_scores, PALM_NUM_ANCHORS,     PALM_SCORES_SCALE, PALM_SCORES_ZP, dequant_scores);
        palm_dequantize_i8(out_boxes,  PALM_NUM_ANCHORS * 4, PALM_BOXES_SCALE,  PALM_BOXES_ZP,  dequant_boxes);
        palm_dequantize_i8(out_kp0,    PALM_NUM_ANCHORS * 2, PALM_KP_SCALE,     PALM_KP_ZP,     dequant_kp0);
        palm_dequantize_i8(out_kp2,    PALM_NUM_ANCHORS * 2, PALM_KP_SCALE,     PALM_KP_ZP,     dequant_kp2);

        /* Run NMS post-processing */
        palm_postprocess_config_t pp_cfg;
        pp_cfg.score_thresh = 0.45f;
        pp_cfg.iou_thresh   = 0.30f;
        pp_cfg.max_dets     = MAX_LANDMARK_HANDS;

        const int num_dets = palm_postprocess_pre_nms(
            dequant_boxes, dequant_scores, dequant_kp0, dequant_kp2,
            PALM_NUM_ANCHORS, &pp_cfg, palm_results, MAX_LANDMARK_HANDS
        );

        const int   resized_w = g_palm_preprocess_meta.resized_w;   /* 192 */
        const int   start_y   = g_palm_preprocess_meta.start_y;     /*  24 */
        const float scale_lm  = (float)resized_w / CAM_W_F;         /* 192/640 = 0.3 */

        /* ---------------------------------------------------------------
         * 1. Rotated hand crop rectangle per detection, from the palm
         *    keypoints — the original computation, unchanged:
         *      a. Remap kp0/kp2 from model [0,1] → camera pixel coords
         *      b. Center = midpoint of kp0 and kp2
         *      c. Size from kp0↔kp2 distance: rr_size = sqrt(dist²×2.8)×2
         *      d. Angle = atan2(dy, dx) − π/2
         *      e. Normalize for landmark_preprocess
         * --------------------------------------------------------------- */
        palm_rotated_hand_t det_rect[MAX_LANDMARK_HANDS];
        int num_rects = 0;

        for (int i = 0; (i < num_dets) && (num_rects < MAX_LANDMARK_HANDS); i++)
        {
            const float kp0_xp = palm_results[i].kp0_x * (float)AI_INPUT_IMAGE_WIDTH / scale_lm;
            const float kp0_yp = ((palm_results[i].kp0_y * (float)AI_INPUT_IMAGE_HEIGHT) - (float)start_y) / scale_lm;
            const float kp2_xp = palm_results[i].kp2_x * (float)AI_INPUT_IMAGE_WIDTH / scale_lm;
            const float kp2_yp = ((palm_results[i].kp2_y * (float)AI_INPUT_IMAGE_HEIGHT) - (float)start_y) / scale_lm;

            const float dx_p = kp2_xp - kp0_xp;
            const float dy_p = kp2_yp - kp0_yp;
            const float dist = sqrtf((dx_p * dx_p) + (dy_p * dy_p));

            det_rect[num_rects].sqn_rr_size     = sqrtf(dist * dist * 2.8f) * 2.0f / CAM_REF_F;
            det_rect[num_rects].rotation        = atan2f(dy_p, dx_p) - 1.5707963f; /* − π/2 */
            det_rect[num_rects].sqn_rr_center_x = (kp0_xp + kp2_xp) * 0.5f / CAM_W_F;
            det_rect[num_rects].sqn_rr_center_y = (kp0_yp + kp2_yp) * 0.5f / CAM_H_F;
            num_rects++;
        }

        /* ---------------------------------------------------------------
         * 2. Assign detections to slots by nearest previous position, so a
         *    hand keeps the same slot — and therefore the same filter history
         *    — across detector runs. Without this the two hands can swap slots
         *    whenever NMS reorders them, and each would inherit the other's
         *    history.
         * --------------------------------------------------------------- */
        for (int d = 0; d < num_rects; d++)
        {
#if (MP_ROI_TRACKING == 1)
            /* The detector still sees hands that are already being tracked. A
             * detection landing on one of them is that same hand, not a new
             * one, and must not be allowed to claim a second slot. */
            bool duplicate = false;

            for (int s = 0; s < MAX_LANDMARK_HANDS; s++)
            {
                if (!g_hand[s].tracking || !hand_valid[s]) continue;

                const float ddx = det_rect[d].sqn_rr_center_x - hands[s].sqn_rr_center_x;
                const float ddy = det_rect[d].sqn_rr_center_y - hands[s].sqn_rr_center_y;

                if (((ddx * ddx) + (ddy * ddy)) <
                    (SLOT_MATCH_MAX_DIST * SLOT_MATCH_MAX_DIST))
                {
                    /* Refresh the clamp reference while the detector and the
                     * track still agree on where this hand is. */
                    g_hand[s].ref_size = det_rect[d].sqn_rr_size;
                    duplicate = true;
                    break;
                }
            }

            if (duplicate) continue;
#endif

            int   best      = -1;
            float best_dist = SLOT_MATCH_MAX_DIST * SLOT_MATCH_MAX_DIST;

            for (int s = 0; s < MAX_LANDMARK_HANDS; s++)
            {
                if (!g_hand[s].has_rect || hand_valid[s]) continue;

                const float ddx = det_rect[d].sqn_rr_center_x - g_hand[s].prev_cx;
                const float ddy = det_rect[d].sqn_rr_center_y - g_hand[s].prev_cy;
                const float dd  = (ddx * ddx) + (ddy * ddy);

                if (dd < best_dist)
                {
                    best      = s;
                    best_dist = dd;
                }
            }

            if (best < 0)
            {
                /* New hand: take a free slot and start its history clean. */
                for (int s = 0; s < MAX_LANDMARK_HANDS; s++)
                {
                    if (!hand_valid[s] && !g_hand[s].has_rect) { best = s; break; }
                }
                if (best < 0)
                {
                    for (int s = 0; s < MAX_LANDMARK_HANDS; s++)
                    {
                        if (!hand_valid[s]) { best = s; break; }
                    }
                }
                if (best < 0) break;   /* no slot left */

                hand_slot_reset(best);
            }

            hands[best]      = det_rect[d];
            hand_valid[best] = true;
        }

        /* Slots the detector found nothing for lose their history — including
         * their cached rectangle, so nothing stale is reused for a hand that
         * is no longer there. A tracking slot is exempt: the detector missing
         * it for one run is not evidence the hand has gone, the landmark
         * confidence check below is. */
        for (int s = 0; s < MAX_LANDMARK_HANDS; s++)
        {
            if (!hand_valid[s]) hand_slot_reset(s);
        }

        /* ---------------------------------------------------------------
         * 3. Smooth each rectangle and record it for the frames in between.
         *    The filters are only stepped here, on frames where a real
         *    measurement arrived, so the skipped frames do not feed them a
         *    run of identical samples and flatten their speed estimate.
         * --------------------------------------------------------------- */
        for (int s = 0; s < MAX_LANDMARK_HANDS; s++)
        {
            if (!hand_valid[s]) continue;

#if (MP_ROI_TRACKING == 1)
            /* A tracking slot already has its crop for this frame and did not
             * take a detection, so there is nothing here to smooth or record.
             * Its reference size is refreshed instead, from whichever detection
             * (if any) matched it, so the clamp stays anchored to something the
             * detector recently agreed with. */
            if (g_hand[s].tracking) continue;
#endif

            /* Raw centre for the next detector run's slot matching, taken
             * before the filters move it. */
            g_hand[s].prev_cx = hands[s].sqn_rr_center_x;
            g_hand[s].prev_cy = hands[s].sqn_rr_center_y;

            /* Smoothing before the crop is taken removes the palm keypoint
             * noise at the source, so the landmark model gets a stable image
             * to work from. Not a feedback loop: the input is a detector
             * measurement, not the filter's own previous output.
             *
             * det_dt_s, not dt_s: this steps once per detector run. */
            hand_rect_smooth(&g_hand[s], &hands[s], det_dt_s);

            g_hand[s].used_rect = hands[s];
            g_hand[s].has_rect  = true;
            g_hand[s].ref_size  = hands[s].sqn_rr_size;
        }
    }
    else
    {
        s_since_detect++;

        /* Reuse the rectangle from the most recent detector run, for any slot
         * not already supplying its own. */
        for (int s = 0; s < MAX_LANDMARK_HANDS; s++)
        {
            if (!hand_valid[s] && g_hand[s].has_rect)
            {
                hands[s]      = g_hand[s].used_rect;
                hand_valid[s] = true;
            }
        }
    }

    /* Clear every display slot; repopulated below for whichever slots produce
     * a valid landmark result this frame. */
    for (uint16_t i = 0; i < AI_MAX_DETECTION_NUM; i++)
    {
        update_detection_result(i, 0, 0, 0, 0);
        landmark_result_t empty_lm;
        memset(&empty_lm, 0, sizeof(empty_lm));
        update_landmark_result(i, &empty_lm);
    }

    /* ---------------------------------------------------------------
     * 4. Run the landmark model for every slot that has a rectangle, smooth
     *    the points, and draw. This happens every frame, detector or not.
     * --------------------------------------------------------------- */
    landmark_preprocess_config_t lm_cfg;
    lm_cfg.img_w = CAM_VGA_WIDTH;
    lm_cfg.img_h = CAM_VGA_HEIGHT;
    lm_cfg.input_is_bgr = 0;
    lm_cfg.input_is_rgb565 = 1;

    for (int hi = 0; hi < MAX_LANDMARK_HANDS; hi++)
    {
        if (!hand_valid[hi]) continue;

        /* Too much of the crop is off-frame for the result to mean anything.
         * Skip the inference entirely and leave this slot's display cleared,
         * so the overlay disappears as the hand leaves rather than collapsing
         * against the border. Dropping has_rect as well hands the slot back to
         * the detector, which re-acquires the moment enough of the hand is
         * inside again. */
        if (crop_visible_fraction(&hands[hi]) < MP_EDGE_MIN_VISIBLE)
        {
            hand_slot_reset(hi);
            continue;
        }

        landmark_preprocess_meta_t lm_meta;

        /* 1. Preprocess: crop rotated hand from camera RGB565, resize to 224×224 INT8 */
        const int ret = landmark_preprocess(
            (const void*)camera_capture_image_rgb565,
            &lm_cfg, &hands[hi],
            landmark_input_buf, &lm_meta
        );
        if (ret < 0)
        {
#if (MP_ROI_TRACKING == 1)
            g_hand[hi].tracking = false;
#endif
            continue;
        }

        /* 2. Copy to landmark model input tensor */
        memcpy((void*)landmark_input_ptr(), (const void*)landmark_input_buf, LANDMARK_INPUT_SIZE);

#if (BSP_CFG_DCACHE_ENABLED == 1)
        SCB_CleanDCache_by_Addr((uint32_t*)sub_0000__landmark_arena, (int32_t)kArenaSize_sub_0000__landmark);
#endif

        /* 3. Run landmark inference */
        landmark_invoke();

#if (BSP_CFG_DCACHE_ENABLED == 1)
        SCB_InvalidateDCache_by_Addr((uint32_t*)sub_0000__landmark_arena, (int32_t)kArenaSize_sub_0000__landmark);
#endif

        /* 4. Postprocess: dequantize + threshold + remap to camera coords */
        landmark_result_t lm_result;
        const int valid = landmark_postprocess(
            landmark_output_xyz_ptr(),
            landmark_output_score_ptr(),
            landmark_output_hand_ptr(),
            &lm_meta,
            0.5f,   /* score threshold */
            &lm_result
        );

        if (valid > 0)
        {
#if (MP_ROI_MEASURE_ONLY == 1)
            /* Measure the upstream rule against the rectangle that was
             * actually used for this frame's crop. Taken from the RAW landmark
             * output, before the display smoothing below, because that is what
             * a closed loop would have to work from.
             *
             * What the numbers need to show before this can drive anything:
             *   ratio  — MP size as a percentage of the size in use. Any stable
             *            value is fine, including one far from 100%; what
             *            matters is that it does not wander, since a drifting
             *            ratio is what compounds in a loop.
             *   dcx/dcy — centre offset in px. Expected to be non-zero and
             *            roughly constant: the shift_y term deliberately moves
             *            the rect off the landmark centroid.
             *   drot   — rotation difference in degrees, expected near 0.
             */
            {
                static uint32_t s_mp_log = 0;

                if ((s_mp_log % MP_ROI_LOG_STRIDE) == 0)
                {
                    palm_rotated_hand_t mp;
                    mp_roi_from_landmarks(&lm_result, &mp);

                    const int used_px  = (int)lrintf(hands[hi].sqn_rr_size * CAM_REF_F);
                    const int mp_px    = (int)lrintf(mp.sqn_rr_size * CAM_REF_F);
                    const int ratio    = (used_px != 0) ? (int)lrintf(100.0f * (float)mp_px / (float)used_px) : -1;

                    const int dcx = (int)lrintf((mp.sqn_rr_center_x - hands[hi].sqn_rr_center_x) * CAM_W_F);
                    const int dcy = (int)lrintf((mp.sqn_rr_center_y - hands[hi].sqn_rr_center_y) * CAM_H_F);

                    int drot = (int)lrintf((mp.rotation - hands[hi].rotation) * 180.0f / 3.14159265f);
                    while (drot >  180) drot -= 360;
                    while (drot < -180) drot += 360;

                    sprintf(sprintf_buffer,
                            "[MPROI] h%d used=%4d mp=%4d ratio=%4d%% dcx=%4d dcy=%4d drot=%4d\r\n",
                            hi, used_px, mp_px, ratio, dcx, dcy, drot);
                    print_to_console(sprintf_buffer);
                }
                s_mp_log++;
            }
#endif

#if (MP_ROI_TRACKING == 1)
            /* 5. Close the loop: this hand supplies its own crop next frame.
             *
             * Derived from the RAW landmarks, before the display smoothing
             * below — the crop should follow the model, not a filtered version
             * of it, and keeping the filter out of the loop keeps the loop
             * first-order and its measured gain meaningful.
             *
             * The clamp is the guarantee that does not depend on that gain
             * being right: the tracked size cannot leave a band around the
             * last size the detector produced for this slot, so no sequence of
             * frames can walk the crop away the way the two earlier attempts
             * did. Landing on the clamp is itself the signal that the loop is
             * not behaving, so the track is dropped rather than pinned there. */
            {
                palm_rotated_hand_t roi;
                mp_roi_from_landmarks(&lm_result, &roi);

                const float ref = (g_hand[hi].ref_size > 0.0f) ? g_hand[hi].ref_size
                                                               : hands[hi].sqn_rr_size;

                if ((roi.sqn_rr_size >= (ref * MP_TRACK_MIN_VS_REF)) &&
                    (roi.sqn_rr_size <= (ref * MP_TRACK_MAX_VS_REF)))
                {
                    g_hand[hi].track_roi = roi;
                    g_hand[hi].tracking  = true;

                    /* Keep the slot-matching centre current, so a detector run
                     * still recognises this hand while it is being tracked. */
                    g_hand[hi].prev_cx = roi.sqn_rr_center_x;
                    g_hand[hi].prev_cy = roi.sqn_rr_center_y;
                }
                else
                {
                    g_hand[hi].tracking = false;
                }
            }
#endif

            /* 6. Smooth the 21 points, then take the on-screen box from those
             *    same smoothed points so box and skeleton always agree. */
            landmark_points_smooth(&g_hand[hi], &lm_result, dt_s);
            update_landmark_result((uint16_t)hi, &lm_result);

            float box_x1, box_y1, box_w, box_h;
            landmark_bounding_box(&lm_result, &box_x1, &box_y1, &box_w, &box_h);
            update_detection_result((uint16_t)hi,
                                    (signed short)box_x1, (signed short)box_y1,
                                    (signed short)box_w,  (signed short)box_h);
        }
#if (MP_ROI_TRACKING == 1)
        else
        {
            /* Confidence below threshold: the hand may be gone or badly
             * cropped. Hand the slot back to the detector rather than tracking
             * from landmarks that were not trusted enough to display. */
            g_hand[hi].tracking = false;
        }
#endif
    }

    /* Record full pipeline time (pre + palm inference + post + landmark),
     * averaged over PIPE_TIME_AVG_WINDOW frames — see that macro for why. */
    {
        static uint32_t s_hist[PIPE_TIME_AVG_WINDOW] = { 0 };
        static uint32_t s_idx   = 0;
        static uint32_t s_sum   = 0;
        static uint32_t s_count = 0;

        const uint32_t elapsed_ms = TimeCounter_CountValueConvertToMs(pipeline_start, TimeCounter_CurrentCountGet());

        s_sum -= s_hist[s_idx];
        s_hist[s_idx] = elapsed_ms;
        s_sum += elapsed_ms;
        s_idx  = (s_idx + 1u) % PIPE_TIME_AVG_WINDOW;

        if (s_count < PIPE_TIME_AVG_WINDOW)
        {
            s_count++;
        }

        application_processing_time.ai_inference_time_ms = s_sum / s_count;
    }

    return true;
}
