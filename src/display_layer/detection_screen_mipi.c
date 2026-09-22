/*
* Copyright (c) 2020 - 2025 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**********************************************************************************************************************
 * File Name    : detection_screen_mipi.c
 * Version      : .
 * Description  : The palm detection screen display on mipi lcd.
 *********************************************************************************************************************/
/***************************************************************************************************************************
 * Includes   <System Includes> , "Project Includes"
 ***************************************************************************************************************************/

#include "hal_data.h"
#include <stdio.h>

#include "common_util.h"

#include "camera_layer.h"
#include "display_layer.h"
#include "bg_font_18_full.h"

#include "time_counter.h"
#include "./ai_application/palm_detection/landmark_display.h"
#include "./ai_application/palm_detection/gesture_classify.h"

#include "display_layer_config.h"

#include "application_config.h"
#include "ai_application_config.h"

/***************************************************************************************************************************
 * Macro definitions
 * Refer to the application note for the physical definition of these values
 ***************************************************************************************************************************/

/* Area of the screen covered by the camera image, i.e. the only part that
 * display_camera_image() repaints every frame. Everything to the right of it
 * is the static sidebar, which is painted once at start-up and thereafter only
 * touched by the small text-clearing writes in print_inf_time_and_detections()
 * and print_gesture_info(). 640x480 camera x 1.25 = 800x600 on a 1024x600 panel. */
#define CAMERA_DISPLAY_WIDTH   ((int)(CAMERA_CAPTURE_IMAGE_WIDTH  * CAMERA_IMAGE_SCALING))
#define CAMERA_DISPLAY_HEIGHT  ((int)(CAMERA_CAPTURE_IMAGE_HEIGHT * CAMERA_IMAGE_SCALING))

/* Draw a box around each hand as well as the skeleton.
 *
 * Off by default here, which is what this project has shown since the gesture
 * work: the skeleton already marks the hand, and a second outline competes
 * with the Left/Right label sitting next to the wrist.
 *
 * Worth reconsidering now that the box means something different. It used to
 * be the palm detector's own rectangle, which is larger than the hand and
 * disagrees with the drawn skeleton because the two come from different
 * models. MainLoop_obj.cc now derives it from the same smoothed 21 points the
 * skeleton is drawn from, so it fits the hand and cannot disagree with it.
 * Set to 1 to turn it on. */
#define SHOW_LANDMARK_BOUNDING_BOX  (0)

/* ---- Handedness decision -------------------------------------------------
 * The landmark model's handedness output is only weakly determined when the
 * back of the hand faces the camera: a palm-forward left hand and a
 * back-forward right hand project to almost the same silhouette, so the model
 * has little to separate them and its output sits near 0.5 and wanders.
 * Judging each frame on its own — even with a threshold band — lets that
 * wander through as a flickering label.
 *
 * So accumulate evidence over time instead of thresholding one sample.
 * Raise EVIDENCE_LIMIT to make an established decision harder to overturn;
 * raise DECISION_DEADZONE to demand more confidence before labelling at all.
 * At ~26 inferences/s, a confident hand (handedness ~0.9) commits in about
 * 3 frames and saturates in about 10.
 * ------------------------------------------------------------------------ */
#define HANDEDNESS_EVIDENCE_LIMIT     (4.0f)
#define HANDEDNESS_DECISION_DEADZONE  (1.0f)

#define HANDEDNESS_UNDECIDED          (0U)
#define HANDEDNESS_LEFT               (1U)
#define HANDEDNESS_RIGHT              (2U)

/***************************************************************************************************************************
 * Typedef definitions
 ***************************************************************************************************************************/

/***************************************************************************************************************************
 * Imported global variables and functions (from other files)
 ***************************************************************************************************************************/

/***************************************************************************************************************************
 * Exported global variables and functions (to be accessed by other files)
 ***************************************************************************************************************************/

void  do_detection_screen(bool ai_result_new);
uint8_t detection_count = 0;

uint8_t exe_count_print_static_text = 0;

/***************************************************************************************************************************
 * Private global variables and functions
 ***************************************************************************************************************************/
#if (SHOW_LANDMARK_BOUNDING_BOX == 1)
/* Scratch for the box corners, written by calculate_and_draw_bounding_box()
 * and read by draw_bounding_box(). Nothing else touches them, so they follow
 * the same switch rather than sitting unused. */
static d2_point top_left_x[AI_MAX_DETECTION_NUM];
static d2_point top_left_y[AI_MAX_DETECTION_NUM];
static d2_point bottom_right_x[AI_MAX_DETECTION_NUM];
static d2_point bottom_right_y[AI_MAX_DETECTION_NUM];
#endif

static void print_static_text (void);
static void draw_landmark_points(uint8_t det_idx);
#if (SHOW_LANDMARK_BOUNDING_BOX == 1)
static void draw_bounding_box(uint8_t i);
static void calculate_and_draw_bounding_box(uint8_t i);
#endif
static void display_camera_image(void);
static void print_gesture_info(void);
static void update_handedness(uint8_t det_idx);
static void update_gesture(uint8_t det_idx);

/* Persistent gesture state per detection slot (written by update_gesture, read by print_gesture_info) */
static gesture_t g_last_gesture[AI_MAX_DETECTION_NUM] = {0};

/* Persistent handedness state per detection slot. Written by update_handedness
 * once per inference result, read by draw_landmark_points every frame. */
typedef struct {
    float   evidence;   /* >0 favours Left, <0 favours Right (as displayed) */
    uint8_t decision;   /* HANDEDNESS_UNDECIDED / _LEFT / _RIGHT */
} handedness_state_t;

static handedness_state_t g_handedness[AI_MAX_DETECTION_NUM] = {0};

/*********************************************************************************************************************
 *  display_camera_image function
 *  			 This function selects the rotated camera image buffer in QVGA mode 240x320 format and
 *  			 displays in VGA 480x640 format at the center of the mipi lcd.
 *  @param   	None
 *  @retval     None.
***********************************************************************************************************************/
static void display_camera_image(void)
{
#if (BSP_CFG_DCACHE_ENABLED == 1)
    // Clean cache data for camera capture image buffer because this buffer will be accessed by DRW hardware
    SCB_CleanDCache_by_Addr(&camera_capture_image_rgb565[0], (int32_t)(camera_capture_image_rgb565_size));
#endif

	/* Specify camera input. */
	d2_setblitsrc(d2_handle, (void *)&camera_capture_image_rgb565[0], CAMERA_CAPTURE_IMAGE_WIDTH, CAMERA_CAPTURE_IMAGE_WIDTH, CAMERA_CAPTURE_IMAGE_HEIGHT, d2_mode_rgb565);

	/* display as VGA 640x480 on mipi lcd */
	d2_blitcopy(d2_handle,
	            (d2_s32) CAMERA_CAPTURE_IMAGE_WIDTH, (d2_s32) CAMERA_CAPTURE_IMAGE_HEIGHT, // Source width/height
	            (d2_blitpos) 0, (d2_blitpos) 0,                                          // Source position
	            (d2_width) ((uint32_t)(CAMERA_CAPTURE_IMAGE_WIDTH * CAMERA_IMAGE_SCALING) << 4), (d2_width) ((uint32_t)(CAMERA_CAPTURE_IMAGE_HEIGHT * CAMERA_IMAGE_SCALING) << 4),   // Destination size width/height
	            (d2_point) (0 << 4), (d2_point) (0 << 4), // Destination offset position
	            d2_tm_filter);
}

#if (SHOW_LANDMARK_BOUNDING_BOX == 1)
/*********************************************************************************************************************
 *  draw_bounding_box function
 *  			 This function picks the index of the detection result which has a bounding box and
 *  			 uses DRW to render a red bounding box on the mipi lcd.
 *  @param[IN]   i: index of the detection result
 *  @retval     None
***********************************************************************************************************************/
static void draw_bounding_box(uint8_t i)
{
	d2_setcolor(d2_handle, 0, AI_INFERENCE_RESULT_BOUNDING_BOX_COLOR);

	d2_renderline(d2_handle, (d2_point) ((top_left_x[i]) << 4), (d2_point) ((top_left_y[i])<< 4), (d2_point) ((bottom_right_x[i]) << 4), (d2_point) ((top_left_y[i]) << 4), (d2_point) (2 << 4), 0);
	d2_renderline(d2_handle, (d2_point) ((bottom_right_x[i]) << 4), (d2_point) ((top_left_y[i]) << 4), (d2_point) ((bottom_right_x[i]) << 4), (d2_point) ((bottom_right_y[i]) << 4), (d2_point) (2 << 4), 0);
	d2_renderline(d2_handle, (d2_point) ((bottom_right_x[i]) << 4), (d2_point) ((bottom_right_y[i]) << 4), (d2_point) ((top_left_x[i]) << 4), (d2_point) ((bottom_right_y[i]) << 4), (d2_point) (2 << 4), 0);
	d2_renderline(d2_handle, (d2_point) ((top_left_x[i]) << 4), (d2_point) ((bottom_right_y[i]) << 4), (d2_point) ((top_left_x[i]) << 4), (d2_point) ((top_left_y[i]) << 4), (d2_point) (2 << 4), 0);
}
#endif /* SHOW_LANDMARK_BOUNDING_BOX */

/*********************************************************************************************************************
 *  draw_landmark_points function
 *  Draws the 21 hand landmark keypoints as small green circles on the LCD.
 *  Landmark coordinates are in camera space (640x480) and are scaled to display.
 *  @param[IN]   det_idx: detection slot index
 *  @retval      None
***********************************************************************************************************************/
static void draw_landmark_points(uint8_t det_idx)
{
    const landmark_result_t* lm = &g_landmark_results[det_idx];
    if (lm->num_points <= 0 || lm->hand_score < 0.5f) return;

    /* Red color for keypoints */
    d2_setcolor(d2_handle, 0, 0x00FF0000);

    for (int k = 0; k < lm->num_points; k++)
    {
        /* Landmark points are in camera pixel coords (640x480).
         * Scale to display coords using CAMERA_IMAGE_SCALING. */
        d2_point dx = (d2_point)((float)lm->pts[k].x * CAMERA_IMAGE_SCALING);
        d2_point dy = (d2_point)((float)lm->pts[k].y * CAMERA_IMAGE_SCALING);

        /* Draw a filled circle (radius = 6 display pixels) */
        d2_rendercircle(d2_handle,
            (d2_point)(dx << 4), (d2_point)(dy << 4),
            (d2_width)(6 << 4), (d2_width)(0));
    }

    /* Draw skeleton lines connecting key joints (optional but nice visual) */
    /* Hand skeleton connections: wrist→thumb, wrist→index, etc. */
    static const int skeleton[][2] = {
        {0,1},{1,2},{2,3},{3,4},         /* thumb */
        {0,5},{5,6},{6,7},{7,8},         /* index */
        {0,9},{9,10},{10,11},{11,12},    /* middle */
        {0,13},{13,14},{14,15},{15,16},  /* ring */
        {0,17},{17,18},{18,19},{19,20},  /* pinky */
        {5,9},{9,13},{13,17}             /* palm cross */
    };

    d2_setcolor(d2_handle, 0, 0x000000FF);
    for (int s = 0; s < 23; s++)
    {
        int a = skeleton[s][0];
        int b = skeleton[s][1];
        d2_point ax = (d2_point)((float)lm->pts[a].x * CAMERA_IMAGE_SCALING);
        d2_point ay = (d2_point)((float)lm->pts[a].y * CAMERA_IMAGE_SCALING);
        d2_point bx = (d2_point)((float)lm->pts[b].x * CAMERA_IMAGE_SCALING);
        d2_point by = (d2_point)((float)lm->pts[b].y * CAMERA_IMAGE_SCALING);

        d2_renderline(d2_handle,
            (d2_point)(ax << 4), (d2_point)(ay << 4),
            (d2_point)(bx << 4), (d2_point)(by << 4),
            (d2_point)(1 << 4), 0);
    }

    /* Draw the "Left"/"Right" label near the wrist (point 0). The decision
     * itself is made in update_handedness(), once per inference result. */
    {
        const uint8_t decision = g_handedness[det_idx].decision;

        if (HANDEDNESS_UNDECIDED != decision)
        {
            d2_point wx = (d2_point)((float)lm->pts[0].x * CAMERA_IMAGE_SCALING);
            d2_point wy = (d2_point)((float)lm->pts[0].y * CAMERA_IMAGE_SCALING);
            char *lr_str = (HANDEDNESS_LEFT == decision) ? (char*)"Left" : (char*)"Right";
            print_bg_font_18(d2_handle, (int16_t)(wx - 15), (int16_t)(wy + 15),
                             DISPLAY_FONT_SCALING, lr_str);
        }
    }
}

/*********************************************************************************************************************
 *  update_handedness function
 *  Decides whether a detected hand is shown as "Left" or "Right".
 *
 *  Must be called exactly once per inference result, not once per display
 *  frame: the display thread runs roughly twice as fast as the AI thread, so
 *  calling it per frame would count the same measurement twice.
 *
 *  @param[IN]   det_idx: detection slot index
 *  @retval      None
***********************************************************************************************************************/
static void update_handedness(uint8_t det_idx)
{
    const landmark_result_t * lm = &g_landmark_results[det_idx];
    handedness_state_t      * st = &g_handedness[det_idx];

    /* Slot empty. Forget everything, so that a different hand appearing in
     * this slot later is judged from scratch rather than inheriting a stale
     * decision (the palm tracking may reorder slots between frames). */
    if ((lm->num_points <= 0) || (lm->hand_score < 0.5f))
    {
        st->evidence = 0.0f;
        st->decision = HANDEDNESS_UNDECIDED;
        return;
    }

    /* handedness is a sigmoid in [0,1]; 0.5 means the model cannot tell.
     * The camera image is mirrored, so the model's "right" is our "Left",
     * which is why positive evidence maps to HANDEDNESS_LEFT below.
     *
     * Each frame contributes its distance from 0.5, so a confident frame
     * moves the accumulator a lot and an uncertain one barely moves it. */
    st->evidence += (lm->handedness - 0.5f);

    if (st->evidence > HANDEDNESS_EVIDENCE_LIMIT)
    {
        st->evidence = HANDEDNESS_EVIDENCE_LIMIT;
    }
    else if (st->evidence < -HANDEDNESS_EVIDENCE_LIMIT)
    {
        st->evidence = -HANDEDNESS_EVIDENCE_LIMIT;
    }

    /* Commit only outside the dead zone; inside it the previous decision
     * stands. Combined with the saturation above this gives an asymmetric
     * hysteresis: reaching a decision from scratch costs
     * HANDEDNESS_DECISION_DEADZONE of evidence, but overturning a saturated
     * one costs LIMIT + DEADZONE. */
    if (st->evidence > HANDEDNESS_DECISION_DEADZONE)
    {
        st->decision = HANDEDNESS_LEFT;
    }
    else if (st->evidence < -HANDEDNESS_DECISION_DEADZONE)
    {
        st->decision = HANDEDNESS_RIGHT;
    }
}

/*********************************************************************************************************************
 *  update_gesture function
 *  Runs the keypoint classifier for one detection slot. Same call-once-per-
 *  inference-result contract as update_handedness().
 *  @param[IN]   det_idx: detection slot index
 *  @retval      None
***********************************************************************************************************************/
static void update_gesture(uint8_t det_idx)
{
    const landmark_result_t * lm = &g_landmark_results[det_idx];

    /* Slot empty. Drop the remembered gesture, otherwise the sidebar keeps
     * reporting the last thing this hand did long after it left the frame. */
    if ((lm->num_points <= 0) || (lm->hand_score < 0.5f))
    {
        g_last_gesture[det_idx] = GESTURE_UNKNOWN;
        return;
    }

    /* A low-confidence frame deliberately keeps the previous label instead of
     * blanking it, so the text does not flicker in the middle of a gesture. */
    gesture_t g = gesture_classify(lm);
    if (g != GESTURE_UNKNOWN)
    {
        g_last_gesture[det_idx] = g;
    }
}

#if (SHOW_LANDMARK_BOUNDING_BOX == 1)
/*********************************************************************************************************************
 *  calculate_and_draw_bounding_box function
 *  This function takes the ai inference boundary box center of the image and scales it to the 480x640
 *  mipi lcd center area as a bounding box.
 *  @param[IN]   i: index of the detection result
 *  @retval     None
***********************************************************************************************************************/
static void calculate_and_draw_bounding_box(uint8_t i)
{
    detection_count++;

    /* m_x/y/w/h are now in camera pixel coords (640×480).
     * Just scale directly to display coords. */
    top_left_x[i]     = (d2_point)((float)g_ai_detection[i].m_x * CAMERA_IMAGE_SCALING);
    top_left_y[i]     = (d2_point)((float)g_ai_detection[i].m_y * CAMERA_IMAGE_SCALING);
    bottom_right_x[i] = (d2_point)((float)(g_ai_detection[i].m_x + g_ai_detection[i].m_w) * CAMERA_IMAGE_SCALING);
    bottom_right_y[i] = (d2_point)((float)(g_ai_detection[i].m_y + g_ai_detection[i].m_h) * CAMERA_IMAGE_SCALING);

    draw_bounding_box(i);
}
#endif /* SHOW_LANDMARK_BOUNDING_BOX */

/*********************************************************************************************************************
 *  print_static_text function
 *  This function prints the static text which does not change based on inference result.
 *  @param   	None
 *  @retval     None
***********************************************************************************************************************/
static void print_static_text(void)
{
	/* show model information */
	print_bg_font_18(d2_handle, 820,  50, DISPLAY_FONT_SCALING, (char*)"Model:");
	print_bg_font_18(d2_handle, 820,  90, DISPLAY_FONT_SCALING, (char*)"Landmark &");
	print_bg_font_18(d2_handle, 820, 120, DISPLAY_FONT_SCALING, (char*)"Gesture");

	/*show pipeline time in ms*/
	print_bg_font_18(d2_handle, 820, 200, DISPLAY_FONT_SCALING, (char*)"Pipeline");
	print_bg_font_18(d2_handle, 820, 230, DISPLAY_FONT_SCALING, (char*)"time:");

	/*print the number of palms detected */
	print_bg_font_18(d2_handle, 820, 350, DISPLAY_FONT_SCALING, (char*)"No of ");
	print_bg_font_18(d2_handle, 820, 380, DISPLAY_FONT_SCALING, (char*)"Hands:");

	/*print the gesture label */
	print_bg_font_18(d2_handle, 820, 480, DISPLAY_FONT_SCALING, (char*)"Gesture:");
}

/*********************************************************************************************************************
 *  print_inf_time_and_detections function
 *  This function prints the time used in the previously finished inference and the number of palms detected.
 *  @param   	None
 *  @retval     None
***********************************************************************************************************************/
static void print_inf_time_and_detections(void)
{
       /* The inference_time is acquired in MainLoop_obj.cc.
	 * This time does not include the pre and post processing routine.
	 * It is the time used for inference only.
	 */

	uint32_t time = (uint32_t)(application_processing_time.ai_inference_time_ms); // ms

	// Clear last draw
    print_bg_font_18(d2_handle, 820, 280, DISPLAY_FONT_SCALING,  "             ");
    print_bg_font_18(d2_handle, 820, 430, DISPLAY_FONT_SCALING,  "       ");

    // update string on display
	char time_str[8] = {'0', '0', '0', '0', ' ', 'm', 's', '\0'};
	time_str[0] += (char)(time / 1000);
	time_str[1] += (char)((time / 100) % 10);
	time_str[2] += (char)((time / 10) % 10);
	time_str[3] += (char)(time % 10);
	print_bg_font_18(d2_handle, 820, 280, DISPLAY_FONT_SCALING, (char*)time_str);

	char num_str[3] = {'0', '0', '\0'};
	num_str[0] += (char) (detection_count / 10);
	num_str[1] +=  (char) (detection_count % 10);
	print_bg_font_18(d2_handle, 820, 430, DISPLAY_FONT_SCALING, (char*)num_str);

}

/*********************************************************************************************************************
 *  print_gesture_info function
 *  Displays the current gesture name(s) in the sidebar below the hand count.
 *  @param       None
 *  @retval      None
***********************************************************************************************************************/
static void print_gesture_info(void)
{
    /* Clear previous gesture text */
    print_bg_font_18(d2_handle, 820, 520, DISPLAY_FONT_SCALING, "            ");

    /* Show the gesture of the first slot that currently holds a hand.
     *
     * Gating on the live landmark result, and not on g_last_gesture alone,
     * also covers start-up: g_last_gesture is zero-initialised and GESTURE_OPEN
     * happens to be 0, so before the first inference lands the array would
     * otherwise read as a genuine "Open" and print it with no hand in frame. */
    for (uint8_t i = 0; i < AI_MAX_DETECTION_NUM; i++)
    {
        const landmark_result_t * lm = &g_landmark_results[i];

        if ((lm->num_points <= 0) || (lm->hand_score < 0.5f))
        {
            continue;
        }

        /* GESTURE_UNKNOWN == GESTURE_COUNT, so this rejects it too. */
        if (g_last_gesture[i] < GESTURE_COUNT)
        {
            print_bg_font_18(d2_handle, 820, 520, DISPLAY_FONT_SCALING,
                             (char*)gesture_name(g_last_gesture[i]));
            break;
        }
    }
}

/*********************************************************************************************************************
 *  do_detection_screen function: display the camera image and palm detection result on the mipi lcd
 *  @param       None
 *  @retval      None
***********************************************************************************************************************/
void  do_detection_screen(bool ai_result_new)
{
    vision_ai_app_err_t vision_ai_status = VISION_AI_APP_SUCCESS;

    if(!(xEventGroupGetBits(g_ai_app_event) & DISPLAY_PAUSE))
    {
        /* Clear stale vsync flag, then wait for the NEXT vsync so the
         * GLCDC has finished scanning the front-buffer before we touch it. */
        xEventGroupClearBits(g_ai_app_event, GLCDC_VSYNC);
        xEventGroupWaitBits(g_ai_app_event, GLCDC_VSYNC, pdTRUE, pdTRUE, portMAX_DELAY);

        graphics_start_frame();

        /* show static information */
        display_camera_image();

        /* Print static test */
        if(exe_count_print_static_text < 2)
        {
            print_static_text();
            exe_count_print_static_text++;
        }

        /* Clip landmark drawing to the camera image.
         *
         * The landmark model extrapolates keypoints past the frame border when
         * a hand reaches the edge, so pts[].x can exceed the 640 px camera
         * width; scaled by 1.25 that lands beyond x=800, inside the sidebar.
         * Nothing repaints the sidebar, so such pixels would stay on screen
         * permanently and accumulate with every pass of the hand. Let DRW
         * discard the overflow instead. */
        d2_cliprect(d2_handle,
                    (d2_border) 0, (d2_border) 0,
                    (d2_border) (CAMERA_DISPLAY_WIDTH  - 1),
                    (d2_border) (CAMERA_DISPLAY_HEIGHT - 1));

        /* if a new inference has finished, update the detection result: bounding box and number of palms */
        if(ai_result_new)
        {
            detection_count = 0;
            for(uint8_t i = 0; i < AI_MAX_DETECTION_NUM; i++)
            {
                if((g_ai_detection[i].m_x != 0) && (g_ai_detection[i].m_y != 0))
                {
#if (SHOW_LANDMARK_BOUNDING_BOX == 1)
                    /* Also increments detection_count. */
                    calculate_and_draw_bounding_box(i);
#else
                    detection_count++;
#endif
                }

                /* Per-result state. Updated here and not in draw_landmark_points
                 * because that runs on every display frame, which would count
                 * the same inference result more than once. */
                update_handedness(i);
                update_gesture(i);

                /* Draw landmark keypoints if available */
                draw_landmark_points(i);
            }
        }
        else
        {
            /* Human movement is slower than the mipi lcd refresh rate. Keep the previous bounding box until a new inference is finished. */
            for(uint8_t i = 0; i < AI_MAX_DETECTION_NUM; i++)
            {
                draw_landmark_points(i);
            }
        }

        /* Restore full-screen clipping — the sidebar text below is drawn at
         * x=820 and would otherwise be clipped away entirely. */
        d2_cliprect(d2_handle,
                    (d2_border) 0, (d2_border) 0,
                    (d2_border) (DISPLAY_SCREEN_WIDTH  - 1),
                    (d2_border) (DISPLAY_SCREEN_HEIGHT - 1));

        print_inf_time_and_detections();
        print_gesture_info();

        /* Wait for previous frame rendering to finish, then finalize this frame and flip the buffers */
        graphics_end_frame();
    }
}
