import argparse
from typing import final

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'shot_identification'))
from shot_indetification.tennis_shot_identification_and_counts import ShotIdentification
import cv2
import tempfile
from main import process_video
from sympy.physics.units import current
from utils import read_video
from trackers import PlayerTracker
import numpy as np

def calculate_index(streamlit_coordinates, height, width, grid_len):
    def get_coordinates(streamlit_coordinates):
        if streamlit_coordinates:
            x = streamlit_coordinates["x"]
            y = streamlit_coordinates["y"]
            return x, y
        return None
    x, y = get_coordinates(streamlit_coordinates)
    print(x, y, height, width, grid_len)

    x_index = x // width
    y_index = y // height

    print(x_index, y_index)


    return x_index if y_index == 0 else (x_index + grid_len)



def draw_bounding_boxes(frame, player_detections):
    for player_id, bbox in player_detections.items():
            print(player_id, bbox)
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, f"ID: {player_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main():
    st.set_page_config(page_title="Web Application for Tennis Analysis", layout="wide",
                       initial_sidebar_state="expanded")
    st.title("Assignment for Deep Learning course: Tennis Analysis")

    st.sidebar.title("Main Settings")
    st.sidebar.markdown('---')
    st.sidebar.subheader("Video Upload")
    input_video_file = st.sidebar.file_uploader('Upload a video file', type=['mp4', 'mov', 'avi', 'm4v', 'asf'])

    tempf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

    # Get the directory of the current script file
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    video_path   = os.path.join(current_dir, 'input_videos', 'demo_video.mp4')
    video_path_2 = os.path.join(current_dir, 'input_videos', 'demo_video2.mp4')


    demo_selected = st.sidebar.radio(label="Select Demo Video", options=["Demo 1", "Demo 2"], horizontal=True)
    demo_vid_paths = {
        "Demo 1": video_path,
        "Demo 2": video_path_2,
    }

    demo_vid_path = demo_vid_paths[demo_selected]

    if not input_video_file:
        tempf.name = demo_vid_path
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()
        st.sidebar.text('Demo video')
        st.sidebar.video(demo_bytes)
    else:
        tempf.write(input_video_file.read())
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()
        st.sidebar.text('Input video')
        st.sidebar.video(demo_bytes)

    st.sidebar.markdown('---')

    # Main View
    # tab1, tab2, tab3 = st.tabs(['How to use', 'Player Stats', 'Player Analysis'])
    tab2, tab3, tab4  = st.tabs(['Detection Settings', 'Players Analysis', 'Pose Analysis'])
    # How to use tab
    # with tab1:
    #     st.header(':blue[Welcome!]')
    #     st.subheader('Main Application Functionalities:', divider='blue')
    #     st.markdown("""
    #                         1. Tennis players, tennis ball detection and tracking.
    #                         2. Tennis court's key-points detection and .
    #                         3. Estimation of players speed and distance covered.
    #                         4. Estimation of players and ball speed.
    #                         5. Estimation of player's shot (forehand, backhand).
    #                         6. Extract player statistics to Excel file.SS
    #                         """)
    #     st.subheader('How to use?', divider='blue')
    #     st.markdown("""
    #                         **There are two demo videos that are automatically loaded when you start the app, alongside the recommended settings and hyperparameters**
    #                         1. Upload a video to analyse, using the sidebar menu "Browse files" button.
    #                         2. Access the "Select Players" tab in the main page.
    #                         3. Select a frame where both players can be detected.
    #                         4. Follow the instruction on the page to select players
    #                         5. Go to the "Model Detection" tab, adjust hyperparameters and select the annotation options. (Default hyperparameters are recommended)
    #                         6. Run Detection!
    #                         7. If "save outputs" option was selected the saved video can be found in the "outputs" directory
    #                         """)

    # Players Detection Tab
    with tab2:
        st.header(':blue[Selecting Players]')
        t2col1, t2col2 = st.columns([1, 1])
        with t2col1:
            cap_temp = cv2.VideoCapture(tempf.name)
            frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_slide = st.slider(label="Select frame", min_value=0, max_value=frame_count-1, value=0,
                        help="Testing video's frames")

            cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_slide)
            ret, frame = cap_temp.read()

            with st.spinner('Detecting players in selected frame..'):
                player_tracker = PlayerTracker(model_path='models/yolov8x')
                player_detections = player_tracker.detect_frame(frame)

                detections_imgs_list = []
                detections_imgs_grid = []
                padding_img = np.ones((140, 140, 3), dtype=np.uint8) * 255

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_with_boxes = draw_bounding_boxes(frame, player_detections)

                for player_id, bbox in player_detections.items():
                    obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    obj_img = cv2.resize(obj_img, (140, 140))
                    detections_imgs_list.append([obj_img, player_id])

                detections_imgs_grid.append([detections_imgs_list[i][0] for i in range(len(detections_imgs_list) // 2)])
                detections_imgs_grid.append(
                    [detections_imgs_list[i][0] for i in
                     range((len(detections_imgs_list) // 2), len(detections_imgs_list))])

                if len(detections_imgs_list) % 2 != 0:
                    detections_imgs_grid[0].append(padding_img)
                concat_det_imgs_row1 = cv2.hconcat(detections_imgs_grid[0])
                concat_det_imgs_row2 = cv2.hconcat(detections_imgs_grid[1])
                concat_det_imgs = cv2.vconcat([concat_det_imgs_row1, concat_det_imgs_row2])
                # player_detections = player_tracker.choose_and_filter_players(court_keypoints=[0, 0, 0, 0], player_detections=player_detections)
            st.write(f"Detected {len(list(detections_imgs_list))} players")
            value = streamlit_image_coordinates(concat_det_imgs, key="numpy")

            st.markdown('---')
            auto_choose = st.checkbox(label='Auto Detect Tennis Players', value=True,
                                     help='Automatically detect tennis players. Otherwise,'
                                          'you can mannualy pick the players from the '
                                          'image above.')

            if not auto_choose:
                st.write('Please select the players from the image above')
                if 'active_player' not in st.session_state:
                    st.session_state.active_player = "Player 1"
                    st.session_state.selected_players = set()
                    st.session_state.filtered_player_detection = []

                radio_options = [f"Player 1", "Player 2"]
                st.session_state.active_player = st.radio(
                    label="Select which player to pick from the image above",
                    options=radio_options,
                    index=radio_options.index(st.session_state.active_player),
                    horizontal=True,
                    help="Choose the player you want to pick and click on the image above to pick the player."
                )
                active_player = st.session_state.active_player

                if value:
                    player_index = calculate_index(value, height=140, width=140, grid_len=len(detections_imgs_grid[0]))
                    st.write(f"Selected {active_player} with player index: {player_index}")
                    tmp_player_id = detections_imgs_list[player_index][1]
                    st.session_state.selected_players.add(tmp_player_id)


                if len(st.session_state.selected_players) == 2:
                    st.write(f"Selected all 2 players")
                    st.write('Please move to the next tab to start detection')
                    st.write(st.session_state.filtered_player_detection)
        with t2col2:
            extracted_frame = st.empty()
            extracted_frame.image(frame_with_boxes, use_column_width=True, channels="BGR")

    with tab3:
        st.header(':blue[Player Analysis]')
        bcol21t, bcol22t = st.columns([1, 1])
        with bcol21t:
            show_k = st.toggle(label="Show Keypoints Detections", value=True)
            show_p = st.toggle(label="Show Players Detections", value=True)
        with bcol22t:
            # show_pal = st.toggle(label="Show Color Palettes", value=True)
            show_b = st.toggle(label="Show Ball Tracks", value=True)
        plot_hyperparams = {
            0: show_k,
            # 1: show_pal,
            2: show_b,
            3: show_p
        }
        st.markdown('---')


        bcol21, bcol22, bcol23, bcol24 = st.columns([1.5, 1, 1, 1])
        with bcol21:
            st.write('')
        with bcol22:
            start_detection = st.button(label='Start Detection')
        with bcol23:
            stop_btn_state = True if not start_detection else False
            stop_detection = st.button(label='Stop Detection', disabled=stop_btn_state)
        with bcol24:
            st.write('')

        st_frame = st.empty()
        cap = cv2.VideoCapture(tempf.name)

        status = False
        if start_detection and not stop_detection:
            if not auto_choose:
                filtered_player_dct = {track_id: bbox for track_id, bbox in player_detections.items() if
                                       track_id in st.session_state.selected_players}

                st.session_state.filtered_player_detection.append(filtered_player_dct)
                st.write(st.session_state.filtered_player_detection)
                status, final_video = process_video(cap, save_vid=True,
                                                    selected_players=st.session_state.filtered_player_detection)
            else:
                status, final_video = process_video(cap, save_vid=True)
        else:
            try:
                cap.release()
            except:
                pass
            st.toast("Detection stopped!")

        if status:
            st.toast("Detection completed!")
            for frame in final_video:
                st_frame.image(frame, channels="BGR")
            cap.release()

    with tab4:
        st.header(':blue[Pose Analysis]')
        pose_frame = st.empty()
        cap = cv2.VideoCapture(tempf.name)
        bcol21, bcol22, bcol23, bcol24 = st.columns([1.5, 1, 1, 1])
        with bcol21:
            st.write('')
        with bcol22:
            start_pose = st.button(label='Start Pose Analysis')
        with bcol23:
            stop_btn_state = True if not start_detection else False
            stop_pose = st.button(label='Stop Pose Analysis', disabled=stop_btn_state)
        with bcol24:
            st.write('')

        if start_pose and not stop_pose:
            shot = ShotIdentification()
            opt = argparse.Namespace(
                device='0',
                line_thickness=3,
                poseweights='yolov7-w6-pose.pt',
                source=cap
            )
            shot.pose_analysis(opt)
            # cap = cv2.VideoCapture("finaltestvideo_wed.mp4")
            # final_video = read_video(cap)
            # for frame in final_video:
            #     pose_frame.image(frame, channels="BGR")
            # cap.release()

        st.write('Completed')


if __name__ == "__main__":
    main()